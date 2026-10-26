import hashlib
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from decimal import Decimal
import statistics

from red_team.schemas.observable import ObservableAttackTrace
from red_team.schemas.ground_truth import AttackGroundTruth
from red_team.world.state import WorldState

class BaseAttackFingerprint(BaseModel):
    attack_family: str

class ATOAttackFingerprint(BaseAttackFingerprint):
    """Normalized structural representation of an ATO attack."""
    phase_sequence: Tuple[str, ...]
    event_sequence: Tuple[str, ...]
    transaction_count: int
    amount_buckets: Tuple[str, ...]  
    normalized_amount_sum: float  
    split_count: int
    timing_category: str  
    device_pattern: Tuple[str, ...]  
    beneficiary_pattern: Tuple[str, ...]  
    outcome_pattern: Tuple[str, ...]

    def __hash__(self):
        return hash((
            self.attack_family, self.phase_sequence, self.event_sequence, self.transaction_count,
            self.amount_buckets, round(self.normalized_amount_sum, 1),
            self.split_count, self.timing_category,
            self.device_pattern, self.beneficiary_pattern, self.outcome_pattern
        ))

class APPAttackFingerprint(BaseAttackFingerprint):
    """Normalized structural representation of an APP attack."""
    phase_sequence: Tuple[str, ...]
    event_sequence: Tuple[str, ...]
    transaction_count: int
    amount_buckets: Tuple[str, ...]
    normalized_amount_sum: float
    split_count: int
    timing_category: str
    device_continuity: bool
    session_continuity: bool
    beneficiary_novelty: str # "new", "reused", "none"
    outcome_pattern: Tuple[str, ...]
    hesitation_category: str
    amount_trend: str

    def __hash__(self):
        return hash((
            self.attack_family, self.phase_sequence, self.event_sequence, self.transaction_count,
            self.amount_buckets, round(self.normalized_amount_sum, 1),
            self.split_count, self.timing_category,
            self.device_continuity, self.session_continuity, self.beneficiary_novelty,
            self.outcome_pattern, self.hesitation_category, self.amount_trend
        ))

# Type alias for backward compatibility in internal type hints
AttackFingerprint = BaseAttackFingerprint

def extract_fingerprint(trace: ObservableAttackTrace, gt: AttackGroundTruth, state: WorldState) -> BaseAttackFingerprint:
    phase_seq = tuple(p.phase for p in gt.phases_executed)
    event_seq = tuple(e.event_type for e in trace.events)
    
    txs = [e for e in trace.events if e.event_type == "TRANSACTION"]
    tx_count = len(txs)
    
    amount_buckets = []
    normalized_sum = 0.0
    for tx in txs:
        amt = float(getattr(tx, "amount", 0))
        acct_id = getattr(tx, "account_id", None)
        acct_bal = float(state.accounts[acct_id].balance) if acct_id in state.accounts and state.accounts[acct_id].balance > 0 else 100.0
        ratio = amt / acct_bal
        normalized_sum += ratio
        if ratio < 0.3:
            amount_buckets.append("small")
        elif ratio < 0.7:
            amount_buckets.append("medium")
        else:
            amount_buckets.append("large")
            
    split_count = tx_count if tx_count > 0 else 0
    
    timestamps = [e.timestamp for e in trace.events]
    gaps = []
    for i in range(1, len(timestamps)):
        gaps.append((timestamps[i] - timestamps[i-1]).total_seconds())
        
    if not gaps:
        timing_cat = "none"
    else:
        max_gap = max(gaps)
        if max_gap > 3600 * 12:
            if min(gaps) < 600:
                timing_cat = "bursty"
            else:
                timing_cat = "slow"
        elif max_gap > 1200:
            timing_cat = "normal"
        else:
            timing_cat = "rapid"
            
    outcomes = tuple(getattr(tx, "transaction_status", "none") for tx in txs)

    if gt.attack_family == "AUTHORIZED_PUSH_PAYMENT":
        dev_continuity = True
        session_continuity = True
        ben_novelty = "none"
        session_ids = set()
        
        for e in trace.events:
            if e.event_type == "DEVICE_REGISTRATION":
                dev_continuity = False
            elif e.event_type == "SESSION_LOGIN":
                if getattr(e, "action", "") == "register":
                    dev_continuity = False
                session_ids.add(getattr(e, "session_id", ""))
            elif e.event_type == "BENEFICIARY_ADDITION":
                ben_novelty = "new"
                
        if len(session_ids) > 1:
            session_continuity = False
            
        if ben_novelty == "none" and tx_count > 0:
            ben_novelty = "reused"

        # Extract hesitation gap
        hesitation = "immediate"
        ben_time = None
        tx_time = None
        for e in trace.events:
            if e.event_type == "BENEFICIARY_ADDITION":
                ben_time = e.timestamp
            elif e.event_type == "TRANSACTION" and ben_time and not tx_time:
                tx_time = e.timestamp
                
        if ben_time and tx_time:
            gap_seconds = (tx_time - ben_time).total_seconds()
            if gap_seconds < 600:
                hesitation = "immediate"
            elif gap_seconds < 3600:
                hesitation = "hesitant"
            else:
                hesitation = "delayed"

        # Extract amount trend
        trend = "single"
        if tx_count > 1:
            amts = [float(getattr(tx, "amount", 0)) for tx in txs if getattr(tx, "transaction_status", "") == "completed"]
            if len(amts) > 1:
                if all(amts[i] <= amts[i+1] for i in range(len(amts)-1)) and amts[0] < amts[-1]:
                    trend = "escalating"
                elif all(amts[i] >= amts[i+1] for i in range(len(amts)-1)) and amts[0] > amts[-1]:
                    trend = "decreasing"
                elif len(set(amts)) == 1:
                    trend = "fragmented"
                else:
                    trend = "random"

        return APPAttackFingerprint(
            attack_family=gt.attack_family,
            phase_sequence=phase_seq,
            event_sequence=event_seq,
            transaction_count=tx_count,
            amount_buckets=tuple(amount_buckets),
            normalized_amount_sum=normalized_sum,
            split_count=split_count,
            timing_category=timing_cat,
            device_continuity=dev_continuity,
            session_continuity=session_continuity,
            beneficiary_novelty=ben_novelty,
            outcome_pattern=outcomes,
            hesitation_category=hesitation,
            amount_trend=trend
        )
    else:
        # Default to ATO behavior
        dev_pat = []
        for e in trace.events:
            if e.event_type in ("SESSION_LOGIN", "DEVICE_REGISTRATION"):
                action = getattr(e, "action", "")
                if action == "register" or e.event_type == "DEVICE_REGISTRATION":
                    dev_pat.append("new")
                else:
                    dev_pat.append("known")
                    
        ben_pat = []
        for e in trace.events:
            if e.event_type == "BENEFICIARY_ADDITION":
                ben_pat.append("new")
            elif e.event_type == "TRANSACTION" and getattr(e, "beneficiary_id", None):
                ben_pat.append("known")
                
        return ATOAttackFingerprint(
            attack_family="ACCOUNT_TAKEOVER",
            phase_sequence=phase_seq,
            event_sequence=event_seq,
            transaction_count=tx_count,
            amount_buckets=tuple(amount_buckets),
            normalized_amount_sum=normalized_sum,
            split_count=split_count,
            timing_category=timing_cat,
            device_pattern=tuple(dev_pat),
            beneficiary_pattern=tuple(ben_pat),
            outcome_pattern=outcomes
        )

def calculate_fingerprint_similarity(f1: BaseAttackFingerprint, f2: BaseAttackFingerprint) -> float:
    if f1.attack_family != f2.attack_family:
        return 0.0

    score = 0.0
    if f1.attack_family == "AUTHORIZED_PUSH_PAYMENT":
        weights = {
            "phase": 0.15,
            "event": 0.1,
            "amount": 0.2, 
            "split": 0.05,
            "timing": 0.15,
            "device_cont": 0.1,
            "session_cont": 0.1,
            "beneficiary_nov": 0.05,
            "outcome": 0.1
        }
        if f1.phase_sequence == f2.phase_sequence: score += weights["phase"]
        elif len(set(f1.phase_sequence) & set(f2.phase_sequence)) > 0: score += weights["phase"] * 0.5
        
        if f1.event_sequence == f2.event_sequence: score += weights["event"]
        
        if f1.amount_buckets == f2.amount_buckets and abs(f1.normalized_amount_sum - f2.normalized_amount_sum) < 0.2:
            score += weights["amount"]
        elif f1.amount_buckets == f2.amount_buckets:
            score += weights["amount"] * 0.8
            
        if f1.split_count == f2.split_count: score += weights["split"]
        elif abs(f1.split_count - f2.split_count) == 1: score += weights["split"] * 0.5
        
        if f1.timing_category == f2.timing_category: score += weights["timing"]
        if getattr(f1, "device_continuity") == getattr(f2, "device_continuity"): score += weights["device_cont"]
        if getattr(f1, "session_continuity") == getattr(f2, "session_continuity"): score += weights["session_cont"]
        if getattr(f1, "beneficiary_novelty") == getattr(f2, "beneficiary_novelty"): score += weights["beneficiary_nov"]
        if f1.outcome_pattern == f2.outcome_pattern: score += weights["outcome"]
        return score
    else:
        weights = {
            "phase": 0.2,
            "event": 0.15,
            "amount": 0.15,
            "split": 0.05,
            "timing": 0.15,
            "device": 0.1,
            "beneficiary": 0.1,
            "outcome": 0.1
        }
        
        if f1.phase_sequence == f2.phase_sequence: score += weights["phase"]
        elif len(set(f1.phase_sequence) & set(f2.phase_sequence)) > 0: score += weights["phase"] * 0.5
            
        if f1.event_sequence == f2.event_sequence: score += weights["event"]
            
        if f1.amount_buckets == f2.amount_buckets and abs(f1.normalized_amount_sum - f2.normalized_amount_sum) < 0.2:
            score += weights["amount"]
        elif f1.amount_buckets == f2.amount_buckets:
            score += weights["amount"] * 0.8
            
        if f1.split_count == f2.split_count: score += weights["split"]
        elif abs(f1.split_count - f2.split_count) == 1: score += weights["split"] * 0.5
            
        if f1.timing_category == f2.timing_category: score += weights["timing"]
        if getattr(f1, "device_pattern") == getattr(f2, "device_pattern"): score += weights["device"]
        if getattr(f1, "beneficiary_pattern") == getattr(f2, "beneficiary_pattern"): score += weights["beneficiary"]
        if f1.outcome_pattern == f2.outcome_pattern: score += weights["outcome"]
            
        return score

class NoveltyResult(BaseModel):
    is_novel: bool
    novelty_score: float  # 0.0 to 1.0
    similarity_to_closest: float
    closest_match_index: Optional[int]
    rejection_reason: Optional[str] = None

class NoveltyIndex:
    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        # Dictionary of family -> difficulty -> list of fingerprints
        self.fingerprints: Dict[str, Dict[str, List[BaseAttackFingerprint]]] = {
            "ACCOUNT_TAKEOVER": {"easy": [], "medium": [], "hard": [], "advanced": []},
            "AUTHORIZED_PUSH_PAYMENT": {"easy": [], "medium": [], "hard": [], "advanced": []}
        }
        
    def evaluate(self, fp: BaseAttackFingerprint, difficulty: str) -> NoveltyResult:
        family_buckets = self.fingerprints.setdefault(fp.attack_family, {})
        bucket = family_buckets.setdefault(difficulty, [])
        if not bucket:
            return NoveltyResult(is_novel=True, novelty_score=1.0, similarity_to_closest=0.0, closest_match_index=None)
            
        max_sim = 0.0
        closest_idx = -1
        for i, existing in enumerate(bucket):
            sim = calculate_fingerprint_similarity(fp, existing)
            if sim > max_sim:
                max_sim = sim
                closest_idx = i
                
        novelty_score = 1.0 - max_sim
        is_novel = max_sim < self.similarity_threshold
        
        reason = None
        if not is_novel:
            reason = f"Too similar to existing trace in {fp.attack_family}/{difficulty} bucket (similarity: {max_sim:.2f})"
            
        return NoveltyResult(
            is_novel=is_novel,
            novelty_score=novelty_score,
            similarity_to_closest=max_sim,
            closest_match_index=closest_idx,
            rejection_reason=reason
        )
        
    def add(self, fp: BaseAttackFingerprint, difficulty: str):
        family_buckets = self.fingerprints.setdefault(fp.attack_family, {})
        bucket = family_buckets.setdefault(difficulty, [])
        if len(bucket) >= self.max_size:
            bucket.pop(0)
        bucket.append(fp)
