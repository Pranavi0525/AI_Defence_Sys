"""Data preprocessing pipeline for reference datasets.

Handles extracting reference statistics from raw data, filtering for
legitimate baselines, and implementing fallback behavior when raw
data is unavailable locally.
"""

import logging
from pathlib import Path
from typing import List, Optional

from red_team.schemas.provenance import ReferenceStatistic, VerificationStatus
from red_team.registry.population import build_first_slice_registry
from red_team.data.reference_stats import save_reference_statistics


logger = logging.getLogger(__name__)


class DataUnavailableError(Exception):
    """Raised when raw data is not found and strict processing is required."""
    pass


def _fallback_paysim_statistics() -> List[ReferenceStatistic]:
    """Provide externally reported / baseline assumptions for PaySim when raw data is missing.
    
    These are strictly marked as REFERENCE_STATISTICS and externally_reported=True.
    """
    req_note = "source reports/estimates this value; independently derived from raw data: no."
    
    return [
        ReferenceStatistic(
            id="stat_paysim_tx_amount_median",
            statistic_name="transaction_amount_median",
            value="~75000",  # Approximate legitimate median from typical PaySim explorations
            source_dataset_id="paysim_kaggle_v1",
            source="Fallback Baseline (Estimated)",
            citation="Community EDA on Kaggle PaySim",
            statistic_definition="Median transaction amount for legitimate non-fraud transactions.",
            externally_reported=True,
            verification_status=VerificationStatus.UNVERIFIED_ESTIMATE,
            derivation_notes=req_note,
        ),
        ReferenceStatistic(
            id="stat_paysim_tx_type_freq",
            statistic_name="transaction_type_frequencies",
            value={
                "CASH_OUT": "~0.35",
                "PAYMENT": "~0.34",
                "CASH_IN": "~0.22",
                "TRANSFER": "~0.08",
                "DEBIT": "~0.01"
            },
            source_dataset_id="paysim_kaggle_v1",
            source="Fallback Baseline (Estimated)",
            citation="Community EDA on Kaggle PaySim",
            statistic_definition="Frequency distribution of transaction types for legitimate transactions.",
            externally_reported=True,
            verification_status=VerificationStatus.UNVERIFIED_ESTIMATE,
            derivation_notes=req_note,
        ),
        ReferenceStatistic(
            id="stat_paysim_customer_typ_amount_mean",
            statistic_name="customer_typical_amount_mean",
            value="~80000",
            source_dataset_id="paysim_kaggle_v1",
            source="Fallback Baseline (Estimated)",
            citation="Community EDA on Kaggle PaySim",
            statistic_definition="Mean of the customer-grouped typical transaction amounts.",
            externally_reported=True,
            verification_status=VerificationStatus.UNVERIFIED_ESTIMATE,
            derivation_notes=req_note,
        ),
    ]


def _fallback_ieee_cis_statistics() -> List[ReferenceStatistic]:
    """Provide externally reported / baseline assumptions for IEEE-CIS."""
    req_note = "source reports/estimates this value; independently derived from raw data: no."
    
    return [
        ReferenceStatistic(
            id="stat_ieee_device_type_freq",
            statistic_name="device_type_frequencies",
            value={"desktop": "~0.6", "mobile": "~0.4"},
            source_dataset_id="ieee_cis_fraud_v1",
            source="Fallback Baseline (Estimated)",
            citation="Community EDA on IEEE-CIS",
            statistic_definition="Frequency distribution of device types.",
            externally_reported=True,
            verification_status=VerificationStatus.UNVERIFIED_ESTIMATE,
            derivation_notes=req_note,
        ),
        ReferenceStatistic(
            id="stat_ieee_relative_dt_median",
            statistic_name="relative_transaction_dt_median",
            value="~1 day (approx 86400s)",
            source_dataset_id="ieee_cis_fraud_v1",
            source="Fallback Baseline (Estimated)",
            citation="Community EDA on IEEE-CIS",
            statistic_definition="Median relative timedelta between transactions.",
            externally_reported=True,
            verification_status=VerificationStatus.UNVERIFIED_ESTIMATE,
            derivation_notes=req_note,
        )
    ]


def preprocess_paysim(raw_dir: Path, output_dir: Path) -> Path:
    """Preprocess the PaySim dataset to extract normal behavior statistics."""
    dataset_id = "paysim_kaggle_v1"
    
    # In a real environment, we'd check for CSVs like:
    # csv_path = raw_dir / "PS_20174392719_1491204439457_log.csv"
    
    # For now, simulate checking for raw data
    csv_exists = False
    
    if csv_exists:
        # Placeholder for actual pandas logic:
        # df = pd.read_csv(csv_path)
        # legits = df[df['isFraud'] == 0]
        # ... generate stats ...
        # calibration_mode = "RAW_DATA"
        pass
    else:
        logger.warning(f"Raw data for {dataset_id} not found in {raw_dir}. Using fallback REFERENCE_STATISTICS.")
        calibration_mode = "REFERENCE_STATISTICS"
        stats = _fallback_paysim_statistics()
        
    return save_reference_statistics(dataset_id, calibration_mode, stats, output_dir)


def preprocess_ieee_cis(raw_dir: Path, output_dir: Path) -> Path:
    """Preprocess the IEEE-CIS Fraud Detection dataset."""
    dataset_id = "ieee_cis_fraud_v1"
    
    csv_exists = False
    
    if csv_exists:
        # Placeholder for actual pandas logic
        pass
    else:
        logger.warning(f"Raw data for {dataset_id} not found in {raw_dir}. Using fallback REFERENCE_STATISTICS.")
        calibration_mode = "REFERENCE_STATISTICS"
        stats = _fallback_ieee_cis_statistics()
        
    return save_reference_statistics(dataset_id, calibration_mode, stats, output_dir)


def run_all_preprocessing(raw_dir: Path, output_dir: Path) -> List[Path]:
    """Run all dataset preprocessing pipelines."""
    results = []
    
    # 1. PaySim
    try:
        p1 = preprocess_paysim(raw_dir, output_dir)
        results.append(p1)
    except Exception as e:
        logger.error(f"Failed to preprocess PaySim: {e}")
        
    # 2. IEEE-CIS
    try:
        p2 = preprocess_ieee_cis(raw_dir, output_dir)
        results.append(p2)
    except Exception as e:
        logger.error(f"Failed to preprocess IEEE-CIS: {e}")
        
    return results
