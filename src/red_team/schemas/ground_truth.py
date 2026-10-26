"""Attack ground-truth schemas — internal evaluation only.

Contains metadata about attack generation that must NEVER be exposed to
the Blue Team. This includes attack family, phases, objectives, planner
metadata, generation seeds, and evaluation scores.

Design decisions:
    - Completely separate from observable.py — no shared base class.
    - Ground truth references events by ID only (linked_event_ids),
      never duplicates observable event payloads.
    - EvaluationMetadata is minimal for now; realism/novelty scores
      will be added in later stages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# NOTE: No imports from observable.py — these models are fully independent.


# ---------------------------------------------------------------------------
# Supporting internal models
# ---------------------------------------------------------------------------

class AttackPhaseRecord(BaseModel):
    """Record of a single phase executed during an attack simulation."""

    phase: str = Field(..., min_length=1, description="Phase name (e.g., 'reconnaissance').")
    entered_at: datetime = Field(..., description="When the simulator entered this phase.")
    exited_at: datetime | None = Field(
        default=None, description="When the simulator exited this phase (None if still active).",
    )
    transition_to: str | None = Field(
        default=None, description="Next phase transitioned to (None if terminal).",
    )
    was_optional: bool = Field(..., description="Whether this phase was optional in the signature.")


class GenerationMetadata(BaseModel):
    """Reproducibility metadata for a generated attack."""

    random_seed: int = Field(..., description="RNG seed used for generation.")
    generator_version: str = Field(..., min_length=1, description="Simulator version.")
    signature_version: str = Field(..., min_length=1, description="Attack signature version.")
    provenance_registry_version: str = Field(
        ..., min_length=1, description="Provenance registry version.",
    )
    configuration_hash: str = Field(..., min_length=1, description="Hash of generation config.")
    generated_at: datetime = Field(..., description="When this attack was generated.")


class PlannerMetadata(BaseModel):
    """Metadata about the attack planner that composed the scenario."""

    planner_type: str = Field(
        ..., min_length=1,
        description="Planner implementation used (e.g., 'mock', 'gemini').",
    )
    plan_json: dict = Field(
        ..., description="The structured attack plan as produced by the planner.",
    )
    model_version: str | None = Field(
        default=None, description="LLM model version (None for mock planner).",
    )
    prompt_hash: str | None = Field(
        default=None, description="Hash of the prompt sent to the LLM (None for mock).",
    )


class EvaluationMetadata(BaseModel):
    """Evaluation results for a generated attack.

    Minimal for Stage 2.3. Realism and novelty scores will be
    extended in later stages (Stages 11, 13, 15).
    """

    structural_valid: bool = Field(
        ..., description="Whether the trace passed structural validation.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Reason for rejection (None if accepted).",
    )


# ---------------------------------------------------------------------------
# Attack Ground Truth — internal evaluation artifact
# ---------------------------------------------------------------------------

class AttackGroundTruth(BaseModel):
    """Internal ground truth for a generated attack.

    Used ONLY for:
        - evaluation
        - experiment tracking
        - training-data construction
        - validation
        - future Blue-Team feedback

    NEVER exposed through the Blue Team interface.
    NEVER accessible via HTTP API.
    """

    attack_id: str = Field(..., min_length=1, description="Unique attack identifier.")
    attack_family: str = Field(
        ..., min_length=1,
        description="Attack family (e.g., 'account_takeover').",
    )
    attack_difficulty: Literal["easy", "medium", "hard", "advanced"] = Field(
        ..., description="Intended difficulty level.",
    )
    hidden_objective: str = Field(
        ..., min_length=1,
        description="What the simulated attacker aims to achieve.",
    )
    phases_executed: list[AttackPhaseRecord] = Field(
        ..., description="Ordered list of attack phases that were executed.",
    )
    linked_event_ids: list[str] = Field(
        ...,
        description="Event IDs (references only — no observable content duplication).",
    )
    generation_metadata: GenerationMetadata = Field(
        ..., description="Reproducibility metadata.",
    )
    planner_metadata: PlannerMetadata = Field(
        ..., description="Attack planner metadata.",
    )
    evaluation_metadata: EvaluationMetadata | None = Field(
        default=None, description="Evaluation results (None until validation runs).",
    )
