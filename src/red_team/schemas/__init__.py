"""Canonical entity and data schemas for Red Team AI."""

from red_team.schemas.entities import (
    Account,
    Beneficiary,
    Customer,
    Device,
    Merchant,
    Relationship,
    Session,
    Transaction,
)
from red_team.schemas.events import (
    AccountContextEventPayload,
    BeneficiaryEventPayload,
    DeviceEventPayload,
    Event,
    EventEnvelope,
    EventPayload,
    EventType,
    EVENTTYPE_TO_PAYLOAD_TYPE,
    RelationshipEventPayload,
    SessionEventPayload,
    TransactionEventPayload,
)
from red_team.schemas.ground_truth import (
    AttackGroundTruth,
    AttackPhaseRecord,
    EvaluationMetadata,
    GenerationMetadata,
    PlannerMetadata,
)
from red_team.schemas.observable import (
    ObservableAccountContextEvent,
    ObservableAttackTrace,
    ObservableBeneficiaryEvent,
    ObservableDeviceEvent,
    ObservableEvent,
    ObservableRelationshipEvent,
    ObservableSessionEvent,
    ObservableTransactionEvent,
    extract_observable,
)
from red_team.schemas.provenance import (
    CalibrationMode,
    DatasetSource,
    DatasetSourceType,
    FeatureProvenance,
    ProvenanceTier,
    ReferenceStatistic,
    VerificationStatus,
)
from red_team.schemas.calibration import (
    CalibrationDefinition,
    FeaturePairCalibration,
    FeatureType,
    MarginalCalibrationConfig,
    MetricType,
    ThresholdDirection,
)

__all__ = [
    # Entities (Stage 2.1)
    "Account",
    "Beneficiary",
    "Customer",
    "Device",
    "Merchant",
    "Relationship",
    "Session",
    "Transaction",
    # Events (Stage 2.2)
    "AccountContextEventPayload",
    "BeneficiaryEventPayload",
    "DeviceEventPayload",
    "Event",
    "EventEnvelope",
    "EventPayload",
    "EventType",
    "EVENTTYPE_TO_PAYLOAD_TYPE",
    "RelationshipEventPayload",
    "SessionEventPayload",
    "TransactionEventPayload",
    # Observable (Stage 2.3)
    "ObservableAccountContextEvent",
    "ObservableAttackTrace",
    "ObservableBeneficiaryEvent",
    "ObservableDeviceEvent",
    "ObservableEvent",
    "ObservableRelationshipEvent",
    "ObservableSessionEvent",
    "ObservableTransactionEvent",
    "extract_observable",
    # Provenance (Stage 2.4)
    "CalibrationMode",
    "DatasetSource",
    "DatasetSourceType",
    "FeatureProvenance",
    "ProvenanceTier",
    "ReferenceStatistic",
    "VerificationStatus",
    # Calibration (Stage 2.5)
    "CalibrationDefinition",
    "FeaturePairCalibration",
    "FeatureType",
    "MarginalCalibrationConfig",
    "MetricType",
    "ThresholdDirection",
    # Ground Truth (Stage 2.3)
    "AttackGroundTruth",
    "AttackPhaseRecord",
    "EvaluationMetadata",
    "GenerationMetadata",
    "PlannerMetadata",
]
