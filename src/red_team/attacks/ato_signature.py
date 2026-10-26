"""Account Takeover (ATO) Attack Signature.

Provides the constrained state graph describing ATO behavior based on 
published research and typologies, not deterministic scripts.
"""

from red_team.attacks.signature_library import (
    AttackSignature,
    AttackState,
    AttackTransition,
    ObservableConsequence,
    Observability,
    SignalFamily,
    VariationAxis,
    AttackConstraint,
    ResearchSource,
)


def get_ato_signature() -> AttackSignature:
    """Return the static, validated ATO Attack Signature."""
    
    return AttackSignature(
        attack_family="ACCOUNT_TAKEOVER",
        version="1.0",
        description=(
            "Account Takeover involving progression from initial access "
            "through modification and eventual exploitation. Describes plausible "
            "observable behaviors supported by industry research."
        ),
        entry_states=["RECONNAISSANCE", "ACCOUNT_ACCESS"],
        states={
            "RECONNAISSANCE": AttackState(
                state_name="RECONNAISSANCE",
                description="Attacker probes the account or authentication mechanisms.",
                affected_entities=["session", "customer", "device"],
                observable_consequences=[
                    ObservableConsequence(
                        description="Unusual failed authentication or session activity.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.DEVICE_SESSION, SignalFamily.BEHAVIORAL],
                        affected_entities=["session"]
                    ),
                    ObservableConsequence(
                        description="Abnormal access patterns (e.g., brute force timing).",
                        observability=Observability.INDIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.VELOCITY],
                        affected_entities=["session"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="ACCOUNT_ACCESS",
                        min_weight=0.1, max_weight=0.8,
                        reason="Successful credential compromise."
                    ),
                    AttackTransition(
                        target_state="RECONNAISSANCE",
                        min_weight=0.1, max_weight=0.5,
                        reason="Continued probing after failures."
                    ),
                    AttackTransition(
                        target_state="END",
                        min_weight=0.1, max_weight=0.9,
                        reason="Attacker abandons attempt."
                    )
                ]
            ),
            "ACCOUNT_ACCESS": AttackState(
                state_name="ACCOUNT_ACCESS",
                description="Attacker successfully logs into the account.",
                affected_entities=["session", "device", "customer"],
                observable_consequences=[
                    ObservableConsequence(
                        description="New or unrecognized device context.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.DEVICE_SESSION, SignalFamily.CONTEXT],
                        affected_entities=["device", "session"]
                    ),
                    ObservableConsequence(
                        description="Unusual geographical location or network context.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.CONTEXT],
                        affected_entities=["session"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="ACCOUNT_MODIFICATION",
                        min_weight=0.2, max_weight=0.7,
                        reason="Attacker alters account settings for persistence/monetization."
                    ),
                    AttackTransition(
                        target_state="EXPLOITATION",
                        min_weight=0.3, max_weight=0.9,
                        reason="Direct exploitation without modifying profile."
                    ),
                    AttackTransition(
                        target_state="PERSISTENCE",
                        min_weight=0.1, max_weight=0.4,
                        reason="Attacker establishes persistence before exploiting."
                    ),
                    AttackTransition(
                        target_state="END",
                        min_weight=0.0, max_weight=0.2,
                        reason="Session terminates without malicious action."
                    )
                ]
            ),
            "ACCOUNT_MODIFICATION": AttackState(
                state_name="ACCOUNT_MODIFICATION",
                description="Attacker changes account details (e.g. password, email, beneficiaries).",
                affected_entities=["customer", "beneficiary", "relationship"],
                observable_consequences=[
                    ObservableConsequence(
                        description="Security setting or contact profile change.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.BEHAVIORAL],
                        affected_entities=["customer"]
                    ),
                    ObservableConsequence(
                        description="New beneficiary added unexpectedly.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.BENEFICIARY, SignalFamily.RELATIONSHIP],
                        affected_entities=["beneficiary", "relationship"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="EXPLOITATION",
                        min_weight=0.5, max_weight=0.9,
                        reason="Immediate cash out following modification."
                    ),
                    AttackTransition(
                        target_state="PERSISTENCE",
                        min_weight=0.1, max_weight=0.5,
                        reason="Setting up long-term access."
                    ),
                    AttackTransition(
                        target_state="ACCOUNT_ACCESS",
                        min_weight=0.0, max_weight=0.3,
                        reason="Further lateral movement within the account."
                    ),
                    AttackTransition(
                        target_state="END",
                        min_weight=0.0, max_weight=0.2,
                        reason="Attacker pauses operation."
                    )
                ]
            ),
            "EXPLOITATION": AttackState(
                state_name="EXPLOITATION",
                description="Attacker extracts value (e.g., transfers, purchases).",
                affected_entities=["account", "transaction", "beneficiary", "merchant"],
                observable_consequences=[
                    ObservableConsequence(
                        description="Unusual transaction amount or frequency.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.TRANSACTION, SignalFamily.VELOCITY],
                        affected_entities=["transaction", "account"]
                    ),
                    ObservableConsequence(
                        description="Unusual beneficiary or merchant interaction.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.RELATIONSHIP, SignalFamily.BEHAVIORAL],
                        affected_entities=["beneficiary", "merchant"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="EXPLOITATION",
                        min_weight=0.2, max_weight=0.8,
                        reason="Multiple rapid cash-out transactions."
                    ),
                    AttackTransition(
                        target_state="PERSISTENCE",
                        min_weight=0.1, max_weight=0.4,
                        reason="Switching back to maintain access."
                    ),
                    AttackTransition(
                        target_state="END",
                        min_weight=0.5, max_weight=1.0,
                        reason="Funds exhausted or operation complete."
                    )
                ]
            ),
            "PERSISTENCE": AttackState(
                state_name="PERSISTENCE",
                description="Attacker reinforces access (e.g., registering additional trusted devices).",
                affected_entities=["device", "relationship", "customer"],
                observable_consequences=[
                    ObservableConsequence(
                        description="Additional device registration.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.DEVICE_SESSION, SignalFamily.RELATIONSHIP],
                        affected_entities=["device"]
                    ),
                    ObservableConsequence(
                        description="Subtle, ongoing context changes to blend in.",
                        observability=Observability.POTENTIALLY_UNOBSERVABLE,
                        signal_families=[SignalFamily.CONTEXT],
                        affected_entities=["customer"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="EXPLOITATION",
                        min_weight=0.3, max_weight=0.8,
                        reason="Returning to extract value."
                    ),
                    AttackTransition(
                        target_state="ACCOUNT_MODIFICATION",
                        min_weight=0.1, max_weight=0.5,
                        reason="Further tuning of account settings."
                    ),
                    AttackTransition(
                        target_state="END",
                        min_weight=0.1, max_weight=0.5,
                        reason="Entering dormancy."
                    )
                ]
            )
        },
        variation_axes=[
            VariationAxis(
                name="ENTRY_PATH",
                description="The initial state the attack becomes observable.",
                allowed_values=["RECONNAISSANCE", "ACCOUNT_ACCESS"],
                reason="Not all probing is visible; some attacks use previously stolen session tokens."
            ),
            VariationAxis(
                name="PHASE_SKIPPING",
                description="Whether intermediate phases like MODIFICATION are skipped.",
                allowed_values=["TRUE", "FALSE"],
                reason="Fast cash-out actors skip modification; stealth actors rely on it."
            ),
            VariationAxis(
                name="PATH_LENGTH",
                description="Number of state transitions before END.",
                allowed_values=["SHORT", "MEDIUM", "LONG"],
                reason="Determines the volume of observable consequences."
            ),
            VariationAxis(
                name="LOOPING",
                description="Whether states like EXPLOITATION repeat.",
                allowed_values=["TRUE", "FALSE"],
                reason="Some attackers drain in one burst; others make multiple small transfers."
            ),
            VariationAxis(
                name="TIMING_SCALE",
                description="Speed of the attack progression.",
                allowed_values=["RAPID", "STRETCHED"],
                reason="Differentiates automated scripts from manual operators."
            ),
            VariationAxis(
                name="SIGNAL_INTENSITY",
                description="Loudness of the observable consequences.",
                allowed_values=["LOW", "HIGH"],
                reason="Sophisticated actors attempt to mimic legitimate baselines."
            ),
            VariationAxis(
                name="AFFECTED_ENTITY_SET",
                description="Scope of entities impacted.",
                allowed_values=["MINIMAL", "BROAD"],
                reason="Varies depending on whether just an account or also connected identities are targeted."
            ),
        ],
        constraints=[
            AttackConstraint(
                description="Beneficiary must exist before a transaction references it.",
                enforcement_layer="SIMULATOR"
            ),
            AttackConstraint(
                description="Events must remain chronologically ordered.",
                enforcement_layer="WORLD_STATE"
            ),
            AttackConstraint(
                description="Device/session changes must be temporally coherent.",
                enforcement_layer="SIMULATOR"
            ),
            AttackConstraint(
                description="Transactions must reference valid accounts.",
                enforcement_layer="WORLD_STATE"
            ),
            AttackConstraint(
                description="Transaction amounts must satisfy world/account constraints.",
                enforcement_layer="SIMULATOR"
            ),
            AttackConstraint(
                description="Relationships must reference valid entities.",
                enforcement_layer="WORLD_STATE"
            ),
            AttackConstraint(
                description="Phase transitions must produce corresponding observable consequences.",
                enforcement_layer="SIMULATOR"
            ),
        ],
        research_sources=[
            ResearchSource(
                source_name="FinCEN",
                title="Advisory on Account Takeover Activity",
                publication_year=2021,
                relevant_claim="Describes rapid phase skipping and immediate exploitation using stolen credentials."
            ),
            ResearchSource(
                source_name="FATF",
                title="Money Laundering and Terrorist Financing through Trade in Diamonds",  # Place holder generic typology
                publication_year=2013,
                relevant_claim="Noted persistence mechanisms and modification of beneficiary lists prior to large transfers."
            )
        ]
    )
