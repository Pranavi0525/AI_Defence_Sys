from red_team.attacks.signature_library import (
    AttackSignature,
    AttackState,
    AttackTransition,
    ObservableConsequence,
    Observability,
    SignalFamily,
    VariationAxis,
    AttackConstraint,
    ResearchSource
)

def get_app_signature() -> AttackSignature:
    """Return the static, validated APP Attack Signature."""
    
    return AttackSignature(
        attack_family="AUTHORIZED_PUSH_PAYMENT",
        version="1.0",
        description=(
            "Authorized Push Payment (APP) fraud involving a socially engineered "
            "customer executing a transfer to an attacker-controlled account from "
            "their own established device/session."
        ),
        entry_states=["SOCIAL_ENGINEERING"],
        states={
            "SOCIAL_ENGINEERING": AttackState(
                state_name="SOCIAL_ENGINEERING",
                description="Attacker deceives or coerces the victim off-platform (e.g., phone call).",
                observable_consequences=[
                    ObservableConsequence(
                        description="Off-platform communication (phone, SMS, email).",
                        observability=Observability.POTENTIALLY_UNOBSERVABLE,
                        signal_families=[SignalFamily.CONTEXT],
                        affected_entities=["customer"]
                    )
                ],
                transitions=[
                    AttackTransition(target_state="SESSION_ACTIVE", min_weight=0.8, max_weight=1.0, reason="Victim complies and logs in."),
                    AttackTransition(target_state="END", min_weight=0.0, max_weight=0.2, reason="Victim gets suspicious and aborts.")
                ]
            ),
            "SESSION_ACTIVE": AttackState(
                state_name="SESSION_ACTIVE",
                description="Victim logs into their own account from their known primary device.",
                observable_consequences=[
                    ObservableConsequence(
                        description="Standard session login from primary device.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.DEVICE_SESSION],
                        affected_entities=["customer", "device"]
                    )
                ],
                transitions=[
                    AttackTransition(target_state="BENEFICIARY_SETUP", min_weight=0.5, max_weight=0.9, reason="Victim adds new attacker-controlled beneficiary."),
                    AttackTransition(target_state="PAYMENT_EXECUTION", min_weight=0.1, max_weight=0.5, reason="Victim pays an existing trusted beneficiary that attacker somehow controls/impersonates.")
                ]
            ),
            "BENEFICIARY_SETUP": AttackState(
                state_name="BENEFICIARY_SETUP",
                description="Victim adds the attacker's account as a beneficiary.",
                observable_consequences=[
                    ObservableConsequence(
                        description="New beneficiary addition.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.BENEFICIARY, SignalFamily.RELATIONSHIP],
                        affected_entities=["customer", "beneficiary"]
                    )
                ],
                transitions=[
                    AttackTransition(target_state="PAYMENT_EXECUTION", min_weight=0.9, max_weight=1.0, reason="Victim proceeds to payment."),
                    AttackTransition(target_state="END", min_weight=0.0, max_weight=0.1, reason="Victim abandons before payment.")
                ]
            ),
            "PAYMENT_EXECUTION": AttackState(
                state_name="PAYMENT_EXECUTION",
                description="Victim authorizes the push payment to the attacker.",
                observable_consequences=[
                    ObservableConsequence(
                        description="High-value or unusual transaction.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.TRANSACTION, SignalFamily.VELOCITY],
                        affected_entities=["customer", "account", "beneficiary", "transaction"]
                    )
                ],
                transitions=[
                    AttackTransition(target_state="PAYMENT_EXECUTION", min_weight=0.0, max_weight=0.3, condition="LOOP", reason="Victim makes another payment (split or follow-up)."),
                    AttackTransition(target_state="END", min_weight=0.7, max_weight=1.0, reason="Payment complete, scam concludes.")
                ]
            ),
            "END": AttackState(
                state_name="END",
                description="The attack concludes.",
                transitions=[]
            )
        },
        variation_axes=[
            VariationAxis(
                name="amount_scale",
                description="Scale of the transaction relative to account balance.",
                allowed_values=["moderate", "high", "maximum"],
                reason="APP fraud often drains the majority of the victim's available balance."
            ),
            VariationAxis(
                name="beneficiary_novelty",
                description="Whether the beneficiary is newly added or existing.",
                allowed_values=["new", "existing"],
                reason="Most APP uses new mules, but some use compromised existing beneficiaries."
            )
        ],
        constraints=[
            AttackConstraint(description="MUST originate from customer's known primary device.", enforcement_layer="StatefulSimulator"),
            AttackConstraint(description="MUST NOT trigger new device registration.", enforcement_layer="StatefulSimulator")
        ],
        research_sources=[
            ResearchSource(
                source_name="UK Finance",
                title="Fraud - The Facts 2023",
                publication_year=2023,
                relevant_claim="APP fraud is initiated by the victim using their own device and credentials."
            )
        ]
    )
