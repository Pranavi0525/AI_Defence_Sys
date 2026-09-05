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


def get_mule_network_signature() -> AttackSignature:
    """Return the static, validated MULE_NETWORK Attack Signature.

    Models a single hop in a money-mule network: a mule customer whose
    OWN account and device look individually unremarkable, receiving
    funds and rapidly relaying them onward to a shared collector entity
    (a common beneficiary_id, or a common bank_id corridor) that several
    different mule customers route through.

    This signature only describes ONE mule's per-customer state machine
    -- the same shape the existing StatefulSimulator already knows how
    to walk. The network-level correlation (several different customers
    converging on the same collector) is deliberately NOT modeled here;
    it is the responsibility of MuleNetworkOrchestrator, which runs this
    signature once per mule and forces them to share a collector entity.
    Keeping that split honest: this file only claims what a single
    customer's state machine can represent.
    """

    return AttackSignature(
        attack_family="MULE_NETWORK",
        version="1.0",
        description=(
            "A single hop in a money-mule network: an individually ordinary "
            "customer receives funds into their own account and rapidly "
            "relays them onward to a shared collector beneficiary or "
            "cross-bank corridor. Individually the hop is unremarkable; "
            "the pattern is only visible as a network-level correlation "
            "across several different mules' accounts."
        ),
        entry_states=["SESSION_ACCESS"],
        states={
            "SESSION_ACCESS": AttackState(
                state_name="SESSION_ACCESS",
                description=(
                    "The mule logs into their own account from their own, "
                    "already-known device -- unlike ATO, there is no "
                    "credential compromise here, this is the mule's genuine "
                    "session."
                ),
                observable_consequences=[
                    ObservableConsequence(
                        description="Ordinary session login from the mule's own device.",
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.DEVICE_SESSION],
                        affected_entities=["customer", "device", "session"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="OUTBOUND_RELAY", min_weight=0.85, max_weight=1.0,
                        reason="Mule proceeds to relay the funds already sitting in the account."
                    ),
                    AttackTransition(
                        target_state="END", min_weight=0.0, max_weight=0.15,
                        reason="Mule delays this session; relay happens on a later trace."
                    )
                ]
            ),
            "OUTBOUND_RELAY": AttackState(
                state_name="OUTBOUND_RELAY",
                description=(
                    "The mule sends funds onward to the shared collector "
                    "entity (a common beneficiary shared with other mules, "
                    "or the same cross-bank corridor via bank_id)."
                ),
                observable_consequences=[
                    ObservableConsequence(
                        description=(
                            "Outbound transaction to the network's shared "
                            "collector, often split across several transfers."
                        ),
                        observability=Observability.DIRECTLY_OBSERVABLE,
                        signal_families=[SignalFamily.TRANSACTION, SignalFamily.VELOCITY, SignalFamily.RELATIONSHIP],
                        affected_entities=["customer", "account", "beneficiary", "transaction"]
                    )
                ],
                transitions=[
                    AttackTransition(
                        target_state="OUTBOUND_RELAY", min_weight=0.0, max_weight=0.35, condition="LOOP",
                        reason="Mule splits the relay across multiple smaller transfers to stay under per-transfer limits."
                    ),
                    AttackTransition(
                        target_state="END", min_weight=0.65, max_weight=1.0,
                        reason="Relay complete; this mule's hop in the chain concludes."
                    )
                ]
            ),
            "END": AttackState(
                state_name="END",
                description="This mule's hop in the network concludes.",
                transitions=[]
            )
        },
        variation_axes=[
            VariationAxis(
                name="hop_position",
                description="This mule's position within the wider relay chain.",
                allowed_values=["first_hop", "middle_hop", "last_hop"],
                reason="First-hop mules receive directly from the fraud proceeds; later hops receive from prior mules, and typically relay faster and split more to launder distance."
            ),
            VariationAxis(
                name="corridor_type",
                description="Whether this hop stays within one simulated bank or crosses a bank_id boundary.",
                allowed_values=["same_bank", "cross_bank"],
                reason="A single-institution view cannot see a cross-bank corridor; this is the axis MULE_NETWORK exists to exercise, using the bank_id field on Account."
            )
        ],
        constraints=[
            AttackConstraint(
                description=(
                    "MUST route the OUTBOUND_RELAY transaction to a shared "
                    "collector beneficiary_id (or a shared bank_id corridor) "
                    "supplied by MuleNetworkOrchestrator, not a beneficiary "
                    "chosen independently per mule."
                ),
                enforcement_layer="MuleNetworkOrchestrator"
            ),
            AttackConstraint(
                description="SHOULD use the mule's own known/established device, not fresh attacker infrastructure.",
                enforcement_layer="StatefulSimulator"
            )
        ],
        research_sources=[
            ResearchSource(
                source_name="UK Finance",
                title="Annual Fraud Report 2023",
                publication_year=2023,
                relevant_claim="Money mule accounts are used to receive and rapidly relay illicit funds across multiple bank accounts before the trail goes cold."
            ),
            ResearchSource(
                source_name="Financial Action Task Force (FATF)",
                title="Professional Money Laundering",
                publication_year=2018,
                relevant_claim="Layering funds through a chain of mule accounts across different institutions is a standard technique to obscure the origin of illicit proceeds before final consolidation."
            )
        ]
    )
