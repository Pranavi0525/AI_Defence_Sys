import inspect
from red_team.attacks.simulator import StatefulSimulator, VariationProfile
from red_team.attacks.ato_signature import get_ato_signature
from red_team.validation.novelty import AttackFingerprint
from red_team.validation.realism import validate_attack_realism

def run_assessment():
    print("================ STAGE 25 APP ASSESSMENT ================\n")
    print("1. Existing Reusable Abstractions:")
    print("   - StatefulSimulator: High reusability for core loop, event generation, and state limits.")
    print("   - AttackSignature: Fully reusable graph model.")
    print("   - Observable / Ground Truth Isolation: Fully reusable, structural validation works.")
    print("   - RealismValidator: Highly reusable, requires extending behavioral checks for APP.")
    print("   - CorpusGenerationResult & quotas: Fully reusable.")

    print("\n2. ATO-Specific Abstractions:")
    print("   - AttackFingerprint: Hardcodes device changes, beneficiary changes suited to ATO.")
    print("   - get_ato_signature(): Entirely ATO specific.")
    print("   - VariationProfile settings (like device_new_prob=1.0 for Easy) are ATO-tuned.")

    print("\n3. Schemas APP can already use:")
    print("   - SessionLogin, DeviceRegistration, BeneficiaryAddition, Transaction.")
    print("   - Entity schemas (Customer, Device, Account, Beneficiary, Transaction).")

    print("\n4. Schemas that require extension:")
    print("   - AttackGroundTruth (Needs an 'attack_objective' or similar if we want to model social engineering).")
    print("   - AttackFingerprint (needs polymorphism or a new APPAttackFingerprint).")

    print("\n5. WorldState capabilities required by APP:")
    print("   - Fetching the customer's *existing* primary device.")
    print("   - Checking existing beneficiaries to simulate 'known' vs 'new' payments.")

    print("\n6. Realism checks required by APP:")
    print("   - MUST enforce that APP originates from a *known* device.")
    print("   - Reject APP if it uses a brand new device out of nowhere (that would be ATO).")

    print("\n7. Novelty dimensions required by APP:")
    print("   - Known vs New beneficiary.")
    print("   - Amount patterns (e.g. single large vs multiple small).")
    print("   - Timing (rapid succession vs slow drain).")
    print("   - Session continuity (same session vs broken sessions).")

    print("\n8. Provenance requirements:")
    print("   - APP transition probabilities (DOMAIN_MODELED).")
    print("   - APP amount variances (DOMAIN_MODELED).")

if __name__ == "__main__":
    run_assessment()
