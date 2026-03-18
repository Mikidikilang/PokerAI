"""
core/action_mapper.py  –  Absztrakt akció leképező

7 absztrakt akció:
  0 = Fold
  1 = Call / Check
  2 = Raise tier 0  (min-raise / pot 0%)
  3 = Raise tier 1  (pot 25%)
  4 = Raise tier 2  (pot 50%)
  5 = Raise tier 3  (pot 75%)
  6 = Raise tier 4  (pot 100% / all-in)

A "tier" az rlcard belső raise action indexeire képez le
frakcionált interpolációval (closest action).
"""


class PokerActionMapper:
    NUM_CUSTOM_ACTIONS = 7

    # Raise tier → pot fraction (0%..100%)
    RAISE_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

    # Human-readable nevek (RealtimeAssistant outputhoz)
    ACTION_NAMES = {
        0: "Fold",
        1: "Call/Check",
        2: "Raise (min)",
        3: "Raise 25%",
        4: "Raise 50%",
        5: "Raise 75%",
        6: "Raise pot/all-in",
    }

    def get_abstract_legal_actions(self, raw_legal_actions: list) -> list:
        """
        rlcard raw akció lista → absztrakt akció lista.
        Ha raise lehetséges: mind az 5 raise tier elérhető.
        """
        abstract = []
        if 0 in raw_legal_actions:
            abstract.append(0)
        if 1 in raw_legal_actions:
            abstract.append(1)
        if any(isinstance(a, int) and a > 1 for a in raw_legal_actions):
            abstract.extend([2, 3, 4, 5, 6])
        if not abstract:
            abstract = [list(raw_legal_actions)[0] if raw_legal_actions else 1]
        return abstract

    def get_env_action(self, abstract_action: int,
                       raw_legal_actions: list) -> int:
        """
        Absztrakt akció → rlcard env akció.
        Raise: legközelebb eső action az interpolált targethez.
        """
        if abstract_action == 0 and 0 in raw_legal_actions:
            return 0
        if abstract_action == 1 and 1 in raw_legal_actions:
            return 1

        raise_actions = sorted(
            a for a in raw_legal_actions if isinstance(a, int) and a > 1
        )
        if not raise_actions:
            # Nincs raise lehetőség → call/fold fallback
            if 1 in raw_legal_actions:
                return 1
            if 0 in raw_legal_actions:
                return 0
            return list(raw_legal_actions)[0]

        tier   = abstract_action - 2
        frac   = self.RAISE_FRACTIONS[min(tier, 4)]
        target = raise_actions[0] + (raise_actions[-1] - raise_actions[0]) * frac
        return min(raise_actions, key=lambda x: abs(x - target))

    def action_name(self, abstract_action: int) -> str:
        return self.ACTION_NAMES.get(abstract_action, f"Action {abstract_action}")
