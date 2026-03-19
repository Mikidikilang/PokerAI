"""
training/opponent_archetypes.py  –  Szabály-alapú ellenfél arche-típusok

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIÉRT KELLENEK?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Pure self-play-ben minden ellenfél ~azonos policy → a modell
  sosem tanulja meg a HUD feature-öket használni. Ezek a botok
  stabil, eltérő játékstílussal rendelkeznek, így a modell:
    1. Megtanulja a HUD inputokat figyelni
    2. Különböző stratégiákat használ különböző ellenfelek ellen
    3. Nem ragad be agresszív lokális optimumba

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KOMPATIBILITÁS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A collector.py _step_opponents_batched() így hívja:
      x = opp_model._encode(states_batch)
      logits = opp_model.actor_head(x)
  Majd mask + softmax + sample.

  A botok kompatibilisek: _encode() pass-through, actor_head
  callable wrapper ami szabály-alapú logitokat generál.
  A collector-t NEM KELL módosítani.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOTOK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FishBot:         VPIP~55%, passzív, soha nem fold postflop pair+
  NitBot:          VPIP~12%, csak premium kezekkel, soha nem blöfföl
  CallingStation:  VPIP~45%, mindent call-ol, soha nem raise postflop
  LAGBot:          VPIP~35%, PFR~28%, sokat blöfföl, agresszív
"""

import torch
import random


# ═══════════════════════════════════════════════════════════════════════════════
# ALAP OSZTÁLY – collector-kompatibilis interface
# ═══════════════════════════════════════════════════════════════════════════════

class _BotHead:
    """Callable actor_head wrapper – a collector opp_model.actor_head(x)-et hív."""
    def __init__(self, bot):
        self._bot = bot
    def __call__(self, x):
        return self._bot._get_logits_batch(x)


class RuleBasedBot:
    """
    Alap osztály a szabály-alapú botoknak.
    Kompatibilis a BatchedSyncCollector-ral:
      x = bot._encode(states_batch)     → pass-through
      logits = bot.actor_head(x)        → szabály-alapú logitok
      (collector alkalmaz mask + softmax + sample)

    A botok a state tensor-ból kinyerik:
      - equity (utolsó elem)
      - street (one-hot, 4 elem)
      - is_facing_bet
      - pot_odds_norm

    Ezek elegendőek szabály-alapú döntésekhez.
    """

    def __init__(self, num_players, state_size, action_size=7):
        self.action_size = action_size
        self._np = num_players
        self._ss = state_size

        # State tensor offset-ek (features.py struktúra alapján)
        # [obs(54)][stats(np*7)][stack(8)][street(4)][pot_odds(4)][board(6)]...
        base = 54 + num_players * 7
        self._street_start = base + 8          # 4 elem one-hot
        self._pot_odds_start = base + 8 + 4    # 4 elem
        self._equity_idx = state_size - 1      # utolsó elem

        self.actor_head = _BotHead(self)

    def _encode(self, states_batch):
        """Pass-through – a collector ezt hívja először."""
        return states_batch

    def _extract(self, state_vec):
        """Kinyeri a döntéshez szükséges infókat a state tensor-ból."""
        equity = float(state_vec[self._equity_idx])

        # Street (one-hot → int)
        street_oh = state_vec[self._street_start:self._street_start + 4]
        street = int(torch.argmax(street_oh)) if street_oh.sum() > 0 else 0

        # Pot odds
        po = state_vec[self._pot_odds_start:self._pot_odds_start + 4]
        pot_odds_norm = float(po[0])       # pot_odds: call/(pot+call)
        is_facing_bet = float(po[2]) > 0.5 # is_facing_bet flag

        return equity, street, pot_odds_norm, is_facing_bet

    def _get_logits_batch(self, states):
        """Batch logit generálás. Subclass override-olja."""
        B = states.shape[0]
        logits = torch.zeros(B, self.action_size, device=states.device)
        for i in range(B):
            logits[i] = self._get_logits(states[i])
        return logits

    def _get_logits(self, state_vec):
        """Egyetlen state → logits. OVERRIDE subclass-ban."""
        return torch.zeros(self.action_size)

    # Collector kompatibilitás
    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([])

    def state_dict(self):
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(np={self._np})"


# ═══════════════════════════════════════════════════════════════════════════════
# FISHBOT – VPIP ~55%, passzív, pair+ soha nem fold
# ═══════════════════════════════════════════════════════════════════════════════

class FishBot(RuleBasedBot):
    """
    Tipikus micro-stakes hal.
    - Preflop: call ~55% kezet, raise ~8%, fold ~37%
    - Postflop: ha equity > 0.45 (pair+), soha nem fold. Ha air, 50% call.
    - Soha nem raise postflop (passzív)
    - HUD profil: VPIP=55, PFR=8, AF=0.5
    """

    def _get_logits(self, state_vec):
        eq, street, pot_odds, facing = self._extract(state_vec)
        L = torch.zeros(self.action_size)

        if street == 0:  # Preflop
            if eq > 0.55:
                # Erős kéz → raise (ritkán)
                L[1] = 3.0   # call
                L[2] = 1.5   # min raise
                L[0] = -2.0  # nem fold
            elif eq > 0.30:
                # Közepes → call
                L[1] = 3.0   # call
                L[0] = 0.5   # néha fold
            else:
                # Gyenge → fold/call mix
                L[0] = 2.0   # fold
                L[1] = 1.5   # de néha call
        else:  # Postflop
            if eq > 0.45:
                # Van valamije → call mindent (passzív fish)
                L[1] = 4.0   # call
                L[0] = -3.0  # soha nem fold
                L[2] = 0.5   # nagyon ritkán min bet
            elif eq > 0.25:
                # Gyenge húzó → call
                L[1] = 2.0
                L[0] = 1.0
            else:
                # Air → 50/50 call/fold
                L[1] = 1.5
                L[0] = 1.5
                if not facing:  # checked to fish → check
                    L[1] = 3.0
                    L[0] = -1.0

        return L


# ═══════════════════════════════════════════════════════════════════════════════
# NITBOT – VPIP ~12%, tight, soha nem blöfföl
# ═══════════════════════════════════════════════════════════════════════════════

class NitBot(RuleBasedBot):
    """
    Extrém tight játékos (rock/nit).
    - Preflop: csak top ~12% kézzel (equity > 0.62). Mindig raise.
    - Postflop: strong (eq > 0.65) → value bet. Weak → fold.
    - Soha nem blöfföl.
    - HUD profil: VPIP=12, PFR=10, AF=3.0, fold_to_cbet=70
    """

    def _get_logits(self, state_vec):
        eq, street, pot_odds, facing = self._extract(state_vec)
        L = torch.zeros(self.action_size)

        if street == 0:  # Preflop
            if eq > 0.62:
                # Premium → raise
                L[4] = 3.0   # raise 50%
                L[5] = 2.0   # raise 75%
                L[0] = -5.0  # soha nem fold
            elif eq > 0.55:
                # Borderline → call/fold
                L[1] = 1.5
                L[0] = 1.0
            else:
                # Gyenge → fold
                L[0] = 5.0
                L[1] = -2.0
        else:  # Postflop
            if eq > 0.65:
                # Strong made hand → value bet
                L[3] = 3.0   # raise 25%
                L[4] = 2.5   # raise 50%
                L[1] = 1.0   # esetleg check/call (trap)
            elif eq > 0.45:
                # Decent → check/call
                L[1] = 2.5
                L[0] = 1.0
            else:
                # Weak → fold (soha nem blöfföl!)
                L[0] = 4.0
                L[1] = 0.0  # max check
                if not facing:
                    L[1] = 2.0  # check if not facing bet
                    L[0] = -1.0

        return L


# ═══════════════════════════════════════════════════════════════════════════════
# CALLING STATION – VPIP ~45%, soha nem raise postflop
# ═══════════════════════════════════════════════════════════════════════════════

class CallingStation(RuleBasedBot):
    """
    Mindent call-ol, soha nem raise.
    - Preflop: call ~45%, raise ~5%
    - Postflop: call mindent (hacsak nem >2x pot bet + air)
    - Soha nem blöfföl, soha nem raise postflop.
    - HUD profil: VPIP=45, PFR=5, AF=0.3, fold_to_cbet=20
    """

    def _get_logits(self, state_vec):
        eq, street, pot_odds, facing = self._extract(state_vec)
        L = torch.zeros(self.action_size)

        if street == 0:  # Preflop
            if eq > 0.58:
                # Erős → call (néha raise)
                L[1] = 3.5   # call
                L[2] = 0.5   # ritkán min raise
            elif eq > 0.32:
                # Közepes → call
                L[1] = 3.0
                L[0] = 0.5
            else:
                # Gyenge → fold/call
                L[0] = 2.0
                L[1] = 1.0
        else:  # Postflop
            if facing:
                if eq > 0.20:
                    # Van bármi → CALL (ez a calling station lényege)
                    L[1] = 5.0   # call
                    L[0] = -2.0  # nem fold
                    # soha nem raise
                else:
                    # Teljesen air, nagy bet → néha fold
                    if pot_odds > 0.4:  # nagy bet
                        L[0] = 2.0
                        L[1] = 1.5  # de még így is gyakran call
                    else:
                        L[1] = 4.0  # kis bet → call
                        L[0] = -1.0
            else:
                # Checked to us → check (soha nem bet)
                L[1] = 5.0   # check
                L[0] = -3.0

        return L


# ═══════════════════════════════════════════════════════════════════════════════
# LAGBOT – VPIP ~35%, PFR ~28%, agresszív blöffölő
# ═══════════════════════════════════════════════════════════════════════════════

class LAGBot(RuleBasedBot):
    """
    Loose-Aggressive játékos. Sokat raise-el, sokat blöfföl.
    - Preflop: raise ~28%, call ~7%, fold ~65%
    - Postflop: cbet ~80%, double barrel ~50%, blöfföl gyenge kezekkel
    - HUD profil: VPIP=35, PFR=28, AF=3.5, cbet=80
    """

    def _get_logits(self, state_vec):
        eq, street, pot_odds, facing = self._extract(state_vec)
        L = torch.zeros(self.action_size)

        if street == 0:  # Preflop
            if eq > 0.50:
                # Erős → nagy raise
                L[4] = 3.0   # raise 50%
                L[5] = 2.5   # raise 75%
                L[6] = 1.0   # néha all-in
            elif eq > 0.38:
                # Decent → raise (steal)
                L[3] = 3.0   # raise 25%
                L[4] = 2.0   # raise 50%
            elif eq > 0.30:
                # Marginális → mix raise/fold
                L[2] = 2.0   # min raise
                L[0] = 1.5   # fold
            else:
                # Gyenge → fold
                L[0] = 3.5
                L[1] = 0.5
        else:  # Postflop
            if not facing:
                # Nem facing bet → agresszívan bet (cbet 80%)
                if eq > 0.40:
                    # Van valami → value bet
                    L[4] = 3.0
                    L[3] = 2.5
                else:
                    # Air → blöff bet (ez a LAG lényege)
                    L[3] = 2.5   # blöff 25%
                    L[4] = 1.5   # blöff 50%
                    L[1] = 1.0   # néha give up
            else:
                # Facing bet
                if eq > 0.55:
                    # Strong → raise (agresszív)
                    L[4] = 3.0
                    L[5] = 2.0
                    L[1] = 1.0
                elif eq > 0.35:
                    # Decent → call vagy raise
                    L[1] = 2.0
                    L[3] = 1.5   # blöff raise
                else:
                    # Weak → fold / blöff raise (polarizált)
                    L[0] = 2.5
                    L[5] = 1.0   # ritkán blöff raise
                    L[6] = 0.5   # nagyon ritkán blöff shove

        return L


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY – bot létrehozás név alapján
# ═══════════════════════════════════════════════════════════════════════════════

BOT_REGISTRY = {
    'fish': FishBot,
    'nit': NitBot,
    'calling_station': CallingStation,
    'lag': LAGBot,
}

def create_bot(bot_type, num_players, state_size, action_size=7):
    """
    Bot létrehozás típus alapján.

    Használat:
        bot = create_bot('fish', num_players=2, state_size=215)
        # Használható mint opponent modell a collector-ban
    """
    if bot_type not in BOT_REGISTRY:
        raise ValueError(f"Ismeretlen bot: {bot_type}. "
                         f"Elérhető: {list(BOT_REGISTRY.keys())}")
    return BOT_REGISTRY[bot_type](num_players, state_size, action_size)


def create_all_bots(num_players, state_size, action_size=7):
    """Minden bot típusból egy-egy példány."""
    return {name: cls(num_players, state_size, action_size)
            for name, cls in BOT_REGISTRY.items()}
