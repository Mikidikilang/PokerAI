"""
training/opponent_archetypes.py  –  Phase 2 Exploitative Training Botok

MIÉRT KELLENEK (16M ep diagnózis):
  - AF 45+, C-bet 93%: modell over-blöfföl (air river → raise 47%)
  - HUD adaptáció 0%: nit vs neutral → azonos stratégia → HUD feature-ök nem tanulnak
  - Postflop sizing r=0.14: bet méret és kézerő közt nincs korreláció
  - T8o vs 4-bet → raise: self-play Nash-konvergencia mellékhatása

MELYIK BOT MIT CÉLOZ:

  FishBot:         call-ol pair+-szal → BÜNTETI az over-blöfföt (river blöff veszít)
                   reagál bet méretre air-nél → TANÍTJA a sizing-et
  CallingStation:  mindig call ≤ pot → direkt BÜNTETI a random blöfföt
                   csak óriási betnél fold → TANÍTJA hogy kis bet ≠ big bet value
  NitBot:          valódi fold equity → TANÍTJA mikor VAN fold equity (vs NitBot van)
                   fold-ol weak-kel → modell tanul SZELEKTÍVEN blöffölni
  LAGBot:          agresszívan bet → tanítja a 4-bet/fold döntést
                   semi-blöff ellen fold-ol → REWARD-olja a draw agressziót

SIZE-DEPENDENT CALL THRESHOLDS (postflop sizing javításhoz):
  Kis bet (bet_pct ≤ 0.33 → ≤ 1x pot):   liberális call (szinte bármi)
  Normál bet (0.33-0.60 → 1-2x pot):     közepes threshold
  Nagy bet (0.60-0.80 → 2-3x pot):       szigorú threshold
  All-in szintű (> 0.80):                csak strong hand call

Ez a gradient adja a reward-különbséget ami megtanítja:
  "kis bettel érj el sokat mert a fish call-ol → de pontosan ezért ne blöfföl nagyot"

INTERFÉSZ: BatchedSyncCollector kompatibilis (_encode + actor_head batch pipeline).
Debug/test: get_action(state_1d, legal_actions) single-step method is available.

HU (2-player) equity referencia (MC n=200):
  AA ~85% | KK ~82% | QQ ~79% | JJ ~77% | TT ~75% | 99 ~72%
  AKs ~67% | AKo ~65% | AQs ~66% | AQo ~64% | AJs ~63%
  Random hand ~50% | 72o ~35%

  Approximate top-X% thresholds (HU):
    top  8% → eq > 0.70  (AA, KK, QQ, JJ, TT, AKs, AKo)
    top 12% → eq > 0.67  (+ 99, AQs, AQo)
    top 28% → eq > 0.58  (+ 88, 77, AJs, KQs, ...)
    top 35% → eq > 0.55
    top 45% → eq > 0.50  (mid-range, near average)
    top 55% → eq > 0.47  (loose range)
"""

import torch
import random

# ─────────────────────────────────────────────────────────────────────────────
# Callable actor_head wrapper – collector kompatibilitás
# ─────────────────────────────────────────────────────────────────────────────

class _BotHead:
    """
    A BatchedSyncCollector ezt hívja:
        x = opp_model._encode(states_batch)  → pass-through
        logits = opp_model.actor_head(x)     → bot logikából számolt logitok
    """
    def __init__(self, bot):
        self._bot = bot

    def __call__(self, x):
        return self._bot._get_logits_batch(x)


# ─────────────────────────────────────────────────────────────────────────────
# RuleBasedBot – alap osztály
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedBot:
    """
    Alap osztály. Subclass override-olja a _get_logits() metódust.

    State tensor offset-ek (features.py compute_state_size alapján):
      [obs(54)] [stats(np*7)] [stack(8)] [street(4)] [pot_odds(4)] [board(6)]
      [history(8*(np*7+1))] [position(2*np)] [equity(1)]

    Kivont feature-ök per _extract():
      equity        = state[-1]           (0.0–1.0, MC equity becslés)
      street        = argmax(state[s:s+4]) (0=pre, 1=flop, 2=turn, 3=river)
      pot_odds_norm = state[p]            call/(pot+call) – mekkora a call pot-hoz képest
      is_facing_bet = state[p+2] > 0.5   (0 = senki nem bet-elt, 1 = van bet)
      bet_pot_pct   = state[p+3]         bet/pot scaled [0,1] ahol 1 = 3x pot bet
                                          0.33 = 1x pot | 0.67 = 2x pot | >0.80 = all-in szint
    """

    def __init__(self, num_players: int, state_size: int, action_size: int = 7):
        self.action_size = action_size
        self._np = num_players
        self._ss = state_size

        # Offset számítás (features.py struktúra alapján)
        base = 54 + num_players * 7       # obs(54) + stats(np*7)
        self._street_start = base + 8     # stack(8) után
        self._pot_start    = base + 8 + 4 # street(4) után
        self._equity_idx   = state_size - 1

        self.actor_head = _BotHead(self)

    def _extract(self, sv: torch.Tensor):
        """
        1D state vector → (equity, street, pot_odds_norm, is_facing_bet, bet_pot_pct)

        bet_pot_pct értelmezése:
          0.00 = nem volt bet (checking)
          0.33 = ~1x pot bet   ← normál sizing
          0.50 = ~1.5x pot bet ← nagy bet
          0.67 = ~2x pot bet   ← over-bet
          0.80+ = all-in szintű bet
        """
        equity        = float(sv[self._equity_idx])
        street_oh     = sv[self._street_start: self._street_start + 4]
        street        = int(torch.argmax(street_oh)) if street_oh.sum() > 0 else 0
        pot_odds_norm = float(sv[self._pot_start])
        is_facing_bet = float(sv[self._pot_start + 2]) > 0.5
        bet_pot_pct   = float(sv[self._pot_start + 3])
        return equity, street, pot_odds_norm, is_facing_bet, bet_pot_pct

    def _get_logits_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Batch feldolgozás: N state → N logit vektor. Collector-nak szükséges."""
        B = states.shape[0]
        logits = torch.zeros(B, self.action_size, device=states.device)
        for i in range(B):
            logits[i] = self._get_logits(states[i].cpu())
        return logits

    def _get_logits(self, sv: torch.Tensor) -> torch.Tensor:
        """Override subclass-ban. Returns (action_size,) logit tensor."""
        return torch.zeros(self.action_size)

    def get_action(self, state_1d, legal_actions: list) -> int:
        """
        Single-step interface – teszteléshez és debuggoláshoz.
        A BatchedSyncCollector NEM ezt hívja, az _encode + actor_head pipeline-t használja.

        Args:
            state_1d:      1D numpy array vagy 1D tensor, shape (state_size,)
            legal_actions: list[int] – engedélyezett absztrakt akciók
        Returns:
            int – kiválasztott absztrakt akció
        """
        if not isinstance(state_1d, torch.Tensor):
            sv = torch.FloatTensor(state_1d)
        else:
            sv = state_1d.float().cpu()

        logits = self._get_logits(sv)

        mask = torch.full((self.action_size,), -1e9)
        for a in legal_actions:
            if 0 <= a < self.action_size:
                mask[a] = 0.0

        probs = torch.softmax(logits + mask, dim=0)
        return int(probs.argmax().item())

    # Collector kompatibilitás – ezek szükségesek hogy a pool/collector ne omoljon le
    def _encode(self, states_batch): return states_batch  # pass-through
    def eval(self): return self
    def to(self, device): return self
    def parameters(self): return iter([])
    def state_dict(self): return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(np={self._np}, state={self._ss})"


# ─────────────────────────────────────────────────────────────────────────────
# FishBot – VPIP ~55%, PFR ~8%
# ─────────────────────────────────────────────────────────────────────────────

class FishBot(RuleBasedBot):
    """
    Micro-stakes hal archetype. VPIP ~55%, PFR ~8%.

    CÉLZOTT TANULÁS a modellnél:
      1. Over-blöff büntetés: pair+-szal SOSEM fold → minden river blöff veszít
      2. Value bet reward: modell TP+-szal bet → FishBot call → profit signal
      3. Sizing tanítás: air-rel facing kis betnél is call, de NAGY betnél fold
         → modell megtanulja: "kis bettel value-z, ne blöfföl nagyot"

    Preflop:
      eq > 0.70 → raise (2 vagy 6, a fish mindig extrém méretet használ)
      eq > 0.47 → call (top ~55%)
      rest     → fold

    Postflop:
      Pair+ (eq > 0.55): SOHA nem fold, call vagy min-raise
      Gyenge (eq 0.40-0.55): call, kivéve huge bet
      Air (eq < 0.40): SIZE-DEPENDENT: kis bet → call, nagy bet → fold
    """

    def _get_logits(self, sv: torch.Tensor) -> torch.Tensor:
        eq, street, po_norm, facing, bet_pct = self._extract(sv)
        L = torch.zeros(self.action_size)

        if street == 0:  # ── PREFLOP ──────────────────────────────────────
            if eq > 0.70:
                # Top ~8%: raise, fish stílusban (extrém méretek)
                L[2] = 3.0   # min-raise (fish-jellegű)
                L[6] = 2.0   # pot/all-in (fish sokszor all-in-nel nyit)
                L[0] = -15.0 # soha nem fold premium kezekkel
                L[1] = -2.0  # ritkán flat
            elif eq > 0.47:
                # VPIP range (top ~55%): call
                L[1] = 4.5
                L[0] = -1.0
                L[2] = 0.3   # ritkán min-raise a range aljáról is
            else:
                # Bottom ~45%: fold
                L[0] = 4.0
                L[1] = -2.0

        else:  # ── POSTFLOP ──────────────────────────────────────────────
            if facing:
                if eq > 0.55:
                    # Pair+: SOSEM FOLD – ez a fish legsajátosabb jellemzője
                    L[1] = 4.5   # call (fő akció)
                    L[2] = 1.5   # min-raise, fish sokszor felraiser pair-rel is
                    L[6] = 0.5   # all-in push ritkán
                    L[0] = -20.0 # ABSZOLÚT soha nem fold pair+-szal
                elif eq > 0.40:
                    # Gyenge pár / draw: call (size-tól némi függéssel)
                    if bet_pct > 0.67:  # >2x pot: kicsit óvatosabb
                        L[1] = 3.0
                        L[0] = 1.0
                    else:
                        L[1] = 4.0
                        L[0] = 0.2
                else:
                    # Air: SIZE-DEPENDENT (ez a sizing tanulás kulcsa!)
                    # Kis bet  (≤1x pot)  → még a fish is inkább call: ezért értékes a small bet
                    # Közepes bet         → 50/50
                    # Nagy bet (≥2x pot)  → fish is fold: itt nincs reward a blöffnek
                    if bet_pct < 0.33:
                        L[1] = 3.0   # kis bet → call (fish!)
                        L[0] = 1.0
                    elif bet_pct < 0.65:
                        L[0] = 2.0   # közepes bet → 50/50
                        L[1] = 2.0
                    else:
                        L[0] = 3.5   # nagy bet → fold air
                        L[1] = 1.0
            else:
                # Checking to fish
                if eq > 0.58:
                    # Has something: bet (fish mindig big-et bet)
                    L[6] = 3.0   # all-in bet (fish)
                    L[2] = 1.5   # min-bet
                    L[1] = 0.5   # néha check (slow play)
                elif eq > 0.45:
                    # Mediocre: mix bet/check
                    L[2] = 1.5   # small bet
                    L[1] = 2.0   # check
                else:
                    # Weak: check (ne donk-bet air-rel)
                    L[1] = 4.0
                    L[0] = -1.0  # ne fold when not facing bet

        return L


# ─────────────────────────────────────────────────────────────────────────────
# NitBot – VPIP ~12%, PFR ~10%
# ─────────────────────────────────────────────────────────────────────────────

class NitBot(RuleBasedBot):
    """
    Rock/Nit archetype. VPIP ~12%, PFR ~10%. Soha nem blöfföl.

    CÉLZOTT TANULÁS:
      1. Szelektív fold equity: Nit fold-ol → modell megtanulja MIKOR van fold equity
         (A jelenleg "mindenki fold-ol ha elég agresszívan raiselek" tévhit ellen)
      2. Value bet tanítás: Nit CSAK strong kezekkel játszik → ha ő bet-el, az value
         → modell tanul fold-olni vs jól timing-olt nit bet (nem auto-call)
      3. Timing: Nit fold-ol gyenge kezekkel BÁRMILYEN agresszióra
         → a modell tanul: van ellenfél aki fold-ol (nem mindenki CallingStation)

    Preflop:  eq > 0.70 → raise (proper sizing), rest → fold (nincs limp!)
    Postflop: strong (eq > 0.65) → value bet, decent (eq > 0.50) → check/call,
              weak → fold BÁRMILYEN bet-re
    """

    def _get_logits(self, sv: torch.Tensor) -> torch.Tensor:
        eq, street, po_norm, facing, bet_pct = self._extract(sv)
        L = torch.zeros(self.action_size)

        if street == 0:  # ── PREFLOP ──────────────────────────────────────
            if eq > 0.70:
                # Top ~12% prémium: raise (nit proper méretet használ)
                L[4] = 3.5   # raise 50%
                L[5] = 2.0   # raise 75%
                L[3] = 1.0   # raise 25% ritkán
                L[0] = -15.0 # SOHA nem fold premiumot
                L[1] = -5.0  # nincs limp (nit nem limpel)
            else:
                # MINDEN más: fold (nit a nit)
                L[0] = 8.0
                L[1] = -5.0  # nincs limp
                L[2] = -5.0  # nincs speculative raise

        else:  # ── POSTFLOP ──────────────────────────────────────────────
            if facing:
                if eq > 0.65:
                    # Strong: raise for value (nit sosem slow-play a betek ellen)
                    L[4] = 3.5   # raise 50%
                    L[5] = 2.5   # raise 75%
                    L[3] = 1.0   # raise 25% (thin value)
                    L[1] = 0.8   # néha flat, pot control
                    L[0] = -15.0 # soha nem fold strongot
                elif eq > 0.50:
                    # Decent: bet mérettől függő call/fold
                    if bet_pct > 0.67:      # >2x pot facing decent hand: fold
                        L[0] = 3.0
                        L[1] = 0.8
                    elif bet_pct > 0.40:    # normal bet: call
                        L[1] = 3.0
                        L[0] = 1.0
                    else:                   # kis bet: call
                        L[1] = 4.0
                        L[0] = 0.5
                else:
                    # Weak: FOLD minden bet-re (nit soha nem chase-el)
                    L[0] = 7.0
                    L[1] = -5.0
            else:
                # Checked to nit
                if eq > 0.68:
                    # Monster: value bet (varied sizing, nit-szerű)
                    L[4] = 3.5   # raise 50% value
                    L[3] = 2.5   # raise 25% thin value
                    L[5] = 1.5   # raise 75% premium hands
                elif eq > 0.55:
                    # Decent: thin value bet vagy check
                    L[3] = 2.5   # raise 25%
                    L[1] = 2.0   # check is OK
                else:
                    # Weak: CHECK (nit SOHA nem blöfföl)
                    L[1] = 7.0
                    for bluff_a in range(2, self.action_size):
                        L[bluff_a] = -15.0  # absolute never bluff

        return L


# ─────────────────────────────────────────────────────────────────────────────
# CallingStation – VPIP ~45%, postflop sosem raise
# ─────────────────────────────────────────────────────────────────────────────

class CallingStation(RuleBasedBot):
    """
    Calling station archetype. VPIP ~45%, postflop SOHA nem raise.

    CÉLZOTT TANULÁS:
      1. Direkt over-blöff büntetés: calling station call-ol bármit ≤ pot
         → a modell minden river blöffje (ami jelenleg 47% prob) pénzt veszít
      2. SIZE-DEPENDENT reward: CallingStation fold csak >2x pot + air esetén
         → megtanítja: kis-közepes bet-tel value-z (call-olják), ne blöfföl nagyot
         → ez a fő mechanizmus ami a postflop sizing r=0.14-et javítja
      3. Value bet reward: modell TP+-szal bet → CS call → profit
         → erősíti a value bet szokást

    Preflop:  eq > 0.78 → rare raise (AA, KK slow play), eq > 0.50 → call, rest fold
    Postflop: SOSEM raise (actions 2-6 = -15.0 logit).
              Call bármit ha eq > 0.30 VAGY bet ≤ 1.5x pot
              Fold csak ha bet > 2x pot ÉS eq < 0.30
    """

    def _get_logits(self, sv: torch.Tensor) -> torch.Tensor:
        eq, street, po_norm, facing, bet_pct = self._extract(sv)
        L = torch.zeros(self.action_size)

        if street == 0:  # ── PREFLOP ──────────────────────────────────────
            if eq > 0.78:
                # Csak top prémiumok: slow play (AA, KK) – calling station jellemzője
                L[1] = 3.5   # mostly call (slow play)
                L[2] = 0.8   # ritkán min-raise
                L[0] = -15.0
            elif eq > 0.50:
                # Top ~45%: call, call, call
                L[1] = 5.0
                L[0] = -1.0
            else:
                # Bottom ~55%: fold
                L[0] = 3.5
                L[1] = -1.5

        else:  # ── POSTFLOP ──────────────────────────────────────────────
            # KULCSSZABÁLY: SOHA nem raise postflop
            for a in range(2, self.action_size):
                L[a] = -15.0

            if facing:
                # SIZE-DEPENDENT CALLING (ez a sizing tanulás lényege):
                huge_bet = bet_pct > 0.67      # >2x pot
                big_bet  = bet_pct > 0.45      # >1.5x pot
                small_bet = bet_pct < 0.33     # ≤1x pot

                if eq > 0.30:
                    # Van valamije: CALL mindent (calling station!)
                    # A modell ezt tanulhatja: value-betnél mindig kap call-t
                    L[1] = 5.0
                    L[0] = -5.0
                elif huge_bet:
                    # Pure trash + óriási bet: fold (még a CS is fold)
                    # Ez tanítja: big blöffnél nincs reward, de small blöffnél sincs
                    L[0] = 3.0
                    L[1] = 1.5   # de még így is sokszor call (station)
                elif big_bet:
                    # Trash + nagy bet: inkább fold
                    L[0] = 2.5
                    L[1] = 2.0
                else:
                    # Trash + kis bet: call (még ez is call!)
                    # Ez a kulcs: kis bet-nél nincs "safe" blöff profit
                    L[1] = 3.0
                    L[0] = 1.5
            else:
                # Checked to CS: check (calling station sosem donk-bet)
                L[1] = 5.0

        return L


# ─────────────────────────────────────────────────────────────────────────────
# LAGBot – VPIP ~35%, PFR ~28%
# ─────────────────────────────────────────────────────────────────────────────

class LAGBot(RuleBasedBot):
    """
    Loose-Aggressive archetype. VPIP ~35%, PFR ~28%. C-bet ~80%, 3-bet ~12%.

    CÉLZOTT TANULÁS:
      1. Draw semi-blöff reward: LAGBot fold-ol draw-okra aggression ellen
         → modell tanul agresszívan semi-blöffölni (jelenleg 15-out draw call 87%)
      2. 3-bet defense tanítás: LAGBot 3-bet-el széles range-ből
         → modell tanul 4-bet-et csinálni vagy fold-olni (nem auto-call)
      3. Bet sizing diverzitás: LAGBot különböző méreteket használ
         → diverse training signal → modell is diverzifikálni fogja a sizing-et
      4. River blöff (részleges) büntetés: LAGBot call-ol medium equity-vel
         → random blöfföt bünteti, de nem annyira mint CallingStation

    Preflop:  eq > 0.58 → raise (varied), eq > 0.47 → marginal mix, rest fold
    Postflop: c-bet ~80% ha nem facing bet, double barrel, varied sizing (3,4,5)
              fold weak kezekkel ha facing significant aggression
    """

    def _get_logits(self, sv: torch.Tensor) -> torch.Tensor:
        eq, street, po_norm, facing, bet_pct = self._extract(sv)
        L = torch.zeros(self.action_size)

        if street == 0:  # ── PREFLOP ──────────────────────────────────────
            if eq > 0.75:
                # Premium: 3-bet big (LAG jellemzője: értékesít)
                L[5] = 3.5   # raise 75%
                L[6] = 2.5   # pot/all-in 3-bet
                L[4] = 1.5   # raise 50%
                L[0] = -15.0
            elif eq > 0.58:
                # Top ~28%: raise (PFR range, varied sizing)
                L[3] = 3.0   # raise 25%
                L[4] = 2.5   # raise 50%
                L[5] = 1.5   # raise 75%
            elif eq > 0.47:
                # Marginal: call/fold mix (LAG loose de nem teljesen random)
                L[1] = 2.5   # call
                L[0] = 1.5   # fold
            else:
                # Fold
                L[0] = 4.0
                L[1] = -1.0

        else:  # ── POSTFLOP ──────────────────────────────────────────────
            if not facing:
                # NOT facing bet → agresszív c-bet (~80% frequency)
                if eq > 0.65:
                    # Strong made hand: value bet varied
                    L[4] = 3.0   # raise 50%
                    L[3] = 2.5   # raise 25%
                    L[5] = 1.5   # raise 75%
                    L[1] = 0.5   # ritkán check (deception)
                elif eq > 0.40:
                    # Semi-blöff / thin value: bet (LAG c-bet range)
                    L[3] = 3.0   # raise 25% (c-bet sizing)
                    L[4] = 2.5   # raise 50%
                    L[1] = 1.5   # check ~25%
                else:
                    # Air: c-bet ~60%, check ~40% (LAG nem bet 100%)
                    L[3] = 2.5   # blöff raise 25%
                    L[4] = 1.0   # blöff raise 50% ritkán
                    L[1] = 2.0   # check 40%
                    L[0] = -3.0  # ne fold ha nem facing bet

            else:  # facing bet
                if eq > 0.65:
                    # Strong: re-raise (LAG agresszív re-raise-szel)
                    L[4] = 3.0   # raise 50%
                    L[5] = 2.5   # raise 75%
                    L[3] = 1.5   # raise 25%
                    L[1] = 1.0   # flat is OK
                    L[0] = -15.0
                elif eq > 0.48:
                    # Medium: call, ritkán blöff-raise
                    L[1] = 2.5   # call
                    L[3] = 1.5   # blöff-raise 25% (LAG float/raise)
                    L[0] = 0.5
                elif eq > 0.32:
                    # Weak: bet mérettől függő
                    if bet_pct > 0.50:  # big bet facing weak: fold mostly
                        L[0] = 3.0
                        L[1] = 1.0
                        L[3] = 0.5   # ritkán blöff-raise
                    else:              # kis bet: float vagy blöff
                        L[1] = 2.0
                        L[0] = 1.5
                        L[3] = 1.0   # blöff-raise
                else:
                    # Pure trash: give up (LAG tudja mikor kell fold-olni)
                    L[0] = 4.5
                    L[1] = 0.5

        return L


# ─────────────────────────────────────────────────────────────────────────────
# Registry és Factory
# ─────────────────────────────────────────────────────────────────────────────

BOT_REGISTRY: dict = {
    'fish':            FishBot,
    'nit':             NitBot,
    'calling_station': CallingStation,
    'lag':             LAGBot,
}


def create_bot(bot_type: str, num_players: int, state_size: int,
               action_size: int = 7) -> RuleBasedBot:
    """
    Egyetlen bot létrehozása típus szerint.

    Args:
        bot_type:    'fish' | 'nit' | 'calling_station' | 'lag'
        num_players: asztalmméret (state tensor méretéhez kell)
        state_size:  compute_state_size(54, num_players) eredménye
        action_size: default 7

    Returns:
        RuleBasedBot instance

    Example:
        bot = create_bot('nit', num_players=2, state_size=215)
        action = bot.get_action(state_vec, legal_actions=[0,1,2,3,4,5,6])
    """
    if bot_type not in BOT_REGISTRY:
        raise ValueError(
            f"Ismeretlen bot: '{bot_type}'. "
            f"Elérhető: {list(BOT_REGISTRY.keys())}"
        )
    return BOT_REGISTRY[bot_type](num_players, state_size, action_size)


def create_all_bots(num_players: int, state_size: int,
                    action_size: int = 7) -> dict:
    """
    Minden bot típusból egy-egy példány, dict-ben visszaadva.

    Returns:
        {'fish': FishBot, 'nit': NitBot, 'calling_station': CS, 'lag': LAGBot}
    """
    return {
        name: cls(num_players, state_size, action_size)
        for name, cls in BOT_REGISTRY.items()
    }
