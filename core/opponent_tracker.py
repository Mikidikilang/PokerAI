"""
core/opponent_tracker.py  –  OpponentHUDTracker

v4 újítás: valódi póker HUD statisztikák per játékos.
Régi (v3.x): nyers action frequency → nem informatív.
Új (v4):     VPIP, PFR, AF, 3bet%, fold_to_3bet, cbet%, fold_to_cbet.

Hideg start probléma: 0 helyett 0.5 neutral prior – elkerüli a hamis
"soha nem raised" következtetést az első néhány kézben.

Memory: rolling deque per stat per player (maxlen=1000 event).
"""

import collections


# Stat indexek a get_stats_vector() outputban
STAT_VPIP         = 0  # Voluntarily Put money In Pot (preflop call/raise %)
STAT_PFR          = 1  # Pre-Flop Raise %
STAT_AF           = 2  # Aggression Factor: (bet+raise)/call
STAT_3BET         = 3  # 3-bet % (amikor volt 3bet lehetőség)
STAT_FOLD_TO_3BET = 4  # Fold to 3-bet %
STAT_CBET         = 5  # Continuation bet % (flop, mint preflop aggressor)
STAT_FOLD_TO_CBET = 6  # Fold to c-bet %

NUM_HUD_STATS = 7
NEUTRAL_PRIOR = 0.5   # cold start érték


class PlayerStats:
    """Egy játékos HUD statisztikái."""

    def __init__(self, memory: int = 1000):
        # VPIP
        self.vpip_opps  = collections.deque(maxlen=memory)  # 1=volt lehetőség
        self.vpip_acts  = collections.deque(maxlen=memory)  # 1=belement
        # PFR
        self.pfr_opps   = collections.deque(maxlen=memory)
        self.pfr_acts   = collections.deque(maxlen=memory)
        # AF komponensek
        self.aggressive = collections.deque(maxlen=memory)  # bet/raise events
        self.passive    = collections.deque(maxlen=memory)  # call events
        # 3bet
        self.bet3_opps  = collections.deque(maxlen=memory)
        self.bet3_acts  = collections.deque(maxlen=memory)
        # Fold to 3bet
        self.f3b_opps   = collections.deque(maxlen=memory)
        self.f3b_acts   = collections.deque(maxlen=memory)
        # Cbet
        self.cbet_opps  = collections.deque(maxlen=memory)
        self.cbet_acts  = collections.deque(maxlen=memory)
        # Fold to cbet
        self.fcb_opps   = collections.deque(maxlen=memory)
        self.fcb_acts   = collections.deque(maxlen=memory)

    def _ratio(self, acts, opps) -> float:
        n = len(opps)
        if n == 0:
            return NEUTRAL_PRIOR
        return sum(acts) / n

    def _af(self) -> float:
        agg = sum(self.aggressive)
        pas = sum(self.passive)
        if pas == 0:
            return NEUTRAL_PRIOR if agg == 0 else min(agg / 1.0, 5.0) / 5.0
        return min(agg / pas, 5.0) / 5.0  # normált 0-1-re, cap=5

    def vector(self) -> list:
        return [
            self._ratio(self.vpip_acts,  self.vpip_opps),
            self._ratio(self.pfr_acts,   self.pfr_opps),
            self._af(),
            self._ratio(self.bet3_acts,  self.bet3_opps),
            self._ratio(self.f3b_acts,   self.f3b_opps),
            self._ratio(self.cbet_acts,  self.cbet_opps),
            self._ratio(self.fcb_acts,   self.fcb_opps),
        ]


class OpponentHUDTracker:
    """
    Multi-player HUD tracker.

    Használat:
        tracker = OpponentHUDTracker(num_players=6)

        # Preflop: minden aktív játékosnál
        tracker.record_preflop_action(player_id=2, action=1,  # call
                                      is_preflop_aggressor=False,
                                      facing_3bet=False)

        # Flop: c-bet lehetőség
        tracker.record_postflop_action(player_id=0, action=4,  # bet
                                       is_cbet_opportunity=True,
                                       facing_cbet=False)

        stats = tracker.get_stats_vector()  # flat list, num_players × 7
    """

    def __init__(self, num_players: int, memory: int = 1000):
        self.num_players = num_players
        self.memory      = memory
        self._players    = [PlayerStats(memory) for _ in range(num_players)]

    def reset(self):
        """Teljes reset (új session)."""
        self._players = [PlayerStats(self.memory)
                         for _ in range(self.num_players)]

    def _p(self, player_id: int) -> PlayerStats:
        if 0 <= player_id < self.num_players:
            return self._players[player_id]
        return self._players[0]  # fallback

    # ──────────────────────────────────────────────────────────────────────────
    # Esemény rekordálás
    # ──────────────────────────────────────────────────────────────────────────

    def record_preflop_action(self, player_id: int, abstract_action: int,
                               facing_open: bool = True,
                               facing_3bet: bool = False):
        """
        Preflop akció rögzítése.

        abstract_action:
          0=fold, 1=call/check, 2-6=raise tiers

        VPIP: call/raise volt-e (abstract ≥ 1 és nem BB check)
        PFR:  raise volt-e (abstract ≥ 2)
        3bet: facing_open=False, van már raise → ezt 3bet oppnak számítjuk
              egyszerűsítés: ha abstract≥2 és facing_open → PFR, else 3bet_opp
        """
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        # VPIP
        p.vpip_opps.append(1)
        p.vpip_acts.append(1 if (is_call or is_raise) else 0)

        # PFR
        p.pfr_opps.append(1)
        p.pfr_acts.append(1 if is_raise and not facing_3bet else 0)

        # 3bet
        if facing_3bet:
            p.bet3_opps.append(1)
            p.bet3_acts.append(1 if is_raise else 0)
            # Fold to 3bet
            p.f3b_opps.append(1)
            p.f3b_acts.append(1 if abstract_action == 0 else 0)

        # AF
        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

    def record_postflop_action(self, player_id: int, abstract_action: int,
                                is_cbet_opportunity: bool = False,
                                facing_cbet: bool = False):
        """
        Postflop akció rögzítése (flop/turn/river).
        """
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        # AF
        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

        # Cbet
        if is_cbet_opportunity:
            p.cbet_opps.append(1)
            p.cbet_acts.append(1 if is_raise else 0)

        # Fold to cbet
        if facing_cbet:
            p.fcb_opps.append(1)
            p.fcb_acts.append(1 if abstract_action == 0 else 0)

    def record_action(self, player_id: int, abstract_action: int,
                      street: int = 0, context: dict = None):
        """
        Egyszerűsített rekordálás (collector használja).
        street: 0=preflop, 1=flop, 2=turn, 3=river
        context: opcionális {'facing_3bet': bool, 'is_cbet_opp': bool, ...}
        """
        ctx = context or {}
        if street == 0:
            self.record_preflop_action(
                player_id, abstract_action,
                facing_open=ctx.get('facing_open', True),
                facing_3bet=ctx.get('facing_3bet', False),
            )
        else:
            self.record_postflop_action(
                player_id, abstract_action,
                is_cbet_opportunity=ctx.get('is_cbet_opp', False),
                facing_cbet=ctx.get('facing_cbet', False),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # State vector (feature engineering input)
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats_vector(self) -> list:
        """
        Flat lista: num_players × NUM_HUD_STATS float.
        Sorrend: player_0_stat0, player_0_stat1, ..., player_N_statM
        """
        result = []
        for p in self._players:
            result.extend(p.vector())
        return result
