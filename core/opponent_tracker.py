"""
core/opponent_tracker.py  –  OpponentHUDTracker + GlobalPlayerTracker

v4 újítás: valódi póker HUD statisztikák per játékos.
Régi (v3.x): nyers action frequency → nem informatív.
Új (v4):     VPIP, PFR, AF, 3bet%, fold_to_3bet, cbet%, fold_to_cbet.

Hideg start probléma: 0 helyett 0.5 neutral prior – elkerüli a hamis
"soha nem raised" következtetést az első néhány kézben.

Memory: rolling deque per stat per player (maxlen=1000 event).

RTA v2 újítás: GlobalPlayerTracker
  Az OpponentHUDTracker asztali indexek alapján tárol (0..N-1).
  A GlobalPlayerTracker username alapján tárol (dict[str, PlayerStats]),
  így modellváltás és asztalméret-változás esetén sem vesznek el az adatok.
  get_local_stats_vector(seat_map) generálja a build_state_tensor()-hoz
  szükséges flat vektort a széksorrend alapján.
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

# Fallback username prefix ismeretlen székekhez
_UNKNOWN_PREFIX = "unknown_seat_"


class PlayerStats:
    """Egy játékos HUD statisztikái."""

    def __init__(self, memory: int = 1000):
        # VPIP
        self.vpip_opps  = collections.deque(maxlen=memory)
        self.vpip_acts  = collections.deque(maxlen=memory)
        # PFR
        self.pfr_opps   = collections.deque(maxlen=memory)
        self.pfr_acts   = collections.deque(maxlen=memory)
        # AF komponensek
        self.aggressive = collections.deque(maxlen=memory)
        self.passive    = collections.deque(maxlen=memory)
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
        return min(agg / pas, 5.0) / 5.0

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

    def summary(self) -> dict:
        """Human-readable stat dict debuggoláshoz / logoláshoz."""
        v = self.vector()
        return {
            'VPIP':         f'{v[STAT_VPIP]*100:.0f}%',
            'PFR':          f'{v[STAT_PFR]*100:.0f}%',
            'AF':           f'{v[STAT_AF]*5:.1f}',
            '3bet%':        f'{v[STAT_3BET]*100:.0f}%',
            'fold_to_3bet': f'{v[STAT_FOLD_TO_3BET]*100:.0f}%',
            'cbet%':        f'{v[STAT_CBET]*100:.0f}%',
            'fold_to_cbet': f'{v[STAT_FOLD_TO_CBET]*100:.0f}%',
            'hands_seen':   len(self.vpip_opps),
        }


# ─────────────────────────────────────────────────────────────────────────────
# OpponentHUDTracker  (meglévő – training pipeline és egyszerű RTA használja)
# ─────────────────────────────────────────────────────────────────────────────

class OpponentHUDTracker:
    """
    Multi-player HUD tracker – lokális asztali indexek alapján.
    Változatlan a training pipeline számára.

    RTA multi-asztalos használathoz lásd: GlobalPlayerTracker.
    """

    def __init__(self, num_players: int, memory: int = 1000):
        self.num_players = num_players
        self.memory      = memory
        self._players    = [PlayerStats(memory) for _ in range(num_players)]

    def reset(self):
        self._players = [PlayerStats(self.memory)
                         for _ in range(self.num_players)]

    def _p(self, player_id: int) -> PlayerStats:
        if 0 <= player_id < self.num_players:
            return self._players[player_id]
        return self._players[0]

    def record_preflop_action(self, player_id: int, abstract_action: int,
                               facing_open: bool = True,
                               facing_3bet: bool = False):
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        p.vpip_opps.append(1)
        p.vpip_acts.append(1 if (is_call or is_raise) else 0)

        p.pfr_opps.append(1)
        p.pfr_acts.append(1 if is_raise and not facing_3bet else 0)

        if facing_3bet:
            p.bet3_opps.append(1)
            p.bet3_acts.append(1 if is_raise else 0)
            p.f3b_opps.append(1)
            p.f3b_acts.append(1 if abstract_action == 0 else 0)

        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

    def record_postflop_action(self, player_id: int, abstract_action: int,
                                is_cbet_opportunity: bool = False,
                                facing_cbet: bool = False):
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

        if is_cbet_opportunity:
            p.cbet_opps.append(1)
            p.cbet_acts.append(1 if is_raise else 0)

        if facing_cbet:
            p.fcb_opps.append(1)
            p.fcb_acts.append(1 if abstract_action == 0 else 0)

    def record_action(self, player_id: int, abstract_action: int,
                      street: int = 0, context: dict = None):
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

    def get_stats_vector(self) -> list:
        result = []
        for p in self._players:
            result.extend(p.vector())
        return result


# ─────────────────────────────────────────────────────────────────────────────
# GlobalPlayerTracker  (RTA v2 – username alapú, modellváltás-biztos)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalPlayerTracker:
    """
    Username-alapú HUD tracker online RTA használathoz.

    Az OpponentHUDTracker-rel ellentétben ez NEM kötődik asztali szék-
    indexekhez. A statisztikák a teljes session alatt megmaradnak,
    akkor is ha:
      - az asztalméret megváltozik (modellváltás)
      - ugyanaz az ellenfél másik székbe ül
      - az asztalon kevesebben maradnak

    Kulcs: username (str). Ha a kliens nem ad username-t,
    automatikusan "unknown_seat_N" fallback kulcsot kap.

    Példa:
        tracker = GlobalPlayerTracker()

        # Akció rögzítése
        tracker.record_action('fish99', abstract_action=1, street=0)

        # Flat vektor a build_state_tensor()-hoz (6p asztal)
        seat_map = {0: 'hero', 1: 'fish99', 2: 'reg42',
                    3: 'nit77', 4: 'fish99', 5: 'unknown_seat_5'}
        vec = tracker.get_local_stats_vector(seat_map)
        # → list, len = num_players × 7
    """

    def __init__(self, memory: int = 1000):
        self.memory   = memory
        self._players: dict[str, PlayerStats] = {}

    def _get_or_create(self, username: str) -> PlayerStats:
        """PlayerStats lekérése vagy létrehozása username alapján."""
        if username not in self._players:
            self._players[username] = PlayerStats(self.memory)
        return self._players[username]

    def reset(self):
        """Teljes reset – csak session váltáskor hívandó."""
        self._players.clear()

    def reset_player(self, username: str):
        """Egyetlen játékos statjainak törlése."""
        self._players.pop(username, None)

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_preflop_action(self, username: str, abstract_action: int,
                               facing_open: bool = True,
                               facing_3bet: bool = False):
        p        = self._get_or_create(username)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        p.vpip_opps.append(1)
        p.vpip_acts.append(1 if (is_call or is_raise) else 0)

        p.pfr_opps.append(1)
        p.pfr_acts.append(1 if is_raise and not facing_3bet else 0)

        if facing_3bet:
            p.bet3_opps.append(1)
            p.bet3_acts.append(1 if is_raise else 0)
            p.f3b_opps.append(1)
            p.f3b_acts.append(1 if abstract_action == 0 else 0)

        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

    def record_postflop_action(self, username: str, abstract_action: int,
                                is_cbet_opportunity: bool = False,
                                facing_cbet: bool = False):
        p        = self._get_or_create(username)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1

        if is_raise:
            p.aggressive.append(1)
        elif is_call:
            p.passive.append(1)

        if is_cbet_opportunity:
            p.cbet_opps.append(1)
            p.cbet_acts.append(1 if is_raise else 0)

        if facing_cbet:
            p.fcb_opps.append(1)
            p.fcb_acts.append(1 if abstract_action == 0 else 0)

    def record_action(self, username: str, abstract_action: int,
                      street: int = 0, context: dict = None):
        """
        Egyszerűsített rekordálás.
        street: 0=preflop, 1=flop, 2=turn, 3=river
        context: {'facing_3bet': bool, 'is_cbet_opp': bool, 'facing_cbet': bool}
        """
        ctx = context or {}
        if street == 0:
            self.record_preflop_action(
                username, abstract_action,
                facing_open=ctx.get('facing_open', True),
                facing_3bet=ctx.get('facing_3bet', False),
            )
        else:
            self.record_postflop_action(
                username, abstract_action,
                is_cbet_opportunity=ctx.get('is_cbet_opp', False),
                facing_cbet=ctx.get('facing_cbet', False),
            )

    # ── State vector (feature engineering input) ──────────────────────────────

    def get_local_stats_vector(self, seat_map: dict) -> list:
        """
        Flat lista a build_state_tensor() számára.

        Paraméter:
            seat_map: dict[int, str]  – {szék_index: username}
                      Széksorrend szerint rendezi és generálja a vektort.
                      Hiányzó username → neutral prior (0.5 minden stat).

        Visszatér: list[float], len = len(seat_map) × NUM_HUD_STATS

        Példa:
            seat_map = {0: 'hero', 1: 'fish99', 2: 'reg42'}
            vec = tracker.get_local_stats_vector(seat_map)
            # → [hero_stat0..6, fish99_stat0..6, reg42_stat0..6]
        """
        result = []
        for seat_idx in sorted(seat_map.keys()):
            username = seat_map[seat_idx]
            if not username:
                username = f"{_UNKNOWN_PREFIX}{seat_idx}"
            if username in self._players:
                result.extend(self._players[username].vector())
            else:
                # Cold start: neutral prior minden stathoz
                result.extend([NEUTRAL_PRIOR] * NUM_HUD_STATS)
        return result

    def get_stats_vector_by_order(self, username_list: list) -> list:
        """
        Alternatív: username lista sorrendje szerint generál vektort.
        Hasznosabb ha a kliens direkt sorrendben adja az asztal játékosait.
        """
        result = []
        for username in username_list:
            if username and username in self._players:
                result.extend(self._players[username].vector())
            else:
                result.extend([NEUTRAL_PRIOR] * NUM_HUD_STATS)
        return result

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def known_players(self) -> list:
        """Ismert játékosok usernevei."""
        return list(self._players.keys())

    def player_summary(self, username: str) -> dict:
        """Egy játékos stat summaryja debuggoláshoz."""
        if username not in self._players:
            return {'error': 'unknown player', 'username': username}
        return {'username': username, **self._players[username].summary()}

    def all_summaries(self) -> list:
        """Összes ismert játékos summarya."""
        return [self.player_summary(u) for u in self._players]

    def __len__(self):
        return len(self._players)

    def __repr__(self):
        return f"GlobalPlayerTracker({len(self._players)} players tracked)"
