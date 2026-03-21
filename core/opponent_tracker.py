"""
core/opponent_tracker.py  –  OpponentHUDTracker + GlobalPlayerTracker

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITEKTÚRA (RTA v4 – hibrid)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PlayerStats – két réteg:

  1. Rolling window (memória, deque maxlen=1000)
     Az aktuális session eseményei. A modell ezt látja.
     Session végén eldobódik – nem kell perzisztálni.
     500k játékosnál sem probléma mert csak az asztalon
     ülők rolling window-ja él memóriában egyszerre.

  2. Lifetime számlálók (SQLite, csak int-ek)
     Minden valaha látott esemény összege.
     Soha nem törlődik, session-független.
     14 int per játékos → ~8 MB / 50k játékos.

  Cold start blending:
     Ha rolling < 30 esemény: lifetime stat lineárisan bekeveredik.
     Így egy régen látott ellenfél adatai azonnal hasznosak.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKÁLÁZHATÓSÁG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  50k játékos:  ~8 MB SQLite, betöltés on-demand (<1ms/játékos)
  500k játékos: ~80 MB SQLite, ugyanolyan gyors (indexed lookup)
  RAM:          csak az asztalon ülők rolling window-ja él
                max 9 PlayerStats objektum egyszerre aktív

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  tracker = GlobalPlayerTracker('players.db')
  tracker.record_action('fish99', abstract_action=1, street=0)
  vec = tracker.get_local_stats_vector({0: 'hero', 1: 'fish99'})
  tracker.flush()   # session végén (RTAManager auto-hívja)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISSZAFELÉ KOMPATIBILITÁS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OpponentHUDTracker:  változatlan (training pipeline)
  GlobalPlayerTracker: API változatlan (RTAManager kompatibilis)
  JSON → SQLite:       GlobalPlayerTracker.migrate_from_json()

VÁLTOZÁSOK:
  [RF-6 FIX] _PlayerDB thread-local connection pool.
    Minden query új kapcsolatot nyitott → ~0.5-2ms overhead/query.
    Mostantól threading.local() tárolja a szálankénti kapcsolatot,
    ami egyszer nyílik meg és nyitva marad. ~10-50× gyorsabb
    ismételt lookupok esetén. Élő kliensben különösen érzékelhető.
  [RF-10 FIX] _BLEND_THRESHOLD 10 → 50 (Bayesian prior stabilabb).
"""

import collections
import json as _json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger("PokerAI")

# ─────────────────────────────────────────────────────────────────────────────
# Konstansok
# ─────────────────────────────────────────────────────────────────────────────

STAT_VPIP         = 0
STAT_PFR          = 1
STAT_AF           = 2
STAT_3BET         = 3
STAT_FOLD_TO_3BET = 4
STAT_CBET         = 5
STAT_FOLD_TO_CBET = 6

NUM_HUD_STATS    = 7
NEUTRAL_PRIOR    = 0.5
_UNKNOWN_PREFIX  = "unknown_seat_"
_BLEND_THRESHOLD = 50    # [RF-10 FIX] 10→50: 10 kéz statisztikailag nem szignifikáns.
                         # 50 kéznél a Bayesian prior blending stabilabb
                         # korai reads-t ad NL10-NL25 populáción.
_FLUSH_INTERVAL  = 50    # ennyi akció után auto-flush SQLite-ba

# Bayesian prior – tipikus microstakes/low-stakes stat értékek
# Ismeretlen játékosnál ez az alap-feltételezés (jobb mint a 0.5 mindenhol).
# Forrás: általános NL10-NL25 populációs átlagok.
# NEUTRAL_PRIOR (0.5) marad fallback ismeretlen stat_id-re.
STAT_PRIORS = {
    STAT_VPIP:         0.24,   # átlagos VPIP ~24%
    STAT_PFR:          0.16,   # átlagos PFR ~16%
    STAT_AF:           0.35,   # közepes AF (normalizálva 0-1 skálán)
    STAT_3BET:         0.07,   # 3bet % ~7%
    STAT_FOLD_TO_3BET: 0.55,   # fold to 3bet ~55%
    STAT_CBET:         0.58,   # cbet % ~58%
    STAT_FOLD_TO_CBET: 0.46,   # fold to cbet ~46%
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ─────────────────────────────────────────────────────────────────────────────
# PlayerStats – rolling window (memória) + lifetime (SQLite-ból töltve)
# ─────────────────────────────────────────────────────────────────────────────

class PlayerStats:
    """
    Egy játékos HUD statisztikái.

    Rolling window: session-élettartamú, gyors, adaptív.
    Lifetime int-ek: SQLite-ból töltődnek az első érintéskor,
                     flush()-kor visszaíródnak.

    Külső kód NEM példányosítja közvetlenül.
    """

    LT_ATTRS = (
        'lt_vpip_opps', 'lt_vpip_acts',
        'lt_pfr_opps',  'lt_pfr_acts',
        'lt_aggressive','lt_passive',
        'lt_bet3_opps', 'lt_bet3_acts',
        'lt_f3b_opps',  'lt_f3b_acts',
        'lt_cbet_opps', 'lt_cbet_acts',
        'lt_fcb_opps',  'lt_fcb_acts',
    )

    ROLLING_ATTRS = (
        'vpip_opps', 'vpip_acts', 'pfr_opps', 'pfr_acts',
        'aggressive', 'passive',
        'bet3_opps', 'bet3_acts', 'f3b_opps', 'f3b_acts',
        'cbet_opps', 'cbet_acts', 'fcb_opps', 'fcb_acts',
    )

    def __init__(self, memory: int = 1000):
        self.memory = memory
        for attr in self.ROLLING_ATTRS:
            setattr(self, attr, collections.deque(maxlen=memory))
        for attr in self.LT_ATTRS:
            setattr(self, attr, 0)
        self.first_seen = _now_iso()
        self.last_seen  = _now_iso()
        self._dirty     = False

    # ── Számítások ────────────────────────────────────────────────────────────

    def _ratio(self, acts, opps, lt_acts: int, lt_opps: int,
               prior: float = NEUTRAL_PRIOR) -> float:
        """
        Rolling + lifetime blending Bayesian prior-ral.
        Ha nincs semmilyen adat, prior-t ad vissza (tipikus populációs érték)
        a 0.5 semleges érték helyett.
        """
        n = len(opps)
        if n == 0 and lt_opps == 0:
            return prior
        rolling_r  = sum(acts) / n if n > 0 else prior
        lifetime_r = lt_acts / lt_opps if lt_opps > 0 else prior
        if n >= _BLEND_THRESHOLD:
            return rolling_r
        blend = n / _BLEND_THRESHOLD
        return blend * rolling_r + (1.0 - blend) * lifetime_r

    def _af(self) -> float:
        r_agg = sum(self.aggressive)
        r_pas = sum(self.passive)
        r_n   = r_agg + r_pas
        lt_n  = self.lt_aggressive + self.lt_passive

        def _val(agg, pas):
            if pas == 0:
                return NEUTRAL_PRIOR if agg == 0 else min(agg, 5.0) / 5.0
            return min(agg / pas, 5.0) / 5.0

        _AF_PRIOR = STAT_PRIORS.get(STAT_AF, NEUTRAL_PRIOR)
        if r_n == 0 and lt_n == 0:
            return _AF_PRIOR
        r_af  = _val(r_agg, r_pas)
        lt_af = _val(self.lt_aggressive, self.lt_passive)
        if r_n >= _BLEND_THRESHOLD:
            return r_af
        blend = r_n / _BLEND_THRESHOLD
        # Kevés adatnál prior felé húz
        base = blend * r_af + (1.0 - blend) * lt_af
        prior_blend = min(r_n / _BLEND_THRESHOLD, 1.0)
        return prior_blend * base + (1.0 - prior_blend) * _AF_PRIOR

    def vector(self) -> list:
        """7 float [0-1] – feature vector a modellnek (Bayesian prior-ral)."""
        p = STAT_PRIORS  # rövidítés
        return [
            self._ratio(self.vpip_acts, self.vpip_opps,
                        self.lt_vpip_acts, self.lt_vpip_opps,
                        prior=p.get(STAT_VPIP, NEUTRAL_PRIOR)),
            self._ratio(self.pfr_acts,  self.pfr_opps,
                        self.lt_pfr_acts,  self.lt_pfr_opps,
                        prior=p.get(STAT_PFR, NEUTRAL_PRIOR)),
            self._af(),
            self._ratio(self.bet3_acts, self.bet3_opps,
                        self.lt_bet3_acts, self.lt_bet3_opps,
                        prior=p.get(STAT_3BET, NEUTRAL_PRIOR)),
            self._ratio(self.f3b_acts,  self.f3b_opps,
                        self.lt_f3b_acts,  self.lt_f3b_opps,
                        prior=p.get(STAT_FOLD_TO_3BET, NEUTRAL_PRIOR)),
            self._ratio(self.cbet_acts, self.cbet_opps,
                        self.lt_cbet_acts, self.lt_cbet_opps,
                        prior=p.get(STAT_CBET, NEUTRAL_PRIOR)),
            self._ratio(self.fcb_acts,  self.fcb_opps,
                        self.lt_fcb_acts,  self.lt_fcb_opps,
                        prior=p.get(STAT_FOLD_TO_CBET, NEUTRAL_PRIOR)),
        ]

    def summary(self) -> dict:
        v = self.vector()
        def _pct(a, b): return f'{a/b*100:.0f}%' if b > 0 else 'n/a'
        return {
            'VPIP':           f'{v[STAT_VPIP]*100:.0f}%',
            'PFR':            f'{v[STAT_PFR]*100:.0f}%',
            'AF':             f'{v[STAT_AF]*5:.2f}',
            '3bet%':          f'{v[STAT_3BET]*100:.0f}%',
            'fold_to_3bet':   f'{v[STAT_FOLD_TO_3BET]*100:.0f}%',
            'cbet%':          f'{v[STAT_CBET]*100:.0f}%',
            'fold_to_cbet':   f'{v[STAT_FOLD_TO_CBET]*100:.0f}%',
            'rolling_hands':  len(self.vpip_opps),
            'lifetime_hands': self.lt_vpip_opps,
            'lt_VPIP':        _pct(self.lt_vpip_acts, self.lt_vpip_opps),
            'lt_PFR':         _pct(self.lt_pfr_acts,  self.lt_pfr_opps),
            'first_seen':     self.first_seen,
            'last_seen':      self.last_seen,
        }

    def lt_row(self) -> tuple:
        """SQLite upsert-hez: lifetime értékek sorban."""
        return tuple(getattr(self, a) for a in self.LT_ATTRS)

    def load_lt_from_row(self, row: dict):
        """SQLite sor → lifetime attribútumok."""
        for attr in self.LT_ATTRS:
            setattr(self, attr, int(row.get(attr, 0)))
        self.first_seen = row.get('first_seen', _now_iso())
        self.last_seen  = row.get('last_seen',  _now_iso())
        self._dirty = False

    def clear_rolling(self):
        for attr in self.ROLLING_ATTRS:
            getattr(self, attr).clear()


# ─────────────────────────────────────────────────────────────────────────────
# SQLite backend
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    username        TEXT PRIMARY KEY,
    lt_vpip_opps    INTEGER DEFAULT 0,
    lt_vpip_acts    INTEGER DEFAULT 0,
    lt_pfr_opps     INTEGER DEFAULT 0,
    lt_pfr_acts     INTEGER DEFAULT 0,
    lt_aggressive   INTEGER DEFAULT 0,
    lt_passive      INTEGER DEFAULT 0,
    lt_bet3_opps    INTEGER DEFAULT 0,
    lt_bet3_acts    INTEGER DEFAULT 0,
    lt_f3b_opps     INTEGER DEFAULT 0,
    lt_f3b_acts     INTEGER DEFAULT 0,
    lt_cbet_opps    INTEGER DEFAULT 0,
    lt_cbet_acts    INTEGER DEFAULT 0,
    lt_fcb_opps     INTEGER DEFAULT 0,
    lt_fcb_acts     INTEGER DEFAULT 0,
    first_seen      TEXT,
    last_seen       TEXT
);
CREATE INDEX IF NOT EXISTS idx_last_seen ON players(last_seen);
CREATE INDEX IF NOT EXISTS idx_lt_hands  ON players(lt_vpip_opps);
"""

_UPSERT = """
INSERT INTO players (
    username,
    lt_vpip_opps, lt_vpip_acts, lt_pfr_opps, lt_pfr_acts,
    lt_aggressive, lt_passive,
    lt_bet3_opps, lt_bet3_acts, lt_f3b_opps, lt_f3b_acts,
    lt_cbet_opps, lt_cbet_acts, lt_fcb_opps, lt_fcb_acts,
    first_seen, last_seen
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(username) DO UPDATE SET
    lt_vpip_opps  = excluded.lt_vpip_opps,
    lt_vpip_acts  = excluded.lt_vpip_acts,
    lt_pfr_opps   = excluded.lt_pfr_opps,
    lt_pfr_acts   = excluded.lt_pfr_acts,
    lt_aggressive = excluded.lt_aggressive,
    lt_passive    = excluded.lt_passive,
    lt_bet3_opps  = excluded.lt_bet3_opps,
    lt_bet3_acts  = excluded.lt_bet3_acts,
    lt_f3b_opps   = excluded.lt_f3b_opps,
    lt_f3b_acts   = excluded.lt_f3b_acts,
    lt_cbet_opps  = excluded.lt_cbet_opps,
    lt_cbet_acts  = excluded.lt_cbet_acts,
    lt_fcb_opps   = excluded.lt_fcb_opps,
    lt_fcb_acts   = excluded.lt_fcb_acts,
    last_seen     = excluded.last_seen;
"""


class _PlayerDB:
    """
    SQLite wrapper. Thread-safe WAL módban.

    [RF-6 FIX] Thread-local connection pool.

    Eredeti: minden _connect() hívás új sqlite3.connect() → open/close
    overhead minden query-nél. Élő kliensben frequent opponent stat lookup
    esetén ez bottleneck (~0.5-2ms/query csak a kapcsolatnyitásra).

    Új: threading.local() alapú connection pool. Minden szál saját
    kapcsolatot kap, amit egyszer nyit meg és nyitva tart. A PRAGMA-k
    és row_factory csak egyszer kerülnek beállításra per-szál.

    Teljesítmény: ~10-50× gyorsabb ismételt query-k esetén (connection
    reuse), mert nincs sqlite3.connect() + PRAGMA + row_factory overhead.

    Thread-safety: az SQLite WAL mód lehetővé teszi párhuzamos olvasást;
    az írás (flush) explicit Lock-kal védett, ahogy korábban is.
    """

    def __init__(self, db_path: str):
        self._path  = db_path
        self._lock  = threading.Lock()
        self._local = threading.local()   # [RF-6] per-szál connection tároló
        self._init_db()

    # ── [RF-6 FIX] Thread-local connection ───────────────────────────────
    def _get_conn(self) -> sqlite3.Connection:
        """
        Visszaadja a szálhoz tartozó SQLite kapcsolatot.
        Ha nincs még nyitva, megnyitja és beállítja a PRAGMA-kat.
        Az első hívás után a kapcsolat nyitva marad a szál életciklusán át.
        """
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(
                self._path,
                timeout=10,
                check_same_thread=False,  # thread-local = szálankénti 1 conn → biztonságos
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-4000")
            self._local.conn = conn
            logger.debug(f"Új SQLite kapcsolat nyitva szálhoz: {threading.current_thread().name}")
        return conn

    def _connect(self) -> sqlite3.Connection:
        """
        Visszafelé kompatibilis alias – belső kód hívja.
        [RF-6 FIX] Mostantól thread-local kapcsolatot ad vissza,
        nem nyit új kapcsolatot minden híváskor.
        """
        return self._get_conn()

    def close_thread_connection(self):
        """
        Bezárja az aktuális szál SQLite kapcsolatát.
        Főszálban, session végén hívandó – opcionális, de jó practice.
        """
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def load_player(self, username: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM players WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None

    def load_players(self, usernames: list) -> dict:
        if not usernames:
            return {}
        ph = ','.join('?' * len(usernames))
        rows = self._get_conn().execute(
            f"SELECT * FROM players WHERE username IN ({ph})", usernames
        ).fetchall()
        return {r['username']: dict(r) for r in rows}

    def flush(self, dirty_players: dict):
        """Batch upsert – csak dirty játékosok. Write Lock védi az egyidejű írást."""
        if not dirty_players:
            return
        rows = [
            (u,) + ps.lt_row() + (ps.first_seen, ps.last_seen)
            for u, ps in dirty_players.items()
        ]
        with self._lock:
            conn = self._get_conn()
            conn.executemany(_UPSERT, rows)
            conn.commit()
        logger.debug(f"DB flush: {len(rows)} játékos")

    def count(self) -> int:
        return self._get_conn().execute(
            "SELECT COUNT(*) FROM players"
        ).fetchone()[0]

    def top_by_hands(self, n: int = 20) -> list:
        return [dict(r) for r in self._get_conn().execute(
            "SELECT * FROM players ORDER BY lt_vpip_opps DESC LIMIT ?", (n,)
        ).fetchall()]

    def last_seen_after(self, iso_date: str) -> list:
        return [dict(r) for r in self._get_conn().execute(
            "SELECT * FROM players WHERE last_seen >= ? ORDER BY last_seen DESC",
            (iso_date,)
        ).fetchall()]

    def db_size_mb(self) -> float:
        try:
            return os.path.getsize(self._path) / 1024 / 1024
        except OSError:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OpponentHUDTracker – változatlan, training pipeline használja
# ─────────────────────────────────────────────────────────────────────────────

class OpponentHUDTracker:
    """Lokális index alapú tracker – training pipeline és egyszerű RTA."""

    def __init__(self, num_players: int, memory: int = 1000):
        self.num_players = num_players
        self.memory      = memory
        self._players    = [PlayerStats(memory) for _ in range(num_players)]

    def reset(self):
        self._players = [PlayerStats(self.memory) for _ in range(self.num_players)]

    def _p(self, player_id: int) -> PlayerStats:
        return self._players[player_id] if 0 <= player_id < self.num_players \
               else self._players[0]

    def record_preflop_action(self, player_id: int, abstract_action: int,
                               facing_open: bool = True, facing_3bet: bool = False):
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1
        p.vpip_opps.append(1); p.vpip_acts.append(1 if (is_call or is_raise) else 0)
        p.pfr_opps.append(1);  p.pfr_acts.append(1 if is_raise and not facing_3bet else 0)
        p.lt_vpip_opps += 1;   p.lt_vpip_acts += 1 if (is_call or is_raise) else 0
        p.lt_pfr_opps  += 1;   p.lt_pfr_acts  += 1 if is_raise and not facing_3bet else 0
        if facing_3bet:
            p.bet3_opps.append(1); p.bet3_acts.append(1 if is_raise else 0)
            p.f3b_opps.append(1);  p.f3b_acts.append(1 if abstract_action == 0 else 0)
            p.lt_bet3_opps += 1;   p.lt_bet3_acts += 1 if is_raise else 0
            p.lt_f3b_opps  += 1;   p.lt_f3b_acts  += 1 if abstract_action == 0 else 0
        if is_raise:  p.aggressive.append(1); p.lt_aggressive += 1
        elif is_call: p.passive.append(1);    p.lt_passive    += 1
        p._dirty = True; p.last_seen = _now_iso()

    def record_postflop_action(self, player_id: int, abstract_action: int,
                                is_cbet_opportunity: bool = False,
                                facing_cbet: bool = False):
        p = self._p(player_id)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1
        if is_raise:  p.aggressive.append(1); p.lt_aggressive += 1
        elif is_call: p.passive.append(1);    p.lt_passive    += 1
        if is_cbet_opportunity:
            p.cbet_opps.append(1); p.cbet_acts.append(1 if is_raise else 0)
            p.lt_cbet_opps += 1;   p.lt_cbet_acts += 1 if is_raise else 0
        if facing_cbet:
            p.fcb_opps.append(1);  p.fcb_acts.append(1 if abstract_action == 0 else 0)
            p.lt_fcb_opps  += 1;   p.lt_fcb_acts  += 1 if abstract_action == 0 else 0
        p._dirty = True; p.last_seen = _now_iso()

    def record_action(self, player_id: int, abstract_action: int,
                      street: int = 0, context: dict = None):
        ctx = context or {}
        if street == 0:
            self.record_preflop_action(
                player_id, abstract_action,
                facing_open=ctx.get('facing_open', True),
                facing_3bet=ctx.get('facing_3bet', False))
        else:
            self.record_postflop_action(
                player_id, abstract_action,
                is_cbet_opportunity=ctx.get('is_cbet_opp', False),
                facing_cbet=ctx.get('facing_cbet', False))

    def get_stats_vector(self) -> list:
        result = []
        for p in self._players:
            result.extend(p.vector())
        return result


# ─────────────────────────────────────────────────────────────────────────────
# GlobalPlayerTracker – RTA v4, SQLite hibrid
# ─────────────────────────────────────────────────────────────────────────────

class GlobalPlayerTracker:
    """
    Username-alapú HUD tracker. SQLite lifetime + memória rolling hibrid.

    Skálázhatóság:
      50k játékos  → ~8 MB SQLite, <1ms/játékos lookup
      500k játékos → ~80 MB SQLite, azonos sebesség
      RAM          → max ~9 PlayerStats él egyszerre (asztalon ülők)
    """

    def __init__(self, db_path: str = None, memory: int = 1000):
        """
        db_path: SQLite fájl. None → csak memória (tesztelés, régi API).
        memory:  rolling window mérete.
        """
        self.memory   = memory
        self._db_path = db_path
        self._db      = _PlayerDB(db_path) if db_path else None

        self._cache:    dict[str, PlayerStats] = {}   # session cache
        self._lt_cache: dict[str, dict]        = {}   # lifetime batch cache
        self._actions_since_flush = 0
        # Thread-safe cache hozzáférés – RLock, mert flush() rekurzívan hívja
        # a _get_or_create-t a dirty check előtt (re-entrant szükséges)
        self._lock = threading.RLock()

    # ── Belső ────────────────────────────────────────────────────────────────

    def _get_or_create(self, username: str) -> PlayerStats:
        with self._lock:
            if username in self._cache:
                return self._cache[username]
            ps = PlayerStats(self.memory)
            if self._db is not None:
                row = self._lt_cache.get(username) or self._db.load_player(username)
                if row:
                    ps.load_lt_from_row(row)
                    self._lt_cache[username] = row
            self._cache[username] = ps
            return ps

    # ── Asztal preload (batch DB query) ───────────────────────────────────────

    def preload_table(self, seat_map: dict):
        """
        Asztal megnyitásakor hívandó.
        Egyetlen DB query az asztalon ülő összes játékoshoz.
        RTAManager automatikusan hívja manage_table_change()-ben.
        """
        if self._db is None:
            return
        with self._lock:
            to_load = [
                u for u in seat_map.values()
                if u and not u.startswith(_UNKNOWN_PREFIX)
                and u not in self._lt_cache
            ]
        if not to_load:
            return
        # DB lekérdezés a lock-on KÍVÜL (I/O nem blokkolja a cache-t)
        rows = self._db.load_players(to_load)
        with self._lock:
            self._lt_cache.update(rows)
        logger.debug(f"Preload: {len(rows)}/{len(to_load)} játékos betöltve")

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_preflop_action(self, username: str, abstract_action: int,
                               facing_open: bool = True, facing_3bet: bool = False):
        if not username or username.startswith(_UNKNOWN_PREFIX):
            return
        p = self._get_or_create(username)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1
        p.vpip_opps.append(1); p.vpip_acts.append(1 if (is_call or is_raise) else 0)
        p.pfr_opps.append(1);  p.pfr_acts.append(1 if is_raise and not facing_3bet else 0)
        p.lt_vpip_opps += 1;   p.lt_vpip_acts += 1 if (is_call or is_raise) else 0
        p.lt_pfr_opps  += 1;   p.lt_pfr_acts  += 1 if is_raise and not facing_3bet else 0
        if facing_3bet:
            p.bet3_opps.append(1); p.bet3_acts.append(1 if is_raise else 0)
            p.f3b_opps.append(1);  p.f3b_acts.append(1 if abstract_action == 0 else 0)
            p.lt_bet3_opps += 1;   p.lt_bet3_acts += 1 if is_raise else 0
            p.lt_f3b_opps  += 1;   p.lt_f3b_acts  += 1 if abstract_action == 0 else 0
        if is_raise:  p.aggressive.append(1); p.lt_aggressive += 1
        elif is_call: p.passive.append(1);    p.lt_passive    += 1
        p._dirty = True; p.last_seen = _now_iso()
        self._actions_since_flush += 1
        self._maybe_flush()

    def record_postflop_action(self, username: str, abstract_action: int,
                                is_cbet_opportunity: bool = False,
                                facing_cbet: bool = False):
        if not username or username.startswith(_UNKNOWN_PREFIX):
            return
        p = self._get_or_create(username)
        is_raise = abstract_action >= 2
        is_call  = abstract_action == 1
        if is_raise:  p.aggressive.append(1); p.lt_aggressive += 1
        elif is_call: p.passive.append(1);    p.lt_passive    += 1
        if is_cbet_opportunity:
            p.cbet_opps.append(1); p.cbet_acts.append(1 if is_raise else 0)
            p.lt_cbet_opps += 1;   p.lt_cbet_acts += 1 if is_raise else 0
        if facing_cbet:
            p.fcb_opps.append(1);  p.fcb_acts.append(1 if abstract_action == 0 else 0)
            p.lt_fcb_opps  += 1;   p.lt_fcb_acts  += 1 if abstract_action == 0 else 0
        p._dirty = True; p.last_seen = _now_iso()
        self._actions_since_flush += 1
        self._maybe_flush()

    def record_action(self, username: str, abstract_action: int,
                      street: int = 0, context: dict = None):
        """
        Egyszerűsített rekordálás.
        street: 0=preflop, 1=flop, 2=turn, 3=river
        """
        ctx = context or {}
        if street == 0:
            self.record_preflop_action(
                username, abstract_action,
                facing_open=ctx.get('facing_open', True),
                facing_3bet=ctx.get('facing_3bet', False))
        else:
            self.record_postflop_action(
                username, abstract_action,
                is_cbet_opportunity=ctx.get('is_cbet_opp', False),
                facing_cbet=ctx.get('facing_cbet', False))

    def _maybe_flush(self):
        if self._db and self._actions_since_flush >= _FLUSH_INTERVAL:
            self.flush()

    # ── Perzisztencia ─────────────────────────────────────────────────────────

    def flush(self):
        """
        Dirty lifetime értékek → SQLite.
        RTAManager session végén automatikusan hívja.
        DB írás a lock-on KÍVÜL történik (I/O holtpont elkerülése).
        """
        if self._db is None:
            return
        # Dirty játékosok snapshot lock alatt
        with self._lock:
            dirty = {u: ps for u, ps in self._cache.items() if ps._dirty}
        # DB írás lock-on kívül (SQLite-nak saját lockja van)
        if dirty:
            self._db.flush(dirty)
            with self._lock:
                for ps in dirty.values():
                    ps._dirty = False
                self._actions_since_flush = 0
        else:
            with self._lock:
                self._actions_since_flush = 0

    # ── State vector ──────────────────────────────────────────────────────────

    def get_local_stats_vector(self, seat_map: dict) -> list:
        """
        Flat lista a build_state_tensor() számára.
        seat_map: dict[int, str] – {szék_index: username}
        """
        result = []
        for seat_idx in sorted(seat_map.keys()):
            username = seat_map[seat_idx]
            if not username or username.startswith(_UNKNOWN_PREFIX):
                result.extend([NEUTRAL_PRIOR] * NUM_HUD_STATS)
            else:
                result.extend(self._get_or_create(username).vector())
        return result

    def get_stats_vector_by_order(self, username_list: list) -> list:
        result = []
        for u in username_list:
            if u and not u.startswith(_UNKNOWN_PREFIX):
                result.extend(self._get_or_create(u).vector())
            else:
                result.extend([NEUTRAL_PRIOR] * NUM_HUD_STATS)
        return result

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Memória cache törlése. DB NEM érintett."""
        with self._lock:
            self._cache.clear()
            self._lt_cache.clear()
            self._actions_since_flush = 0

    def reset_player(self, username: str):
        """Cache-ből törlés (DB-ből nem)."""
        with self._lock:
            self._cache.pop(username, None)
            self._lt_cache.pop(username, None)

    def reset_rolling_only(self, username: str):
        """Csak rolling window törlése, lifetime megmarad."""
        with self._lock:
            if username in self._cache:
                self._cache[username].clear_rolling()

    # ── Lekérdezések ──────────────────────────────────────────────────────────

    def known_players(self) -> list:
        return [u for u in self._cache if not u.startswith(_UNKNOWN_PREFIX)]

    def player_summary(self, username: str) -> dict:
        if not username or username.startswith(_UNKNOWN_PREFIX):
            return {'error': 'unknown player', 'username': username}
        return {'username': username, **self._get_or_create(username).summary()}

    def all_summaries(self) -> list:
        return [self.player_summary(u) for u in self.known_players()]

    def top_players_by_hands(self, n: int = 20) -> list:
        """Top N játékos lifetime kézszám szerint (DB query ha elérhető)."""
        if self._db:
            return [
                {
                    'username':       r['username'],
                    'lifetime_hands': r['lt_vpip_opps'],
                    'lt_VPIP': (f"{r['lt_vpip_acts']/r['lt_vpip_opps']*100:.0f}%"
                                if r['lt_vpip_opps'] > 0 else 'n/a'),
                    'lt_PFR':  (f"{r['lt_pfr_acts']/r['lt_pfr_opps']*100:.0f}%"
                                if r['lt_pfr_opps'] > 0 else 'n/a'),
                    'last_seen': r['last_seen'],
                }
                for r in self._db.top_by_hands(n)
            ]
        ranked = sorted(
            [(u, ps.lt_vpip_opps) for u, ps in self._cache.items()
             if not u.startswith(_UNKNOWN_PREFIX)],
            key=lambda x: x[1], reverse=True
        )
        return [self.player_summary(u) for u, _ in ranked[:n]]

    def players_last_seen_after(self, iso_date: str) -> list:
        if self._db:
            return self._db.last_seen_after(iso_date)
        return [
            self.player_summary(u)
            for u, ps in self._cache.items()
            if not u.startswith(_UNKNOWN_PREFIX) and ps.last_seen >= iso_date
        ]

    def db_stats(self) -> dict:
        if self._db is None:
            return {'mode': 'memory_only', 'cached': len(self._cache)}
        return {
            'mode':          'sqlite',
            'db_path':       self._db._path,
            'db_size_mb':    round(self._db.db_size_mb(), 3),
            'total_players': self._db.count(),
            'cached':        len(self._cache),
            'dirty':         sum(1 for ps in self._cache.values() if ps._dirty),
            'pending_flush': self._actions_since_flush,
        }

    def __len__(self):
        return len(self._cache)

    def __repr__(self):
        if self._db:
            return (f"GlobalPlayerTracker(sqlite={self._db._path}, "
                    f"cached={len(self._cache)})")
        return f"GlobalPlayerTracker(memory_only, cached={len(self._cache)})"

    # ── JSON → SQLite migráció ────────────────────────────────────────────────

    @classmethod
    def migrate_from_json(cls, json_path: str, db_path: str,
                           memory: int = 1000) -> 'GlobalPlayerTracker':
        """
        Régi JSON adatbázis migrálása SQLite-ba.

        Példa:
            tracker = GlobalPlayerTracker.migrate_from_json(
                'players.json', 'players.db'
            )
            # ezután csak a .db fájlt kell használni
        """
        tracker = cls(db_path=db_path, memory=memory)
        if not os.path.exists(json_path):
            logger.warning(f"JSON fájl nem található: {json_path}")
            return tracker

        with open(json_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)

        lt_map = {
            'vpip_opps':  'lt_vpip_opps',  'vpip_acts':  'lt_vpip_acts',
            'pfr_opps':   'lt_pfr_opps',   'pfr_acts':   'lt_pfr_acts',
            'aggressive': 'lt_aggressive',  'passive':    'lt_passive',
            'bet3_opps':  'lt_bet3_opps',   'bet3_acts':  'lt_bet3_acts',
            'f3b_opps':   'lt_f3b_opps',    'f3b_acts':   'lt_f3b_acts',
            'cbet_opps':  'lt_cbet_opps',   'cbet_acts':  'lt_cbet_acts',
            'fcb_opps':   'lt_fcb_opps',    'fcb_acts':   'lt_fcb_acts',
        }

        migrated = 0
        for username, stats_dict in data.items():
            if username == '__meta__' or username.startswith(_UNKNOWN_PREFIX):
                continue
            ps = tracker._get_or_create(username)
            lt = stats_dict.get('lifetime', {})
            for json_key, attr in lt_map.items():
                setattr(ps, attr, int(lt.get(json_key, 0)))
            ps.first_seen = stats_dict.get('first_seen', _now_iso())
            ps.last_seen  = stats_dict.get('last_seen',  _now_iso())
            ps._dirty = True
            migrated += 1

        tracker.flush()
        logger.info(
            f"Migráció: {migrated} játékos → {db_path} "
            f"({tracker._db.db_size_mb():.2f} MB)"
        )
        return tracker
