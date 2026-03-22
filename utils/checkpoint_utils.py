"""
utils/checkpoint_utils.py  –  Biztonságos checkpoint betöltő (v4.2.2)

Változások v4.2.2:
    [SECURITY-KRITIKUS-2] Arbitrary Code Execution védelem:
        - Az unsafe fallback (weights_only=False) mostantól EXPLICIT
          allow_unsafe=True paramétert igényel.
        - Alapértelmezett: allow_unsafe=False → UnsafeCheckpointError
          ha a checkpoint nem tölthető weights_only=True móddal.
        - UnsafeCheckpointError: egyedi kivétel a biztonsági problémák
          egyértelmű jelzéséhez, remediációs utasításokkal.
        - migrate_checkpoint_to_safe(): egyszeri konverziós utility
          legacy (v4.1 előtti) checkpointok biztonságossá tételéhez.

Miért veszélyes a weights_only=False?

    A torch.load(..., weights_only=False) teljes pickle deserializációt
    hajt végre.  Egy rosszindulatú .pth fájlban elhelyezett pickle
    payload tetszőleges Python kódot futtathat betöltéskor, pl.:

        import os; os.system('rm -rf /')

    Ez különösen kritikus, ha a fájl forrása nem megbízható
    (hálózatról letöltött, felhasználó által feltöltött checkpoint).

Mi az, amit weights_only=True engedélyez?

    PyTorch 2.0+ esetén:
        ✓ torch.Tensor objektumok
        ✓ Alap Python típusok: int, float, bool, str
        ✓ dict, list, tuple (rekurzívan a fentiekből)
        ✗ Tetszőleges Python osztályok és példányok
        ✗ torch.nn.Module objektumok (csak state_dict-en keresztül)

    A v4.2+ _save_checkpoint() kimenete kompatibilis weights_only=True-val:
        - state_dict: OrderedDict[str, Tensor]
        - trainer: {optimizer_state, scheduler_state, scaler_state, int}
        - reward_norm: {mean: float, M2: float, count: float}
        - Metaadatok: int, float, str

Remediáció legacy checkpointokhoz:

    Ha egy régi checkpoint nem töltható weights_only=True-val:

        from utils.checkpoint_utils import migrate_checkpoint_to_safe
        migrate_checkpoint_to_safe('old_model.pth', 'old_model_safe.pth')

    Ez egyszeri konverzió: unsafe módon betölti, majd safe formátumban
    elmenti.  Ezután az eredeti fájl törölhető.

Használat:

    # Normál betöltés (saját, megbízható checkpoint):
    ck = safe_load_checkpoint('model.pth', map_location='cpu')

    # Legacy checkpoint (egyszeri migráció után kerülendő):
    ck = safe_load_checkpoint(
        'legacy_model.pth',
        map_location='cpu',
        allow_unsafe=True,  # EXPLICIT opt-in, indoklással
    )
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any

import torch

logger = logging.getLogger("PokerAI")


# ─────────────────────────────────────────────────────────────────────────────
# Egyedi kivétel
# ─────────────────────────────────────────────────────────────────────────────

class UnsafeCheckpointError(RuntimeError):
    """
    Kivétel, ha egy checkpoint nem tölthető ``weights_only=True`` móddal
    és az ``allow_unsafe`` flag nem engedélyezi az unsafe fallback-et.

    Az üzenet tartalmaz:
        - a betöltési hiba szövegét (az eredeti PyTorch kivétel)
        - a migrációs utasítást (``migrate_checkpoint_to_safe()``)
        - explicit figyelmeztetést az unsafe betöltés kockázatáról

    Attributes:
        path:           A checkpoint fájl elérési útja.
        original_error: Az eredeti PyTorch betöltési kivétel.
    """

    def __init__(self, path: str, original_error: Exception) -> None:
        self.path           = path
        self.original_error = original_error
        super().__init__(
            f"\n"
            f"{'=' * 70}\n"
            f"  BIZTONSÁGI HIBA: Nem biztonságos checkpoint\n"
            f"{'=' * 70}\n"
            f"  Fájl: {path!r}\n"
            f"  Hiba: {original_error}\n"
            f"\n"
            f"  A checkpoint nem tölthető weights_only=True módban.\n"
            f"  Ez azt jelenti, hogy a fájl tetszőleges Python objektumokat\n"
            f"  tartalmazhat, amelyek betöltéskor kódot futtathatnak.\n"
            f"\n"
            f"  MEGOLDÁS (válassz egyet):\n"
            f"\n"
            f"  1. Migráld a checkpointot biztonságos formátumra (AJÁNLOTT):\n"
            f"\n"
            f"       from utils.checkpoint_utils import migrate_checkpoint_to_safe\n"
            f"       migrate_checkpoint_to_safe({path!r}, {path!r})\n"
            f"\n"
            f"  2. Ha biztosan megbízol a fájlban, engedélyezd az unsafe\n"
            f"     betöltést EXPLICIT opt-in-nel:\n"
            f"\n"
            f"       safe_load_checkpoint({path!r!s}, allow_unsafe=True)\n"
            f"\n"
            f"  FIGYELEM: Az allow_unsafe=True CSAK megbízható forrású\n"
            f"  fájloknál alkalmazható!  Hálózatról letöltött vagy\n"
            f"  ismeretlen checkpointok esetén NE használd!\n"
            f"{'=' * 70}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fő betöltő függvény
# ─────────────────────────────────────────────────────────────────────────────

def safe_load_checkpoint(
    path:         str,
    map_location: Any   = "cpu",
    allow_unsafe: bool  = False,
) -> Any:
    """
    Biztonságos torch checkpoint betöltés.

    Először ``weights_only=True``-val próbál betölteni (pickle-mentes,
    csak tensor és alap Python típusok).  Ha ez sikertelen:

    - ``allow_unsafe=False`` (alapértelmezett): ``UnsafeCheckpointError``
      kivételt dob részletes remediációs utasításokkal.
    - ``allow_unsafe=True`` (explicit opt-in): WARNING log után
      ``weights_only=False`` fallback, teljes pickle deserializáció.

    Args:
        path:         A .pth fájl abszolút elérési útja.
        map_location: torch device leírás (pl. ``'cpu'``, ``'cuda:0'``).
                      Alapértelmezett: ``'cpu'``.
        allow_unsafe: Ha ``True``, engedélyezi az unsafe pickle
                      deserializációt ha a safe betöltés sikertelen.
                      **Csak megbízható forrású fájloknál használd!**
                      Alapértelmezett: ``False``.

    Returns:
        A betöltött checkpoint (általában ``dict`` state_dict-tel).

    Raises:
        FileNotFoundError:    Ha a fájl nem létezik.
        UnsafeCheckpointError: Ha ``weights_only=True`` sikertelen és
                               ``allow_unsafe=False``.
        ValueError:           Ha sem safe, sem unsafe móddal nem
                              tölthető be a fájl.

    Példák::

        # Normál használat – saját, megbízható checkpoint:
        ck = safe_load_checkpoint('models/2max/2max_ppo_v4.pth')

        # Legacy checkpoint explicit engedéllyel (migrálás előtt):
        ck = safe_load_checkpoint(
            'old_model.pth',
            allow_unsafe=True,  # INDOK: saját gépről, migrálás után törölni
        )

        # Egy adott GPU-ra betöltés:
        ck = safe_load_checkpoint('model.pth', map_location='cuda:0')
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint fájl nem található: {path!r}"
        )

    # ── 1. Biztonságos betöltési kísérlet ────────────────────────────────
    try:
        ck = torch.load(path, map_location=map_location, weights_only=True)
        logger.debug(f"Checkpoint betöltve (weights_only=True): {path!r}")
        return ck

    except UnsafeCheckpointError:
        # Ne kapja el a saját kivételünket, ha valahogy bekerül
        raise

    except Exception as e_safe:
        # ── 2. Safe betöltés sikertelen ───────────────────────────────────
        if not allow_unsafe:
            # BIZTONSÁGI MEGÁLLÁS: nem esünk vissza unsafe módra
            # implicit módon – explicit opt-in szükséges.
            logger.error(
                f"Checkpoint NEM tölthető weights_only=True módban: "
                f"{path!r}  Hiba: {e_safe}  "
                f"Megoldás: futtasd a migrate_checkpoint_to_safe() "
                f"függvényt, vagy adj meg allow_unsafe=True-t ha "
                f"biztosan megbízol a fájlban."
            )
            raise UnsafeCheckpointError(path, e_safe) from e_safe

        # ── 3. Unsafe fallback – CSAK explicit allow_unsafe=True esetén ──
        logger.warning(
            f"\n"
            f"  ⚠  BIZTONSÁGI FIGYELMEZTETÉS\n"
            f"  weights_only=True sikertelen ({path!r}): {e_safe}\n"
            f"  allow_unsafe=True → weights_only=False (pickle) betöltés.\n"
            f"  Ez CSAK megbízható, saját forrású fájloknál biztonságos!\n"
            f"  Ismeretlen vagy hálózatról letöltött fájlokra NE alkalmazd.\n"
            f"  Ajánlott: futtasd a migrate_checkpoint_to_safe() függvényt."
        )

        try:
            ck = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
            logger.debug(
                f"Checkpoint betöltve (weights_only=False fallback): {path!r}"
            )
            return ck

        except Exception as e_unsafe:
            raise ValueError(
                f"Checkpoint nem tölthető be egyáltalán: {path!r}\n"
                f"  weights_only=True hiba:  {e_safe}\n"
                f"  weights_only=False hiba: {e_unsafe}"
            ) from e_unsafe


# ─────────────────────────────────────────────────────────────────────────────
# Migrációs utility
# ─────────────────────────────────────────────────────────────────────────────

def migrate_checkpoint_to_safe(
    src_path: str,
    dst_path: str,
    map_location: Any = "cpu",
) -> None:
    """
    Konvertál egy legacy (nem weights_only=True kompatibilis) checkpointot
    biztonságos formátumra.

    A folyamat:
        1. Betölti az eredeti fájlt ``weights_only=False`` módban
           (egyszeri unsafe betöltés).
        2. Csak a szükséges mezőket tartja meg (szűri a Python objektumokat).
        3. Elmenti ``torch.save()``-val, amely ``weights_only=True``-kompatibilis
           formátumot produkál.
        4. Atomikus írás (temp fájl + rename), hogy ne keletkezzen
           részleges fájl hiba esetén.

    Migrált mezők (v4.2 checkpoint formátum):
        - ``state_dict``:       model súlyok
        - ``trainer``:          optimizer/scheduler/scaler állapot
        - ``reward_norm``:      RunningMeanStd állapot
        - ``episodes_trained``: int
        - ``time_spent``:       float
        - ``algorithm``:        str
        - ``state_size``:       int
        - ``action_size``:      int
        - ``num_players``:      int (ha jelen van)
        - ``rlcard_obs_size``:  int (ha jelen van)

    Args:
        src_path:     Forrás .pth fájl (legacy checkpoint).
        dst_path:     Cél .pth fájl (safe formátum).
                      Lehet azonos a src_path-szal (helyben konvertál,
                      atomikus felülírással).
        map_location: torch device (alapértelmezett: ``'cpu'``).

    Raises:
        FileNotFoundError: Ha a forrás fájl nem létezik.
        ValueError:        Ha a forrás fájl sem safe, sem unsafe módon
                           nem tölthető be.
        RuntimeError:      Ha a migrált checkpoint nem tartalmaz
                           ``state_dict`` kulcsot.

    Példa::

        # Helyben konvertálás (felülírja az eredetit):
        migrate_checkpoint_to_safe('old_model.pth', 'old_model.pth')

        # Új fájlba mentés (az eredeti megmarad):
        migrate_checkpoint_to_safe('old_model.pth', 'old_model_safe.pth')

    Megjegyzés:
        Ez a függvény szándékosan hívja az unsafe betöltést – ez az
        EGYETLEN helyen elfogadható az egész kódbázisban.
        Egyszeri, ellenőrzött konverzióra tervezve.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Forrás checkpoint nem található: {src_path!r}"
        )

    logger.info(
        f"Checkpoint migrálás: {src_path!r} → {dst_path!r}"
    )

    # 1. Betöltés unsafe módban (ez az egyetlen helyen elfogadható)
    #    allow_unsafe=True itt szándékos és dokumentált.
    ck = safe_load_checkpoint(
        src_path,
        map_location=map_location,
        allow_unsafe=True,  # INDOK: egyszeri konverzió, kontrollált env
    )

    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise RuntimeError(
            f"A forrás checkpoint nem tartalmaz 'state_dict' kulcsot: "
            f"{src_path!r}.  Típus: {type(ck).__name__}"
        )

    # 2. Csak biztonságos (serializable) mezők kinyerése.
    #    A v4.2 formátum ÖSSZES mezője alap Python típus vagy tensor –
    #    ezek weights_only=True-val visszatölthetők.
    safe_ck: dict = {}

    # Kötelező mezők
    safe_ck["state_dict"] = ck["state_dict"]

    # Opcionális, de fontos mezők – típus-ellenőrzéssel
    _copy_if_present(ck, safe_ck, "trainer",          expected_type=dict)
    _copy_if_present(ck, safe_ck, "reward_norm",      expected_type=dict)
    _copy_if_present(ck, safe_ck, "episodes_trained", expected_type=(int, float))
    _copy_if_present(ck, safe_ck, "time_spent",       expected_type=(int, float))
    _copy_if_present(ck, safe_ck, "algorithm",        expected_type=str)
    _copy_if_present(ck, safe_ck, "state_size",       expected_type=int)
    _copy_if_present(ck, safe_ck, "action_size",      expected_type=int)
    _copy_if_present(ck, safe_ck, "num_players",      expected_type=(int, type(None)))
    _copy_if_present(ck, safe_ck, "rlcard_obs_size",  expected_type=int)

    # Elveszett mezők logolása (információ, nem hiba)
    skipped = [k for k in ck if k not in safe_ck]
    if skipped:
        logger.warning(
            f"Migráció: az alábbi mezők kihagyva (nem serializable "
            f"alaptípus, vagy ismeretlen): {skipped}"
        )

    # 3. Atomikus mentés: temp fájl + rename
    dst_dir  = os.path.dirname(os.path.abspath(dst_path)) or "."
    os.makedirs(dst_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=dst_dir, suffix=".pth.tmp"
    )
    os.close(tmp_fd)

    try:
        torch.save(safe_ck, tmp_path)
        shutil.move(tmp_path, dst_path)
    except Exception as exc:
        # Takarítás: temp fájl törlése hiba esetén
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Mentési hiba migrálás közben ({dst_path!r}): {exc}"
        ) from exc

    # 4. Verifikáció: az új fájl betölthető-e weights_only=True-val?
    try:
        safe_load_checkpoint(dst_path, map_location=map_location)
        logger.info(
            f"Migráció sikeres: {dst_path!r}  "
            f"(weights_only=True verifikáció OK)"
        )
    except UnsafeCheckpointError as verify_err:
        logger.error(
            f"Verifikáció SIKERTELEN: a migrált fájl még mindig nem "
            f"kompatibilis weights_only=True-val!  "
            f"Hiba: {verify_err.original_error}  "
            f"A forrás fájl érintetlen maradt: {src_path!r}"
        )
        raise RuntimeError(
            f"Migráció sikertelen: az újramentett checkpoint még mindig "
            f"nem tölthető weights_only=True módban.  "
            f"Forrás: {src_path!r}, Cél: {dst_path!r}"
        ) from verify_err


# ─────────────────────────────────────────────────────────────────────────────
# Belső segédfüggvény
# ─────────────────────────────────────────────────────────────────────────────

def _copy_if_present(
    src:           dict,
    dst:           dict,
    key:           str,
    expected_type: Any = None,
) -> None:
    """
    Másolja a ``key`` értékét ``src``-ból ``dst``-be, ha jelen van.

    Ha ``expected_type`` meg van adva, csak akkor másolja, ha az érték
    az elvárt típusú.  Típus-eltérés esetén WARNING logot ír, de nem
    dob kivételt (graceful degradation migrációban).

    Args:
        src:           Forrás dict.
        dst:           Cél dict.
        key:           A másolandó kulcs neve.
        expected_type: Elvárt Python típus (vagy tuple of types).
                       ``None`` → nincs típusellenőrzés.
    """
    if key not in src:
        return
    value = src[key]
    if expected_type is not None and not isinstance(value, expected_type):
        logger.warning(
            f"Migráció: '{key}' típusa {type(value).__name__!r}, "
            f"elvárt: {expected_type}.  Kihagyva."
        )
        return
    dst[key] = value
