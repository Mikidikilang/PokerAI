"""
utils/checkpoint_utils.py  –  Biztonságos checkpoint betöltő (v4.2.2-BUGFIX)

Változások v4.2.2-BUGFIX:
    [BUG-CRITICAL] SyntaxError javítva: {path!r!s} → {path!r}
        A Python 3.13 szigorúbb f-string parsert használ és nem fogad
        el kettős konverziót (!r!s). Ez SyntaxError-t okozott már az
        import során → az egész projekt indíthatatlan volt.
        Érintett sor: UnsafeCheckpointError.__init__() hibaüzenet.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any

import torch

logger = logging.getLogger("PokerAI")


class UnsafeCheckpointError(RuntimeError):
    """
    Kivétel, ha egy checkpoint nem tölthető weights_only=True móddal
    és az allow_unsafe flag nem engedélyezi az unsafe fallback-et.
    """

    def __init__(self, path: str, original_error: Exception) -> None:
        self.path           = path
        self.original_error = original_error
        path_repr = repr(path)  # [BUG-FIX] !r!s helyett külön változó
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
            f"       safe_load_checkpoint({path_repr}, allow_unsafe=True)\n"
            f"\n"
            f"  FIGYELEM: Az allow_unsafe=True CSAK megbízható forrású\n"
            f"  fájloknál alkalmazható!  Hálózatról letöltött vagy\n"
            f"  ismeretlen checkpointok esetén NE használd!\n"
            f"{'=' * 70}"
        )


def safe_load_checkpoint(
    path:         str,
    map_location: Any  = "cpu",
    allow_unsafe: bool = False,
) -> Any:
    """
    Biztonságos torch checkpoint betöltés.

    Először weights_only=True-val próbál betölteni. Ha ez sikertelen:
    - allow_unsafe=False (alapértelmezett): UnsafeCheckpointError
    - allow_unsafe=True: weights_only=False fallback (pickle)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint fájl nem található: {path!r}")

    # 1. Biztonságos betöltési kísérlet
    try:
        ck = torch.load(path, map_location=map_location, weights_only=True)
        logger.debug(f"Checkpoint betöltve (weights_only=True): {path!r}")
        return ck

    except UnsafeCheckpointError:
        raise

    except Exception as e_safe:
        if not allow_unsafe:
            logger.error(
                f"Checkpoint NEM tölthető weights_only=True módban: "
                f"{path!r}  Hiba: {e_safe}  "
                f"Megoldás: futtasd a migrate_checkpoint_to_safe() "
                f"függvényt, vagy adj meg allow_unsafe=True-t ha "
                f"biztosan megbízol a fájlban."
            )
            raise UnsafeCheckpointError(path, e_safe) from e_safe

        # 2. Unsafe fallback – CSAK explicit allow_unsafe=True esetén
        logger.warning(
            f"weights_only=True sikertelen ({path!r}): {e_safe} – "
            f"allow_unsafe=True → weights_only=False (pickle) betöltés. "
            f"Ajánlott: futtasd a migrate_checkpoint_to_safe() függvényt."
        )

        try:
            ck = torch.load(path, map_location=map_location, weights_only=False)
            logger.debug(f"Checkpoint betöltve (weights_only=False fallback): {path!r}")
            return ck
        except Exception as e_unsafe:
            raise ValueError(
                f"Checkpoint nem tölthető be egyáltalán: {path!r}\n"
                f"  weights_only=True hiba:  {e_safe}\n"
                f"  weights_only=False hiba: {e_unsafe}"
            ) from e_unsafe


def migrate_checkpoint_to_safe(
    src_path: str,
    dst_path: str,
    map_location: Any = "cpu",
) -> None:
    """
    Konvertál egy legacy checkpointot weights_only=True kompatibilis formátumra.

    Helyben is használható: migrate_checkpoint_to_safe('model.pth', 'model.pth')
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Forrás checkpoint nem található: {src_path!r}")

    logger.info(f"Checkpoint migrálás: {src_path!r} → {dst_path!r}")

    ck = safe_load_checkpoint(src_path, map_location=map_location, allow_unsafe=True)

    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise RuntimeError(
            f"A forrás checkpoint nem tartalmaz 'state_dict' kulcsot: "
            f"{src_path!r}.  Típus: {type(ck).__name__}"
        )

    safe_ck: dict = {}
    safe_ck["state_dict"] = ck["state_dict"]

    _copy_if_present(ck, safe_ck, "trainer",          expected_type=dict)
    _copy_if_present(ck, safe_ck, "reward_norm",      expected_type=dict)
    _copy_if_present(ck, safe_ck, "episodes_trained", expected_type=(int, float))
    _copy_if_present(ck, safe_ck, "time_spent",       expected_type=(int, float))
    _copy_if_present(ck, safe_ck, "algorithm",        expected_type=str)
    _copy_if_present(ck, safe_ck, "state_size",       expected_type=int)
    _copy_if_present(ck, safe_ck, "action_size",      expected_type=int)
    _copy_if_present(ck, safe_ck, "num_players",      expected_type=(int, type(None)))
    _copy_if_present(ck, safe_ck, "rlcard_obs_size",  expected_type=int)

    skipped = [k for k in ck if k not in safe_ck]
    if skipped:
        logger.warning(f"Migráció: mezők kihagyva: {skipped}")

    dst_dir = os.path.dirname(os.path.abspath(dst_path)) or "."
    os.makedirs(dst_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=dst_dir, suffix=".pth.tmp")
    os.close(tmp_fd)

    try:
        torch.save(safe_ck, tmp_path)
        shutil.move(tmp_path, dst_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise RuntimeError(f"Mentési hiba migrálás közben ({dst_path!r}): {exc}") from exc

    # Verifikáció
    try:
        safe_load_checkpoint(dst_path, map_location=map_location)
        logger.info(f"Migráció sikeres: {dst_path!r} (weights_only=True OK)")
    except UnsafeCheckpointError as verify_err:
        raise RuntimeError(
            f"Migráció sikertelen: az újramentett checkpoint még mindig "
            f"nem tölthető weights_only=True módban. Forrás: {src_path!r}"
        ) from verify_err


def _copy_if_present(src, dst, key, expected_type=None):
    if key not in src:
        return
    value = src[key]
    if expected_type is not None and not isinstance(value, expected_type):
        logger.warning(
            f"Migráció: '{key}' típusa {type(value).__name__!r}, "
            f"elvárt: {expected_type}. Kihagyva."
        )
        return
    dst[key] = value
