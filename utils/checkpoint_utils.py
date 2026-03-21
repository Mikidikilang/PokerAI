"""
utils/checkpoint_utils.py  –  Biztonságos checkpoint betöltő

weights_only=True: PyTorch 1.13+ ajánlott mód, nem futtat arbitrary kódot.
weights_only=False: Régi checkpointokhoz szükséges fallback (pickle alapú).

Használat:
    from utils.checkpoint_utils import safe_load_checkpoint

    ck = safe_load_checkpoint('model.pth', map_location='cpu')
"""

import logging
import torch

logger = logging.getLogger("PokerAI")


def safe_load_checkpoint(path: str, map_location='cpu') -> dict:
    """
    Biztonságos torch checkpoint betöltés.

    Először weights_only=True-val próbálja (biztonságos, pickle nélkül).
    Ha ez sikertelen (régi formátum), fallback weights_only=False-ra,
    de ezt WARNING szinten logolva – jelzi a potenciális biztonsági kockázatot.

    Paraméterek:
        path:         .pth fájl elérési útja
        map_location: torch.device vagy string (default: 'cpu')

    Visszatér: betöltött checkpoint (dict vagy state_dict)

    Kivételek:
        FileNotFoundError – ha a fájl nem létezik
        ValueError        – ha egyik módszerrel sem tölthető be
    """
    try:
        ck = torch.load(path, map_location=map_location, weights_only=True)
        logger.debug(f"Checkpoint betöltve (weights_only=True): {path}")
        return ck
    except Exception as e_safe:
        logger.warning(
            f"weights_only=True sikertelen ({path}): {e_safe}\n"
            f"  Fallback weights_only=False – régi checkpoint formátum. "
            f"Csak megbízható forrásból betöltött fájloknál elfogadható!"
        )
        try:
            ck = torch.load(path, map_location=map_location, weights_only=False)
            logger.debug(f"Checkpoint betöltve (weights_only=False fallback): {path}")
            return ck
        except Exception as e_unsafe:
            raise ValueError(
                f"Checkpoint nem tölthető be: {path}\n"
                f"  weights_only=True hiba:  {e_safe}\n"
                f"  weights_only=False hiba: {e_unsafe}"
            ) from e_unsafe
