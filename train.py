"""
train.py  –  Poker AI v4 tréning belépési pont

Használat:
    python train.py
    py -3.12 train.py
"""

# sys.path fix MINDENMAS IMPORT ELOTT – Windows-on szükséges
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from utils.logging_setup import setup_logging
from training.runner import run_training_session, menu_system


def main():
    # Alap logger (session-specifikus logot a runner.py indítja)
    setup_logging("training.log")
    logger = logging.getLogger("PokerAI")

    while True:
        num_players, filename = menu_system()

        if num_players == 'ALL':
            try:
                eps = int(input("\nHány epizód ASZTALONKÉNT? (pl. 500000): "))
            except ValueError:
                eps = 50_000
            for p in range(2, 10):
                run_training_session(p, f"{p}max_ppo_v4.pth", eps)
        else:
            try:
                eps = int(input("\nHány epizódot futtassunk? "))
            except ValueError:
                eps = 100_000
            run_training_session(num_players, filename, eps)

        input("\n>>> ENTER a visszatéréshez <<<")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("PokerAI").info("Megszakítva (KeyboardInterrupt).")
    except Exception:
        logging.getLogger("PokerAI").critical(
            "Nem kezelt kivétel:", exc_info=True
        )
        input("\n>>> Hiba. ENTER a bezáráshoz... <<<")
