# Poker AI v4  –  PPO + Self-Play

Moduláris, PPO-alapú No-Limit Hold'em AI realtime asszisztens funkcióval.

---

## Telepítés

```bash
pip install -r requirements.txt
```

---

## Tréning

```bash
python train.py
```

Menüből választható játékosszám (2–9) és checkpoint fájlnév.

---

## Realtime asszisztens használata

```python
from inference import RealtimePokerAssistant

assistant = RealtimePokerAssistant(
    model_path  = '6max_ppo_v4.pth',
    num_players = 6,
    device      = 'cpu',
)

# Kéz kezdete
assistant.new_hand(
    my_stack     = 150.0,
    all_stacks   = [150.0] * 6,
    bb           = 2.0,
    sb           = 1.0,
    my_player_id = 2,
    button_pos   = 1,
)

# Döntés kérése
result = assistant.get_recommendation(
    obs_vector    = obs,             # rlcard obs array
    legal_actions = [0, 1, 2, 3, 4, 5, 6],
    hole_cards    = ['As', 'Kh'],    # opcionális
    board_cards   = [],
    call_amount   = 4.0,
)

print(result['action_name'])    # pl. "Raise 50%"
print(result['confidence'])     # pl. 0.73
print(result['explanation'])    # pl. "Preflop | Raise 50% (73%) | Equity: 67%"

# Ellenfél lépés rögzítése
assistant.record_opponent_action(player_id=3, action=4, bet_amount=10.0)

# Street váltás
assistant.new_street(street=1)  # flop
```

---

## Könyvtár struktúra

```
poker_ai_v4/
├── train.py                   # belépési pont
├── requirements.txt
├── NOTES_MASTER.txt           # teljes fejlesztési dokumentáció
│
├── core/
│   ├── model.py               # AdvancedPokerAI (PPO Actor-Critic)
│   ├── action_mapper.py       # 7 absztrakt akció leképező
│   ├── features.py            # teljes feature engineering
│   ├── opponent_tracker.py    # HUD statisztikák (VPIP/PFR/AF/...)
│   └── equity.py              # Monte Carlo kéz equity becslő
│
├── training/
│   ├── buffer.py              # PPO buffer + GAE
│   ├── trainer.py             # PPO trainer
│   ├── normalizer.py          # futó reward normalizáló
│   ├── opponent_pool.py       # self-play ellenfél pool
│   ├── collector.py           # BatchedSyncCollector (256 párhuzamos env)
│   └── runner.py              # tréning session + menü
│
├── inference/
│   └── realtime_assistant.py  # RealtimePokerAssistant
│
└── utils/
    └── logging_setup.py       # logger konfiguráció
```

---

## v4 újítások (v3.4-hez képest)

| Feature | v3.4 | v4 |
|---|---|---|
| BB/SB paraméter | hardkód 100 | explicit, randomizált |
| Reward skála | abszolút chip | BB-ben mérve |
| Stack features | SPR + 3 dim | SPR + M-ratio + depth one-hot (8 dim) |
| Street context | nincs | one-hot 4 dim |
| Pot odds | nincs | pot_odds + call_bb + facing_bet (4 dim) |
| Board texture | nincs | 6 dim |
| Opponent stats | nyers frekvencia | VPIP/PFR/AF/3bet%/fold_to_3bet/cbet%/fcb% |
| Bet size history | nincs | bet_size_norm per akció |
| Hand equity | nincs | MC becslő (200 szimuláció, LRU cache) |
| Dupla encode bug | jelen | javítva |
| State before bug | jelen | javítva |
| Realtime API | nincs | RealtimePokerAssistant |

---

## Checkpoint kompatibilitás

A v4 **inkompatibilis** v3.x checkpointokkal az `input_proj` réteg
méretváltozása miatt. A checkpoint betöltő részleges betöltést végez
(size-mismatch rétegek kihagyva), warninggal jelezve.
