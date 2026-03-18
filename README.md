# Poker AI v4  –  PPO + Self-Play

Moduláris, PPO-alapú No-Limit Hold'em AI realtime asszisztens funkcióval.

## Telepítés

```bash
pip install -r requirements.txt
```

## Tréning

```bash
python train.py
```

## RTAManager (online póker)

```python
from inference.rta_manager import RTAManager

with RTAManager(
    model_paths = {6: '6max_ppo_v4.pth'},
    db_path     = 'players.db',
) as manager:
    manager.manage_table_change(6, seat_map, my_seat=0)
    manager.new_hand(my_stack=200.0, bb=2.0, sb=1.0)
    result = manager.get_recommendation(obs, legal_actions,
                                         hole_cards=['As','Kh'])
```

## obs_builder – rlcard obs rekonstrukció

```python
from inference.obs_builder import ObsBuilder
builder = ObsBuilder(num_players=6)
obs = builder.build(
    hole_cards  = ['As', 'Kh'],
    board_cards = ['Td', '7c', '2s'],
    my_chips    = 200.0,
    all_chips   = [200.0, 150.0, 300.0, 100.0, 250.0, 180.0],
)
# obs.shape == (54,)
```
