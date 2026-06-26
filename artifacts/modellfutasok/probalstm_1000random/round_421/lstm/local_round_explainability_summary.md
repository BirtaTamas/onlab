# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `19`

## Largest probability jumps

- tick `188637`, seconds `25.50`, LSTM `0.7625`, delta `+0.2410`
- tick `188445`, seconds `22.50`, LSTM `0.4691`, delta `-0.1972`
- tick `188605`, seconds `25.00`, LSTM `0.5216`, delta `+0.1925`
- tick `188285`, seconds `20.00`, LSTM `0.6945`, delta `-0.1428`
- tick `188029`, seconds `16.00`, LSTM `0.6668`, delta `+0.1368`
- tick `188253`, seconds `19.50`, LSTM `0.8373`, delta `+0.1173`
- tick `188317`, seconds `20.50`, LSTM `0.7820`, delta `+0.0875`
- tick `188349`, seconds `21.00`, LSTM `0.7136`, delta `-0.0684`
- tick `188221`, seconds `19.00`, LSTM `0.7200`, delta `+0.0641`
- tick `188541`, seconds `24.00`, LSTM `0.3292`, delta `-0.0620`

## Top 15 local ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.003002`, |coef| `0.003002`
- `lag_10__CT4__flash_duration`: coefficient `0.002176`, |coef| `0.002176`
- `lag_00__CT_kills_last_3s`: coefficient `0.001997`, |coef| `0.001997`
- `lag_10__CT_place_BACKALLEY`: coefficient `-0.001957`, |coef| `0.001957`
- `lag_04__T_kills_last_3s`: coefficient `-0.001948`, |coef| `0.001948`
- `lag_00__damage_diff_last_5s`: coefficient `0.001825`, |coef| `0.001825`
- `lag_00__kill_diff_last_3s`: coefficient `0.001816`, |coef| `0.001816`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_11__CT3__flash_duration`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_06__CT_place_BRIDGE`: coefficient `0.001667`, |coef| `0.001667`
- `lag_07__T5__is_scoped`: coefficient `-0.001634`, |coef| `0.001634`
- `lag_10__T5__is_scoped`: coefficient `0.001629`, |coef| `0.001629`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001585`, |coef| `0.001585`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001584`, |coef| `0.001584`
- `lag_03__CT_place_BALCONY`: coefficient `0.001562`, |coef| `0.001562`

## Top 10 utility ridge features

- `lag_10__CT4__flash_duration`: coefficient `0.002176` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.001671` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `-0.001558` (lowers CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `-0.001271` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.001115` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.001095` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.000971` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000967` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `0.000931` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.000927` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.003002` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001997` (raises CT win probability)
- `lag_10__CT_place_BACKALLEY`: coefficient `-0.001957` (lowers CT win probability)
- `lag_04__T_kills_last_3s`: coefficient `-0.001948` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001825` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001816` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.001812` (lowers CT win probability)
- `lag_06__CT_place_BRIDGE`: coefficient `0.001667` (raises CT win probability)
- `lag_07__T5__is_scoped`: coefficient `-0.001634` (lowers CT win probability)
- `lag_10__T5__is_scoped`: coefficient `0.001629` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `188637`, seconds `25.50`, LSTM delta `+0.2410`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `+0.011632`
- `lag_00__CT4__duck_amount`: contribution `-0.011023`
- `lag_03__CT_place_BALCONY`: contribution `+0.010023`
- `lag_11__CT3__flash_duration`: contribution `+0.009263`
- `lag_10__T5__is_scoped`: contribution `+0.007772`

Top utility-only movements:
- `lag_11__CT3__flash_duration`: contribution `+0.009263`

### tick `188445`, seconds `22.50`, LSTM delta `-0.1972`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `-0.016580`
- `lag_00__kill_diff_last_3s`: contribution `-0.008743`
- `lag_10__T5__is_scoped`: contribution `-0.007772`
- `lag_02__T5__is_scoped`: contribution `-0.006341`
- `lag_04__T_kills_last_3s`: contribution `-0.006170`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.016580`
- `lag_14__CT3__flash_duration`: contribution `-0.005367`
- `lag_05__CT3__flash_duration`: contribution `-0.005142`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.005044`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.003444`

### tick `188605`, seconds `25.00`, LSTM delta `+0.1925`

Top all feature movements:
- `lag_00__CT4__duck_amount`: contribution `+0.011023`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.009429`
- `lag_15__CT4__flash_duration`: contribution `+0.008347`
- `lag_07__T5__is_scoped`: contribution `+0.007794`
- `lag_02__CT_place_BALCONY`: contribution `+0.006581`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.009429`
- `lag_15__CT4__flash_duration`: contribution `+0.008347`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.006309`

### tick `188285`, seconds `20.00`, LSTM delta `-0.1428`

Top all feature movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009429`
- `lag_07__T5__is_scoped`: contribution `-0.007794`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.006309`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005506`
- `lag_05__CT4__flash_duration`: contribution `-0.004410`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.009429`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.006309`
- `lag_05__CT4__flash_duration`: contribution `-0.004410`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.003519`
- `lag_09__CT3__flash_duration`: contribution `-0.002888`

### tick `188029`, seconds `16.00`, LSTM delta `+0.1368`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `+0.016580`
- `lag_07__T5__is_scoped`: contribution `+0.007794`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.006749`
- `lag_00__CT_kills_last_3s`: contribution `+0.005765`
- `lag_11__T_flashed_players`: contribution `+0.005465`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.016580`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.006749`
- `lag_11__T1__flash_duration`: contribution `+0.005294`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.004571`
- `lag_01__T1__flash_duration`: contribution `+0.004436`
