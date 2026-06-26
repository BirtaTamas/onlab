# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `6`

## Largest probability jumps

- tick `39343`, seconds `75.50`, LSTM `0.8826`, delta `+0.2730`
- tick `35215`, seconds `11.00`, LSTM `0.1355`, delta `-0.2369`
- tick `39279`, seconds `74.50`, LSTM `0.6177`, delta `+0.2278`
- tick `39503`, seconds `78.00`, LSTM `0.7712`, delta `-0.1161`
- tick `40047`, seconds `86.50`, LSTM `0.9481`, delta `+0.1051`
- tick `39663`, seconds `80.50`, LSTM `0.7177`, delta `-0.0608`
- tick `35119`, seconds `9.50`, LSTM `0.3875`, delta `+0.0484`
- tick `38831`, seconds `67.50`, LSTM `0.3297`, delta `+0.0455`
- tick `37071`, seconds `40.00`, LSTM `0.2845`, delta `-0.0450`
- tick `38927`, seconds `69.00`, LSTM `0.3725`, delta `+0.0431`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003583`, |coef| `0.003583`
- `lag_00__CT_kills_last_3s`: coefficient `0.003333`, |coef| `0.003333`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002527`, |coef| `0.002527`
- `lag_00__CT_damage_last_5s`: coefficient `0.002495`, |coef| `0.002495`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002429`, |coef| `0.002429`
- `lag_00__damage_diff_last_5s`: coefficient `0.002331`, |coef| `0.002331`
- `lag_02__CT_kills_last_3s`: coefficient `0.002127`, |coef| `0.002127`
- `lag_00__CT_place_BACKOFB`: coefficient `0.001784`, |coef| `0.001784`
- `lag_02__CT_damage_last_5s`: coefficient `0.001718`, |coef| `0.001718`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001704`, |coef| `0.001704`
- `lag_11__T_place_TSTAIRS`: coefficient `-0.001640`, |coef| `0.001640`
- `lag_00__T_place_LONGDOG`: coefficient `-0.001586`, |coef| `0.001586`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001586`, |coef| `0.001586`
- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.001566`, |coef| `0.001566`
- `lag_02__kill_diff_last_3s`: coefficient `0.001553`, |coef| `0.001553`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.002429` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001536` (raises CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.001141` (raises CT win probability)
- `lag_04__T2__flash_duration`: coefficient `-0.001110` (lowers CT win probability)
- `lag_09__T4__smoke`: coefficient `-0.000983` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000976` (lowers CT win probability)
- `lag_07__T4__smoke`: coefficient `-0.000950` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `0.000950` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.000917` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000895` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003583` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003333` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002527` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002495` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002331` (raises CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `0.002127` (raises CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.001784` (raises CT win probability)
- `lag_02__CT_damage_last_5s`: coefficient `0.001718` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001704` (raises CT win probability)
- `lag_11__T_place_TSTAIRS`: coefficient `-0.001640` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `39343`, seconds `75.50`, LSTM delta `+0.2730`

Top all feature movements:
- `lag_02__CT_kills_last_3s`: contribution `+0.012284`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010532`
- `lag_00__CT_kills_last_3s`: contribution `+0.009622`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008816`
- `lag_00__kill_diff_last_3s`: contribution `+0.008625`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35215`, seconds `11.00`, LSTM delta `-0.2369`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.034682`
- `lag_11__T_place_TSTAIRS`: contribution `-0.018592`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.018208`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.013869`
- `lag_14__CT_place_ENTRANCE`: contribution `-0.010464`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.034682`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.013869`
- `lag_10__T_flashes_last_5s`: contribution `-0.010336`
- `lag_01__T3__flash_duration`: contribution `-0.003948`
- `lag_01__T1__flash_duration`: contribution `-0.003495`

### tick `39279`, seconds `74.50`, LSTM delta `+0.2278`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.019244`
- `lag_00__kill_diff_last_3s`: contribution `+0.017250`
- `lag_00__CT_shots_fired_sum`: contribution `+0.014043`
- `lag_09__CT1__shots_fired`: contribution `+0.008698`
- `lag_00__CT_damage_last_5s`: contribution `+0.008213`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39503`, seconds `78.00`, LSTM delta `-0.1161`

Top all feature movements:
- `lag_04__CT4__shots_fired`: contribution `-0.009489`
- `lag_00__kill_diff_last_3s`: contribution `-0.008625`
- `lag_02__T_place_LONGDOG`: contribution `-0.007130`
- `lag_01__CT_kills_last_3s`: contribution `-0.006838`
- `lag_04__CT_shots_fired_sum`: contribution `-0.006088`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40047`, seconds `86.50`, LSTM delta `+0.1051`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009622`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008777`
- `lag_00__kill_diff_last_3s`: contribution `+0.008625`
- `lag_00__T_place_LONGDOG`: contribution `+0.007382`
- `lag_02__CT_place_BACKOFB`: contribution `+0.006481`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.005607`
