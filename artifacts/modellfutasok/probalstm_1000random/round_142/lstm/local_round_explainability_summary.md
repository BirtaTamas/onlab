# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `3`

## Largest probability jumps

- tick `10962`, seconds `0.50`, LSTM `0.0173`, delta `-0.0272`
- tick `11442`, seconds `8.00`, LSTM `0.0404`, delta `+0.0090`
- tick `11506`, seconds `9.00`, LSTM `0.0469`, delta `+0.0071`
- tick `11666`, seconds `11.50`, LSTM `0.0340`, delta `-0.0063`
- tick `13170`, seconds `35.00`, LSTM `0.0068`, delta `-0.0050`
- tick `12946`, seconds `31.50`, LSTM `0.0222`, delta `+0.0043`
- tick `10994`, seconds `1.00`, LSTM `0.0132`, delta `-0.0041`
- tick `12818`, seconds `29.50`, LSTM `0.0198`, delta `-0.0036`
- tick `11794`, seconds `13.50`, LSTM `0.0258`, delta `-0.0035`
- tick `11602`, seconds `10.50`, LSTM `0.0424`, delta `-0.0035`

## Top 15 local ridge features

- `lag_01__CT_macro_A`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__CT_place_BOMBSITEA`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000209`, |coef| `0.000209`
- `lag_00__T_velocity_mean`: coefficient `-0.000180`, |coef| `0.000180`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000168`, |coef| `0.000168`
- `lag_01__centroid_distance_xy`: coefficient `-0.000157`, |coef| `0.000157`
- `lag_00__CT_velocity_mean`: coefficient `-0.000136`, |coef| `0.000136`
- `lag_01__utility_inv_diff`: coefficient `0.000128`, |coef| `0.000128`
- `lag_01__T_mean_Y`: coefficient `0.000127`, |coef| `0.000127`
- `lag_01__T1__has_bomb`: coefficient `-0.000126`, |coef| `0.000126`
- `lag_01__T4__Y`: coefficient `0.000124`, |coef| `0.000124`
- `lag_01__armor_diff`: coefficient `0.000123`, |coef| `0.000123`
- `lag_00__T1__shots_fired`: coefficient `-0.000119`, |coef| `0.000119`
- `lag_01__smoke_inv_diff`: coefficient `0.000118`, |coef| `0.000118`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000128` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000118` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000105` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000104` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000101` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000088` (raises CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000085` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000085` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000073` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000072` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_macro_A`: coefficient `-0.000251` (lowers CT win probability)
- `lag_01__CT_place_BOMBSITEA`: coefficient `-0.000251` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000209` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000180` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000169` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000168` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000157` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000136` (lowers CT win probability)
- `lag_01__T_mean_Y`: coefficient `0.000127` (raises CT win probability)
- `lag_01__T1__has_bomb`: coefficient `-0.000126` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `10962`, seconds `0.50`, LSTM delta `-0.0272`

Top all feature movements:
- `lag_01__CT_place_BOMBSITEA`: contribution `-0.001453`
- `lag_01__CT_macro_A`: contribution `-0.001453`
- `lag_01__T_place_TSPAWN`: contribution `-0.000926`
- `lag_00__T_velocity_mean`: contribution `-0.000666`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000623`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000365`
- `lag_01__smoke_inv_diff`: contribution `-0.000300`
- `lag_01__T_smoke_inv`: contribution `-0.000237`

### tick `11442`, seconds `8.00`, LSTM delta `+0.0090`

Top all feature movements:
- `lag_11__CT_place_LOWERPARK`: contribution `+0.000741`
- `lag_08__T_place_TSTAIRS`: contribution `+0.000597`
- `lag_13__CT_place_BACKOFA`: contribution `+0.000514`
- `lag_12__CT_place_LOWERPARK`: contribution `+0.000418`
- `lag_10__CT_place_BACKOFA`: contribution `+0.000408`

Top utility-only movements:
- `lag_01__T4__molly`: contribution `+0.000185`

### tick `11506`, seconds `9.00`, LSTM delta `+0.0071`

Top all feature movements:
- `lag_08__T_place_TSTAIRS`: contribution `+0.001194`
- `lag_15__CT_place_BACKOFA`: contribution `+0.000597`
- `lag_10__T_place_TSTAIRS`: contribution `+0.000466`
- `lag_14__CT_place_LOWERPARK`: contribution `+0.000457`
- `lag_02__CT_place_WATER`: contribution `+0.000455`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11666`, seconds `11.50`, LSTM delta `-0.0063`

Top all feature movements:
- `lag_10__T_place_TSTAIRS`: contribution `-0.000933`
- `lag_11__T_place_TSTAIRS`: contribution `-0.000572`
- `lag_13__T_place_TSTAIRS`: contribution `-0.000479`
- `lag_02__CT_place_WATER`: contribution `-0.000455`
- `lag_07__CT_place_WATER`: contribution `-0.000367`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13170`, seconds `35.00`, LSTM delta `-0.0050`

Top all feature movements:
- `lag_01__CT_burning_players`: contribution `-0.000610`
- `lag_00__T1__shots_fired`: contribution `-0.000428`
- `lag_01__T1__shots_fired`: contribution `-0.000384`
- `lag_00__T_shots_fired_sum`: contribution `-0.000271`
- `lag_01__T_shots_fired_sum`: contribution `-0.000245`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.000196`
