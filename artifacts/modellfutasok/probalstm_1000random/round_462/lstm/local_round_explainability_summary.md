# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `19120`, seconds `0.50`, LSTM `0.0242`, delta `-0.0303`
- tick `20304`, seconds `19.00`, LSTM `0.0456`, delta `-0.0281`
- tick `20336`, seconds `19.50`, LSTM `0.0266`, delta `-0.0189`
- tick `20560`, seconds `23.00`, LSTM `0.0109`, delta `-0.0183`
- tick `20240`, seconds `18.00`, LSTM `0.0691`, delta `+0.0125`
- tick `19696`, seconds `9.50`, LSTM `0.0387`, delta `+0.0090`
- tick `20368`, seconds `20.00`, LSTM `0.0186`, delta `-0.0081`
- tick `19248`, seconds `2.50`, LSTM `0.0326`, delta `+0.0063`
- tick `19984`, seconds `14.00`, LSTM `0.0464`, delta `+0.0062`
- tick `19216`, seconds `2.00`, LSTM `0.0263`, delta `+0.0060`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000419`, |coef| `0.000419`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000365`, |coef| `0.000365`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000343`, |coef| `0.000343`
- `lag_00__CT_place_BACKOFB`: coefficient `0.000340`, |coef| `0.000340`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000338`, |coef| `0.000338`
- `lag_01__centroid_distance_xy`: coefficient `-0.000322`, |coef| `0.000322`
- `lag_00__T_velocity_mean`: coefficient `-0.000319`, |coef| `0.000319`
- `lag_15__T_place_IVY`: coefficient `-0.000274`, |coef| `0.000274`
- `lag_00__CT_velocity_mean`: coefficient `-0.000271`, |coef| `0.000271`
- `lag_00__T1__shots_fired`: coefficient `-0.000253`, |coef| `0.000253`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__molly_inv_diff`: coefficient `0.000240`, |coef| `0.000240`
- `lag_01__CT4__flash`: coefficient `-0.000229`, |coef| `0.000229`
- `lag_01__armor_diff`: coefficient `0.000216`, |coef| `0.000216`
- `lag_01__T3__has_bomb`: coefficient `-0.000214`, |coef| `0.000214`

## Top 10 utility ridge features

- `lag_01__molly_inv_diff`: coefficient `0.000240` (raises CT win probability)
- `lag_01__CT4__flash`: coefficient `-0.000229` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.000196` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000195` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000195` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `0.000187` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000171` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000167` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000158` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000154` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000419` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000365` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000343` (lowers CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.000340` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000338` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000322` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000319` (lowers CT win probability)
- `lag_15__T_place_IVY`: coefficient `-0.000274` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000271` (lowers CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.000253` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `19120`, seconds `0.50`, LSTM delta `-0.0303`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002003`
- `lag_01__T_place_TSPAWN`: contribution `-0.001618`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001409`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001356`
- `lag_01__centroid_distance_xy`: contribution `-0.001251`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000671`
- `lag_01__CT4__flash`: contribution `-0.000630`
- `lag_01__T_smoke_inv`: contribution `-0.000446`
- `lag_01__T_molly_inv`: contribution `-0.000442`
- `lag_01__smoke_inv_diff`: contribution `-0.000318`

### tick `20304`, seconds `19.00`, LSTM delta `-0.0281`

Top all feature movements:
- `lag_15__T_place_IVY`: contribution `-0.001462`
- `lag_01__T5__flash_duration`: contribution `-0.001190`
- `lag_00__T_shots_fired_sum`: contribution `-0.000941`
- `lag_14__CT_place_BACKOFB`: contribution `-0.000888`
- `lag_10__T_shots_fired_sum`: contribution `-0.000831`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `-0.001190`
- `lag_01__T1__flash_duration`: contribution `-0.000800`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.000693`
- `lag_01__T_flash_duration_sum`: contribution `-0.000623`

### tick `20336`, seconds `19.50`, LSTM delta `-0.0189`

Top all feature movements:
- `lag_00__CT_place_BACKOFB`: contribution `-0.001941`
- `lag_02__T5__flash_duration`: contribution `-0.001038`
- `lag_00__T_shots_fired_sum`: contribution `-0.000941`
- `lag_00__T1__shots_fired`: contribution `-0.000756`
- `lag_02__T1__flash_duration`: contribution `-0.000639`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.001038`
- `lag_02__T1__flash_duration`: contribution `-0.000639`
- `lag_02__T_flash_duration_sum`: contribution `-0.000548`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.000404`

### tick `20560`, seconds `23.00`, LSTM delta `-0.0183`

Top all feature movements:
- `lag_00__CT_place_BACKOFB`: contribution `-0.001941`
- `lag_01__CT_place_ENTRANCE`: contribution `-0.001223`
- `lag_00__T_shots_fired_sum`: contribution `-0.001129`
- `lag_05__T1__shots_fired`: contribution `-0.001032`
- `lag_05__T_shots_fired_sum`: contribution `-0.001007`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.000531`
- `lag_00__T1__flash_duration`: contribution `-0.000476`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.000445`
- `lag_09__T1__flash_duration`: contribution `-0.000329`
- `lag_09__T_flash_duration_sum`: contribution `-0.000283`

### tick `20240`, seconds `18.00`, LSTM delta `+0.0125`

Top all feature movements:
- `lag_12__T2__duck_amount`: contribution `+0.000746`
- `lag_10__T_shots_fired_sum`: contribution `+0.000693`
- `lag_12__CT_place_BACKOFB`: contribution `+0.000540`
- `lag_09__T3__duck_amount`: contribution `+0.000504`
- `lag_01__T3__duck_amount`: contribution `+0.000449`

Top utility-only movements:
- `lag_14__T_utility_damage_last_5s`: contribution `+0.000445`
