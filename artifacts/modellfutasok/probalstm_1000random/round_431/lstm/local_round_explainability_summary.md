# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `18`

## Largest probability jumps

- tick `180960`, seconds `0.50`, LSTM `0.0443`, delta `-0.0552`
- tick `182624`, seconds `26.50`, LSTM `0.0379`, delta `-0.0356`
- tick `182400`, seconds `23.00`, LSTM `0.0573`, delta `+0.0247`
- tick `181536`, seconds `9.50`, LSTM `0.0197`, delta `-0.0229`
- tick `180992`, seconds `1.00`, LSTM `0.0304`, delta `-0.0139`
- tick `182720`, seconds `28.00`, LSTM `0.0165`, delta `-0.0124`
- tick `182528`, seconds `25.00`, LSTM `0.0788`, delta `+0.0111`
- tick `182464`, seconds `24.00`, LSTM `0.0681`, delta `+0.0090`
- tick `181120`, seconds `3.00`, LSTM `0.0407`, delta `+0.0081`
- tick `182240`, seconds `20.50`, LSTM `0.0324`, delta `+0.0068`

## Top 15 local ridge features

- `lag_00__CT_place_IVY`: coefficient `-0.000503`, |coef| `0.000503`
- `lag_00__T_he_last_5s`: coefficient `-0.000463`, |coef| `0.000463`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000447`, |coef| `0.000447`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000427`, |coef| `0.000427`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000400`, |coef| `0.000400`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000385`, |coef| `0.000385`
- `lag_00__T_velocity_mean`: coefficient `-0.000377`, |coef| `0.000377`
- `lag_01__centroid_distance_xy`: coefficient `-0.000374`, |coef| `0.000374`
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000340`, |coef| `0.000340`
- `lag_00__CT_velocity_mean`: coefficient `-0.000311`, |coef| `0.000311`
- `lag_01__utility_inv_diff`: coefficient `0.000299`, |coef| `0.000299`
- `lag_05__T_place_ELECTRICALBOX`: coefficient `0.000271`, |coef| `0.000271`
- `lag_01__molly_inv_diff`: coefficient `0.000258`, |coef| `0.000258`
- `lag_00__T4__smoke`: coefficient `0.000247`, |coef| `0.000247`
- `lag_00__CT_place_DUMPSTER`: coefficient `-0.000246`, |coef| `0.000246`

## Top 10 utility ridge features

- `lag_00__T_he_last_5s`: coefficient `-0.000463` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000299` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000258` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `0.000247` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000243` (raises CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000225` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000223` (raises CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000214` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000203` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000196` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_IVY`: coefficient `-0.000503` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000447` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000427` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000400` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000385` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000377` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000374` (lowers CT win probability)
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000340` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000311` (lowers CT win probability)
- `lag_05__T_place_ELECTRICALBOX`: coefficient `0.000271` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `180960`, seconds `0.50`, LSTM delta `-0.0552`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002139`
- `lag_01__T_place_TSPAWN`: contribution `-0.001893`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001598`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001580`
- `lag_01__centroid_distance_xy`: contribution `-0.001442`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000919`
- `lag_01__molly_inv_diff`: contribution `-0.000721`
- `lag_01__smoke_inv_diff`: contribution `-0.000569`
- `lag_01__flash_inv_diff`: contribution `-0.000547`
- `lag_00__T4__smoke`: contribution `-0.000538`

### tick `182624`, seconds `26.50`, LSTM delta `-0.0356`

Top all feature movements:
- `lag_00__CT_place_DUMPSTER`: contribution `-0.012676`
- `lag_05__T_place_ELECTRICALBOX`: contribution `-0.007123`
- `lag_10__CT_place_IVY`: contribution `-0.002351`
- `lag_08__T_place_ELECTRICALBOX`: contribution `-0.001535`
- `lag_07__CT_place_IVY`: contribution `-0.001442`

Top utility-only movements:
- `lag_01__T1__molly`: contribution `+0.000211`
- `lag_07__T_A_site_active_infernos`: contribution `-0.000193`
- `lag_12__T_A_site_active_infernos`: contribution `-0.000183`

### tick `182400`, seconds `23.00`, LSTM delta `+0.0247`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `+0.005746`
- `lag_01__T_place_ELECTRICALBOX`: contribution `+0.004946`
- `lag_03__CT_place_IVY`: contribution `+0.002157`
- `lag_05__CT_place_ENTRANCE`: contribution `+0.001115`
- `lag_09__CT_place_ENTRANCE`: contribution `-0.000695`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `+0.000545`
- `lag_05__T_A_site_active_infernos`: contribution `+0.000272`

### tick `181536`, seconds `9.50`, LSTM delta `-0.0229`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `-0.011493`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.001896`
- `lag_13__CT_place_ENTRANCE`: contribution `-0.001294`
- `lag_14__CT_place_ENTRANCE`: contribution `-0.001019`
- `lag_07__T_he_last_5s`: contribution `-0.000815`

Top utility-only movements:
- `lag_07__T_he_last_5s`: contribution `-0.000815`
- `lag_01__T3__smoke`: contribution `+0.000263`
- `lag_01__smoke_inv_diff`: contribution `+0.000143`

### tick `180992`, seconds `1.00`, LSTM delta `-0.0139`

Top all feature movements:
- `lag_00__T_he_last_5s`: contribution `-0.006049`
- `lag_02__T_place_TSPAWN`: contribution `-0.000680`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000607`
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000550`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000531`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `-0.006049`
- `lag_02__utility_inv_diff`: contribution `-0.000345`
- `lag_02__molly_inv_diff`: contribution `-0.000240`
- `lag_02__smoke_inv_diff`: contribution `-0.000220`
- `lag_02__flash_inv_diff`: contribution `-0.000219`
