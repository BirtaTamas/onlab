# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `110026`, seconds `0.50`, LSTM `0.0153`, delta `-0.0304`
- tick `111402`, seconds `22.00`, LSTM `0.0121`, delta `-0.0073`
- tick `110954`, seconds `15.00`, LSTM `0.0085`, delta `-0.0053`
- tick `110058`, seconds `1.00`, LSTM `0.0100`, delta `-0.0053`
- tick `111178`, seconds `18.50`, LSTM `0.0174`, delta `+0.0043`
- tick `111242`, seconds `19.50`, LSTM `0.0166`, delta `-0.0038`
- tick `111530`, seconds `24.00`, LSTM `0.0104`, delta `-0.0038`
- tick `110570`, seconds `9.00`, LSTM `0.0100`, delta `-0.0035`
- tick `110826`, seconds `13.00`, LSTM `0.0130`, delta `+0.0033`
- tick `111210`, seconds `19.00`, LSTM `0.0204`, delta `+0.0030`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000259`, |coef| `0.000259`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000243`, |coef| `0.000243`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000228`, |coef| `0.000228`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000226`, |coef| `0.000226`
- `lag_01__centroid_distance_xy`: coefficient `-0.000215`, |coef| `0.000215`
- `lag_01__armor_diff`: coefficient `0.000181`, |coef| `0.000181`
- `lag_00__CT_velocity_mean`: coefficient `-0.000175`, |coef| `0.000175`
- `lag_00__T_velocity_mean`: coefficient `-0.000157`, |coef| `0.000157`
- `lag_01__CT_armor_sum`: coefficient `0.000150`, |coef| `0.000150`
- `lag_01__utility_inv_diff`: coefficient `0.000148`, |coef| `0.000148`
- `lag_00__T1__smoke`: coefficient `0.000146`, |coef| `0.000146`
- `lag_01__T1__has_bomb`: coefficient `-0.000145`, |coef| `0.000145`
- `lag_01__flash_inv_diff`: coefficient `0.000134`, |coef| `0.000134`
- `lag_01__equip_diff`: coefficient `0.000127`, |coef| `0.000127`
- `lag_01__molly_inv_diff`: coefficient `0.000125`, |coef| `0.000125`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000148` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000146` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000134` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000125` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000122` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000118` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000105` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000104` (lowers CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000100` (lowers CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000094` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000259` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000243` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000228` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000226` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000215` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000181` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000175` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000157` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000150` (raises CT win probability)
- `lag_01__T1__has_bomb`: coefficient `-0.000145` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `110026`, seconds `0.50`, LSTM delta `-0.0304`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001241`
- `lag_01__T_place_TSPAWN`: contribution `-0.001075`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000986`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000943`
- `lag_01__centroid_distance_xy`: contribution `-0.000850`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000391`
- `lag_00__T1__smoke`: contribution `-0.000314`
- `lag_01__flash_inv_diff`: contribution `-0.000302`
- `lag_01__molly_inv_diff`: contribution `-0.000272`
- `lag_01__T4__utility_total`: contribution `-0.000260`

### tick `111402`, seconds `22.00`, LSTM delta `-0.0073`

Top all feature movements:
- `lag_11__CT_place_IVY`: contribution `-0.000650`
- `lag_15__T_place_DUMPSTER`: contribution `-0.000491`
- `lag_05__T5__flash_duration`: contribution `-0.000481`
- `lag_02__CT3__flash_duration`: contribution `-0.000182`
- `lag_14__T_place_LONGDOG`: contribution `-0.000179`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.000481`
- `lag_02__CT3__flash_duration`: contribution `-0.000182`
- `lag_02__T3__flash_duration`: contribution `-0.000090`
- `lag_05__T_flash_duration_sum`: contribution `-0.000086`

### tick `110954`, seconds `15.00`, LSTM delta `-0.0053`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `-0.000812`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `-0.000671`
- `lag_01__T1__has_bomb`: contribution `-0.000416`
- `lag_12__CT_place_IVY`: contribution `-0.000268`
- `lag_15__T_place_TSTAIRS`: contribution `-0.000234`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110058`, seconds `1.00`, LSTM delta `-0.0053`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000445`
- `lag_02__T_place_TSPAWN`: contribution `-0.000386`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000340`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000336`
- `lag_02__centroid_distance_xy`: contribution `-0.000304`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000155`
- `lag_02__flash_inv_diff`: contribution `-0.000119`
- `lag_02__T1__utility_total`: contribution `-0.000100`
- `lag_02__molly_inv_diff`: contribution `-0.000100`
- `lag_02__T1__flash`: contribution `-0.000098`

### tick `111178`, seconds `18.50`, LSTM delta `+0.0043`

Top all feature movements:
- `lag_04__CT_place_IVY`: contribution `+0.000376`
- `lag_08__T_place_DUMPSTER`: contribution `+0.000374`
- `lag_11__T5__flash_duration`: contribution `+0.000264`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.000250`
- `lag_00__CT_velocity_mean`: contribution `+0.000211`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.000264`
- `lag_01__T4__smoke`: contribution `+0.000191`
- `lag_01__T4__utility_total`: contribution `+0.000091`
