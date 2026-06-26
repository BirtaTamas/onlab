# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `44668`, seconds `31.50`, LSTM `0.1315`, delta `+0.0475`
- tick `44764`, seconds `33.00`, LSTM `0.1013`, delta `-0.0365`
- tick `42684`, seconds `0.50`, LSTM `0.0225`, delta `-0.0352`
- tick `44636`, seconds `31.00`, LSTM `0.0840`, delta `+0.0319`
- tick `44604`, seconds `30.50`, LSTM `0.0521`, delta `-0.0231`
- tick `46108`, seconds `54.00`, LSTM `0.0137`, delta `-0.0204`
- tick `44252`, seconds `25.00`, LSTM `0.0609`, delta `+0.0193`
- tick `44892`, seconds `35.00`, LSTM `0.0526`, delta `-0.0170`
- tick `44732`, seconds `32.50`, LSTM `0.1378`, delta `+0.0169`
- tick `44828`, seconds `34.00`, LSTM `0.0848`, delta `-0.0159`

## Top 15 local ridge features

- `lag_01__CT_place_ALLEY`: coefficient `-0.000502`, |coef| `0.000502`
- `lag_03__T_flashed_players`: coefficient `0.000480`, |coef| `0.000480`
- `lag_02__T_place_MAINHALL`: coefficient `0.000427`, |coef| `0.000427`
- `lag_01__T3__duck_amount`: coefficient `-0.000412`, |coef| `0.000412`
- `lag_03__T_place_MAINHALL`: coefficient `0.000396`, |coef| `0.000396`
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000385`, |coef| `0.000385`
- `lag_09__T5__duck_amount`: coefficient `0.000351`, |coef| `0.000351`
- `lag_00__kill_diff_last_3s`: coefficient `0.000344`, |coef| `0.000344`
- `lag_15__T2__is_scoped`: coefficient `-0.000328`, |coef| `0.000328`
- `lag_08__T2__duck_amount`: coefficient `-0.000311`, |coef| `0.000311`
- `lag_04__T_place_MAINHALL`: coefficient `0.000311`, |coef| `0.000311`
- `lag_01__T_place_MAINHALL`: coefficient `0.000297`, |coef| `0.000297`
- `lag_13__T_B_site_active_smokes`: coefficient `0.000295`, |coef| `0.000295`
- `lag_02__CT3__is_walking`: coefficient `-0.000293`, |coef| `0.000293`
- `lag_00__CT_velocity_mean`: coefficient `-0.000283`, |coef| `0.000283`

## Top 10 utility ridge features

- `lag_13__T_B_site_active_smokes`: coefficient `0.000295` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.000276` (raises CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `0.000257` (raises CT win probability)
- `lag_13__T_A_site_active_smokes`: coefficient `0.000252` (raises CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `0.000231` (raises CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `0.000220` (raises CT win probability)
- `lag_13__T_active_smokes`: coefficient `0.000203` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `0.000189` (raises CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `0.000181` (raises CT win probability)
- `lag_11__T_active_smokes`: coefficient `0.000179` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_ALLEY`: coefficient `-0.000502` (lowers CT win probability)
- `lag_03__T_flashed_players`: coefficient `0.000480` (raises CT win probability)
- `lag_02__T_place_MAINHALL`: coefficient `0.000427` (raises CT win probability)
- `lag_01__T3__duck_amount`: coefficient `-0.000412` (lowers CT win probability)
- `lag_03__T_place_MAINHALL`: coefficient `0.000396` (raises CT win probability)
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000385` (lowers CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.000351` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000344` (raises CT win probability)
- `lag_15__T2__is_scoped`: coefficient `-0.000328` (lowers CT win probability)
- `lag_08__T2__duck_amount`: coefficient `-0.000311` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44668`, seconds `31.50`, LSTM delta `+0.0475`

Top all feature movements:
- `lag_15__T2__is_scoped`: contribution `+0.002888`
- `lag_03__T_flashed_players`: contribution `+0.001853`
- `lag_01__T2__is_scoped`: contribution `+0.000994`
- `lag_01__CT3__duck_amount`: contribution `+0.000982`
- `lag_01__CT_shots_fired_sum`: contribution `+0.000951`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44764`, seconds `33.00`, LSTM delta `-0.0365`

Top all feature movements:
- `lag_14__T2__is_scoped`: contribution `-0.002324`
- `lag_03__T_flashed_players`: contribution `-0.001853`
- `lag_00__T_place_SIDEHALL`: contribution `-0.001599`
- `lag_02__T_place_MAINHALL`: contribution `-0.001543`
- `lag_13__T2__duck_amount`: contribution `-0.000883`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `42684`, seconds `0.50`, LSTM delta `-0.0352`

Top all feature movements:
- `lag_01__CT_place_ALLEY`: contribution `-0.003689`
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.002850`
- `lag_01__T_place_TSPAWN`: contribution `-0.001089`
- `lag_00__CT_velocity_mean`: contribution `-0.000942`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000776`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000394`
- `lag_00__T4__smoke`: contribution `-0.000313`
- `lag_01__molly_inv_diff`: contribution `-0.000305`
- `lag_00__T5__smoke`: contribution `-0.000295`
- `lag_01__T1__utility_total`: contribution `-0.000239`

### tick `44636`, seconds `31.00`, LSTM delta `+0.0319`

Top all feature movements:
- `lag_14__T2__is_scoped`: contribution `+0.002324`
- `lag_01__T3__duck_amount`: contribution `+0.001553`
- `lag_01__CT_place_ALLEY`: contribution `+0.001271`
- `lag_10__T2__is_scoped`: contribution `+0.001199`
- `lag_08__T2__duck_amount`: contribution `+0.000992`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.000454`

### tick `44604`, seconds `30.50`, LSTM delta `-0.0231`

Top all feature movements:
- `lag_01__T3__duck_amount`: contribution `-0.001553`
- `lag_13__T2__is_scoped`: contribution `-0.001338`
- `lag_08__T2__duck_amount`: contribution `-0.001190`
- `lag_14__T3__duck_amount`: contribution `-0.000664`
- `lag_01__T_flashed_players`: contribution `-0.000616`

Top utility-only movements:
- `lag_11__T_B_site_active_smokes`: contribution `+0.000389`
