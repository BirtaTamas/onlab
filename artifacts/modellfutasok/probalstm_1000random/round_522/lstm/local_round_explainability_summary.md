# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `58013`, seconds `0.50`, LSTM `0.0147`, delta `-0.0366`
- tick `58589`, seconds `9.50`, LSTM `0.0224`, delta `-0.0140`
- tick `58621`, seconds `10.00`, LSTM `0.0343`, delta `+0.0119`
- tick `61853`, seconds `60.50`, LSTM `0.0464`, delta `+0.0098`
- tick `58557`, seconds `9.00`, LSTM `0.0364`, delta `+0.0075`
- tick `61789`, seconds `59.50`, LSTM `0.0370`, delta `+0.0067`
- tick `59517`, seconds `24.00`, LSTM `0.0160`, delta `-0.0066`
- tick `61213`, seconds `50.50`, LSTM `0.0242`, delta `+0.0065`
- tick `59485`, seconds `23.50`, LSTM `0.0226`, delta `-0.0064`
- tick `62653`, seconds `73.00`, LSTM `0.0086`, delta `-0.0063`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000530`, |coef| `0.000530`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000312`, |coef| `0.000312`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000290`, |coef| `0.000290`
- `lag_00__T_velocity_mean`: coefficient `-0.000236`, |coef| `0.000236`
- `lag_00__T_bomb_carrier_alive`: coefficient `0.000230`, |coef| `0.000230`
- `lag_00__T3__has_bomb`: coefficient `0.000223`, |coef| `0.000223`
- `lag_00__CT_velocity_mean`: coefficient `-0.000219`, |coef| `0.000219`
- `lag_00__CT1__is_walking`: coefficient `0.000210`, |coef| `0.000210`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000199`, |coef| `0.000199`
- `lag_00__T_place_JUNGLE`: coefficient `-0.000194`, |coef| `0.000194`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000190`, |coef| `0.000190`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000190`, |coef| `0.000190`
- `lag_01__CT_walking_count`: coefficient `0.000188`, |coef| `0.000188`
- `lag_01__centroid_distance_xy`: coefficient `-0.000183`, |coef| `0.000183`
- `lag_01__T_flashes_last_5s`: coefficient `-0.000173`, |coef| `0.000173`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000530` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000173` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `0.000130` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000128` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000127` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000122` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000107` (raises CT win probability)
- `lag_01__T3__smoke`: coefficient `-0.000107` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000102` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000100` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_TSPAWN`: coefficient `-0.000312` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000290` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000236` (lowers CT win probability)
- `lag_00__T_bomb_carrier_alive`: coefficient `0.000230` (raises CT win probability)
- `lag_00__T3__has_bomb`: coefficient `0.000223` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000219` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000210` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.000199` (lowers CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.000194` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000190` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58013`, seconds `0.50`, LSTM delta `-0.0366`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.004800`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001389`
- `lag_01__T_place_TSPAWN`: contribution `-0.001383`
- `lag_00__CT_velocity_mean`: contribution `-0.000762`
- `lag_00__T_velocity_mean`: contribution `-0.000741`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.004800`
- `lag_01__smoke_inv_diff`: contribution `-0.000326`
- `lag_01__utility_inv_diff`: contribution `-0.000278`
- `lag_01__molly_inv_diff`: contribution `-0.000265`
- `lag_00__T1__smoke`: contribution `-0.000232`

### tick `58589`, seconds `9.50`, LSTM delta `-0.0140`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.004155`
- `lag_08__T_flashes_last_5s`: contribution `-0.001181`
- `lag_06__CT_place_SNIPERSNEST`: contribution `-0.000503`
- `lag_11__CT2__duck_amount`: contribution `-0.000430`
- `lag_05__CT4__duck_amount`: contribution `-0.000377`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `-0.001181`

### tick `58621`, seconds `10.00`, LSTM delta `+0.0119`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.004155`
- `lag_01__CT_place_SCAFFOLDING`: contribution `+0.001159`
- `lag_06__T3__is_scoped`: contribution `+0.000470`
- `lag_00__T3__duck_amount`: contribution `+0.000461`
- `lag_09__T_flashes_last_5s`: contribution `+0.000450`

Top utility-only movements:
- `lag_09__T_flashes_last_5s`: contribution `+0.000450`

### tick `61853`, seconds `60.50`, LSTM delta `+0.0098`

Top all feature movements:
- `lag_03__T_place_UNDERPASS`: contribution `+0.000523`
- `lag_00__T3__duck_amount`: contribution `+0.000461`
- `lag_04__CT5__duck_amount`: contribution `+0.000295`
- `lag_00__T1__is_walking`: contribution `+0.000285`
- `lag_00__T1__duck_amount`: contribution `+0.000272`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58557`, seconds `9.00`, LSTM delta `+0.0075`

Top all feature movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.000814`
- `lag_07__CT_place_SHOP`: contribution `+0.000565`
- `lag_05__CT4__duck_amount`: contribution `+0.000377`
- `lag_03__CT_place_SHOP`: contribution `-0.000352`
- `lag_06__CT4__duck_amount`: contribution `+0.000287`

Top utility-only movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.000814`
