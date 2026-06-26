# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `32841`, seconds `47.50`, LSTM `0.5724`, delta `+0.2900`
- tick `35625`, seconds `91.00`, LSTM `0.8567`, delta `+0.2430`
- tick `33257`, seconds `54.00`, LSTM `0.7784`, delta `+0.1565`
- tick `33833`, seconds `63.00`, LSTM `0.6114`, delta `+0.1243`
- tick `33737`, seconds `61.50`, LSTM `0.4689`, delta `-0.1204`
- tick `33609`, seconds `59.50`, LSTM `0.6236`, delta `-0.1192`
- tick `32809`, seconds `47.00`, LSTM `0.2823`, delta `-0.0533`
- tick `32969`, seconds `49.50`, LSTM `0.5650`, delta `-0.0456`
- tick `35305`, seconds `86.00`, LSTM `0.6478`, delta `+0.0453`
- tick `32393`, seconds `40.50`, LSTM `0.3323`, delta `-0.0431`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002952`, |coef| `0.002952`
- `lag_00__CT_kills_last_3s`: coefficient `0.002876`, |coef| `0.002876`
- `lag_13__CT_place_ARCH`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_00__T3__has_bomb`: coefficient `-0.001997`, |coef| `0.001997`
- `lag_00__CT_flashed_players`: coefficient `0.001887`, |coef| `0.001887`
- `lag_08__CT5__is_scoped`: coefficient `0.001804`, |coef| `0.001804`
- `lag_04__CT5__is_scoped`: coefficient `-0.001752`, |coef| `0.001752`
- `lag_00__damage_diff_last_5s`: coefficient `0.001729`, |coef| `0.001729`
- `lag_00__T3__alive`: coefficient `-0.001717`, |coef| `0.001717`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001696`, |coef| `0.001696`
- `lag_11__T_place_ARCH`: coefficient `0.001664`, |coef| `0.001664`
- `lag_00__bomb_events_last_5s`: coefficient `0.001658`, |coef| `0.001658`
- `lag_01__T_bomb_zone_count`: coefficient `0.001636`, |coef| `0.001636`
- `lag_08__CT_flash_duration_sum`: coefficient `0.001590`, |coef| `0.001590`
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001589`, |coef| `0.001589`

## Top 10 utility ridge features

- `lag_08__CT_flash_duration_sum`: coefficient `0.001590` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001589` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001401` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001379` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `0.001310` (raises CT win probability)
- `lag_03__CT3__flash`: coefficient `-0.001233` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001213` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.001190` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001131` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.000973` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002952` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002876` (raises CT win probability)
- `lag_13__CT_place_ARCH`: coefficient `-0.002249` (lowers CT win probability)
- `lag_00__T3__has_bomb`: coefficient `-0.001997` (lowers CT win probability)
- `lag_00__CT_flashed_players`: coefficient `0.001887` (raises CT win probability)
- `lag_08__CT5__is_scoped`: coefficient `0.001804` (raises CT win probability)
- `lag_04__CT5__is_scoped`: coefficient `-0.001752` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001729` (raises CT win probability)
- `lag_00__T3__alive`: coefficient `-0.001717` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001696` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `32841`, seconds `47.50`, LSTM delta `+0.2900`

Top all feature movements:
- `lag_08__CT_flash_duration_sum`: contribution `+0.011887`
- `lag_09__T4__flash_duration`: contribution `+0.010573`
- `lag_00__T4__flash_duration`: contribution `+0.010111`
- `lag_08__CT1__flash_duration`: contribution `+0.008342`
- `lag_00__CT_kills_last_3s`: contribution `+0.008302`

Top utility-only movements:
- `lag_08__CT_flash_duration_sum`: contribution `+0.011887`
- `lag_09__T4__flash_duration`: contribution `+0.010573`
- `lag_00__T4__flash_duration`: contribution `+0.010111`
- `lag_08__CT1__flash_duration`: contribution `+0.008342`
- `lag_08__CT2__flash_duration`: contribution `+0.007279`

### tick `35625`, seconds `91.00`, LSTM delta `+0.2430`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.009873`
- `lag_01__T_bomb_zone_count`: contribution `+0.009526`
- `lag_13__CT_place_ARCH`: contribution `+0.009177`
- `lag_00__CT_kills_last_3s`: contribution `+0.008302`
- `lag_00__CT_flashed_players`: contribution `+0.008266`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.004729`
- `lag_00__CT5__flash_duration`: contribution `+0.003339`

### tick `33257`, seconds `54.00`, LSTM delta `+0.1565`

Top all feature movements:
- `lag_11__T_place_ARCH`: contribution `+0.015483`
- `lag_10__T_place_ARCH`: contribution `+0.012311`
- `lag_00__CT_kills_last_3s`: contribution `+0.008302`
- `lag_00__kill_diff_last_3s`: contribution `+0.007105`
- `lag_10__CT_shots_fired_sum`: contribution `+0.006376`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.004336`
- `lag_10__CT2__flash_duration`: contribution `+0.001713`

### tick `33833`, seconds `63.00`, LSTM delta `+0.1243`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008302`
- `lag_00__kill_diff_last_3s`: contribution `+0.007105`
- `lag_13__CT_place_BALCONY`: contribution `+0.006086`
- `lag_13__CT_place_PIT`: contribution `+0.003715`
- `lag_14__CT3__duck_amount`: contribution `+0.003496`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33737`, seconds `61.50`, LSTM delta `-0.1204`

Top all feature movements:
- `lag_12__T_place_ARCH`: contribution `-0.012480`
- `lag_00__kill_diff_last_3s`: contribution `-0.007105`
- `lag_03__CT5__is_scoped`: contribution `-0.005280`
- `lag_14__CT_shots_fired_sum`: contribution `-0.003628`
- `lag_11__CT3__duck_amount`: contribution `+0.003210`

Top utility-only movements:
- No utility movement among the top local contributors.
