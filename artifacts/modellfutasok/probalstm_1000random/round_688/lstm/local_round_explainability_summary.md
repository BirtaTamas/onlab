# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv`
- round_num: `18`

## Largest probability jumps

- tick `128401`, seconds `77.00`, LSTM `0.2090`, delta `+0.1894`
- tick `128049`, seconds `71.50`, LSTM `0.0577`, delta `-0.1878`
- tick `128721`, seconds `82.00`, LSTM `0.1360`, delta `-0.1625`
- tick `127761`, seconds `67.00`, LSTM `0.3056`, delta `+0.1619`
- tick `128465`, seconds `78.00`, LSTM `0.3534`, delta `+0.1165`
- tick `128017`, seconds `71.00`, LSTM `0.2455`, delta `-0.1151`
- tick `128657`, seconds `81.00`, LSTM `0.2988`, delta `-0.0813`
- tick `123505`, seconds `0.50`, LSTM `0.1161`, delta `-0.0782`
- tick `128529`, seconds `79.00`, LSTM `0.3469`, delta `-0.0461`
- tick `127953`, seconds `70.00`, LSTM `0.3741`, delta `+0.0435`

## Top 15 local ridge features

- `lag_09__T_place_TRUCK`: coefficient `0.001853`, |coef| `0.001853`
- `lag_06__T_place_TRUCK`: coefficient `0.001698`, |coef| `0.001698`
- `lag_11__T_place_TRUCK`: coefficient `0.001685`, |coef| `0.001685`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001614`, |coef| `0.001614`
- `lag_00__T2__is_scoped`: coefficient `0.001585`, |coef| `0.001585`
- `lag_04__T_place_TRUCK`: coefficient `0.001553`, |coef| `0.001553`
- `lag_01__T_place_TRUCK`: coefficient `-0.001508`, |coef| `0.001508`
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001495`, |coef| `0.001495`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_00__kill_diff_last_3s`: coefficient `0.001467`, |coef| `0.001467`
- `lag_00__CT1__is_walking`: coefficient `0.001421`, |coef| `0.001421`
- `lag_06__CT_shots_fired_sum`: coefficient `0.001404`, |coef| `0.001404`
- `lag_12__CT_place_JUNGLE`: coefficient `0.001358`, |coef| `0.001358`
- `lag_14__T_place_TRUCK`: coefficient `-0.001276`, |coef| `0.001276`
- `lag_00__CT5__is_scoped`: coefficient `0.001274`, |coef| `0.001274`

## Top 10 utility ridge features

- `lag_13__CT2__flash_duration`: coefficient `0.001175` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000991` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `0.000934` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000930` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000845` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000829` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000782` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000750` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000736` (raises CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000723` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_TRUCK`: coefficient `0.001853` (raises CT win probability)
- `lag_06__T_place_TRUCK`: coefficient `0.001698` (raises CT win probability)
- `lag_11__T_place_TRUCK`: coefficient `0.001685` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001614` (raises CT win probability)
- `lag_00__T2__is_scoped`: coefficient `0.001585` (raises CT win probability)
- `lag_04__T_place_TRUCK`: coefficient `0.001553` (raises CT win probability)
- `lag_01__T_place_TRUCK`: coefficient `-0.001508` (lowers CT win probability)
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001495` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001486` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001467` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `128401`, seconds `77.00`, LSTM delta `+0.1894`

Top all feature movements:
- `lag_04__T_place_TRUCK`: contribution `+0.026964`
- `lag_01__T_place_TRUCK`: contribution `+0.026186`
- `lag_00__T2__is_scoped`: contribution `+0.013968`
- `lag_12__CT_shots_fired_sum`: contribution `+0.007971`
- `lag_11__CT_place_UNDERPASS`: contribution `+0.006763`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.002801`

### tick `128049`, seconds `71.50`, LSTM delta `-0.1878`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `-0.010578`
- `lag_10__T4__shots_fired`: contribution `-0.007947`
- `lag_01__CT_shots_fired_sum`: contribution `-0.007672`
- `lag_10__CT2__flash_duration`: contribution `-0.006490`
- `lag_03__CT_place_SHOP`: contribution `-0.005990`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `-0.006490`

### tick `128721`, seconds `82.00`, LSTM delta `-0.1625`

Top all feature movements:
- `lag_11__T_place_TRUCK`: contribution `-0.029271`
- `lag_14__T_place_TRUCK`: contribution `-0.022164`
- `lag_01__CT_shots_fired_sum`: contribution `-0.008220`
- `lag_13__CT_place_SNIPERSNEST`: contribution `-0.008005`
- `lag_01__T2__is_scoped`: contribution `-0.006417`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127761`, seconds `67.00`, LSTM delta `+0.1619`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.015599`
- `lag_01__T4__shots_fired`: contribution `+0.009870`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008969`
- `lag_13__CT2__flash_duration`: contribution `+0.008166`
- `lag_01__CT2__flash_duration`: contribution `+0.006461`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `+0.008166`
- `lag_01__CT2__flash_duration`: contribution `+0.006461`
- `lag_03__T_B_site_active_infernos`: contribution `+0.002801`

### tick `128465`, seconds `78.00`, LSTM delta `+0.1165`

Top all feature movements:
- `lag_06__T_place_TRUCK`: contribution `+0.029492`
- `lag_03__T_place_TRUCK`: contribution `+0.020579`
- `lag_00__T2__is_scoped`: contribution `-0.013968`
- `lag_11__CT_place_JUNGLE`: contribution `+0.005604`
- `lag_13__CT_place_UNDERPASS`: contribution `+0.004768`

Top utility-only movements:
- No utility movement among the top local contributors.
