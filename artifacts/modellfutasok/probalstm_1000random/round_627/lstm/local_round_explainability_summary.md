# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `53828`, seconds `0.50`, LSTM `0.0253`, delta `-0.0309`
- tick `54372`, seconds `9.00`, LSTM `0.0216`, delta `-0.0111`
- tick `55396`, seconds `25.00`, LSTM `0.0111`, delta `-0.0058`
- tick `54852`, seconds `16.50`, LSTM `0.0330`, delta `+0.0055`
- tick `54276`, seconds `7.50`, LSTM `0.0366`, delta `+0.0055`
- tick `55172`, seconds `21.50`, LSTM `0.0293`, delta `-0.0054`
- tick `55012`, seconds `19.00`, LSTM `0.0362`, delta `+0.0052`
- tick `54500`, seconds `11.00`, LSTM `0.0264`, delta `+0.0048`
- tick `55268`, seconds `23.00`, LSTM `0.0258`, delta `-0.0047`
- tick `55364`, seconds `24.50`, LSTM `0.0169`, delta `-0.0042`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000586`, |coef| `0.000586`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000220`, |coef| `0.000220`
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000185`, |coef| `0.000185`
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `-0.000139`, |coef| `0.000139`
- `lag_00__T_velocity_mean`: coefficient `-0.000127`, |coef| `0.000127`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000125`, |coef| `0.000125`
- `lag_00__CT_place_BRIDGE`: coefficient `-0.000123`, |coef| `0.000123`
- `lag_11__CT_place_OUTSIDELONG`: coefficient `0.000117`, |coef| `0.000117`
- `lag_00__CT_velocity_mean`: coefficient `-0.000117`, |coef| `0.000117`
- `lag_01__utility_inv_diff`: coefficient `0.000098`, |coef| `0.000098`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000093`, |coef| `0.000093`
- `lag_13__CT_place_CTSIDEUPPER`: coefficient `0.000092`, |coef| `0.000092`
- `lag_01__armor_diff`: coefficient `0.000091`, |coef| `0.000091`
- `lag_01__flash_inv_diff`: coefficient `0.000091`, |coef| `0.000091`
- `lag_08__CT_place_OUTSIDELONG`: coefficient `0.000090`, |coef| `0.000090`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000098` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000091` (raises CT win probability)
- `lag_01__T_flash_inv`: coefficient `-0.000077` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000071` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000071` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000070` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000065` (lowers CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000063` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000063` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000062` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000586` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000220` (raises CT win probability)
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000185` (lowers CT win probability)
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `-0.000139` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000127` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000125` (lowers CT win probability)
- `lag_00__CT_place_BRIDGE`: coefficient `-0.000123` (lowers CT win probability)
- `lag_11__CT_place_OUTSIDELONG`: coefficient `0.000117` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000117` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000093` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `53828`, seconds `0.50`, LSTM delta `-0.0309`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.015098`
- `lag_01__T_place_TSPAWN`: contribution `-0.000554`
- `lag_00__T_velocity_mean`: contribution `-0.000464`
- `lag_00__CT_velocity_mean`: contribution `-0.000400`
- `lag_01__utility_inv_diff`: contribution `-0.000344`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000344`
- `lag_01__flash_inv_diff`: contribution `-0.000281`
- `lag_01__T_flash_inv`: contribution `-0.000216`
- `lag_01__T_utility_inv`: contribution `-0.000199`
- `lag_01__molly_inv_diff`: contribution `-0.000196`

### tick `54372`, seconds `9.00`, LSTM delta `-0.0111`

Top all feature movements:
- `lag_00__CT_place_BRIDGE`: contribution `-0.002823`
- `lag_13__CT_place_CTSIDEUPPER`: contribution `-0.001431`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `-0.000513`
- `lag_15__CT_place_CTSIDEUPPER`: contribution `-0.000434`
- `lag_07__T_place_STREET`: contribution `-0.000413`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55396`, seconds `25.00`, LSTM delta `-0.0058`

Top all feature movements:
- `lag_06__CT_place_BRICKS`: contribution `-0.001077`
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.000840`
- `lag_09__CT_place_BRIDGE`: contribution `-0.000599`
- `lag_00__CT_place_TSPAWN`: contribution `-0.000356`
- `lag_05__T_place_MAIN`: contribution `-0.000339`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `+0.000062`

### tick `54852`, seconds `16.50`, LSTM delta `+0.0055`

Top all feature movements:
- `lag_11__CT_place_OUTSIDELONG`: contribution `+0.001184`
- `lag_12__CT_place_OUTSIDELONG`: contribution `+0.000758`
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.000712`
- `lag_15__CT_place_BRIDGE`: contribution `+0.000353`
- `lag_11__T_place_STREET`: contribution `-0.000321`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `+0.000253`

### tick `54276`, seconds `7.50`, LSTM delta `+0.0055`

Top all feature movements:
- `lag_15__CT_place_CTSIDEUPPER`: contribution `+0.002167`
- `lag_10__CT_place_CTSIDEUPPER`: contribution `+0.000951`
- `lag_11__CT_place_PALACEINTERIOR`: contribution `+0.000201`
- `lag_12__CT_place_CTSIDEUPPER`: contribution `-0.000192`
- `lag_06__CT_place_LOWERTUNNEL`: contribution `+0.000158`

Top utility-only movements:
- `lag_01__T4__smoke`: contribution `+0.000097`
- `lag_01__T4__utility_total`: contribution `+0.000049`
