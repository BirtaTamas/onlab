# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `23997`, seconds `11.00`, LSTM `0.0447`, delta `-0.1298`
- tick `23325`, seconds `0.50`, LSTM `0.1953`, delta `-0.0748`
- tick `23837`, seconds `8.50`, LSTM `0.1336`, delta `-0.0325`
- tick `25501`, seconds `34.50`, LSTM `0.0622`, delta `-0.0291`
- tick `26365`, seconds `48.00`, LSTM `0.0890`, delta `+0.0268`
- tick `23869`, seconds `9.00`, LSTM `0.1595`, delta `+0.0259`
- tick `25341`, seconds `32.00`, LSTM `0.0449`, delta `+0.0251`
- tick `23357`, seconds `1.00`, LSTM `0.1710`, delta `-0.0243`
- tick `25437`, seconds `33.50`, LSTM `0.0724`, delta `+0.0229`
- tick `25469`, seconds `34.00`, LSTM `0.0913`, delta `+0.0189`

## Top 15 local ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.001121`, |coef| `0.001121`
- `lag_03__CT_place_BDOORS`: coefficient `-0.000958`, |coef| `0.000958`
- `lag_00__T_flashes_last_5s`: coefficient `-0.000870`, |coef| `0.000870`
- `lag_13__T_flashes_last_5s`: coefficient `-0.000841`, |coef| `0.000841`
- `lag_15__CT_place_MIDDOORS`: coefficient `-0.000798`, |coef| `0.000798`
- `lag_04__CT_place_HOLE`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_11__CT_flashes_last_5s`: coefficient `0.000668`, |coef| `0.000668`
- `lag_02__T_place_LONGDOORS`: coefficient `-0.000659`, |coef| `0.000659`
- `lag_10__T_place_LONGDOORS`: coefficient `-0.000658`, |coef| `0.000658`
- `lag_02__CT2__is_scoped`: coefficient `-0.000623`, |coef| `0.000623`
- `lag_11__T_flashes_last_5s`: coefficient `0.000597`, |coef| `0.000597`
- `lag_07__T_place_LONGDOORS`: coefficient `-0.000570`, |coef| `0.000570`
- `lag_00__T_velocity_mean`: coefficient `-0.000562`, |coef| `0.000562`
- `lag_00__T_place_SIDE`: coefficient `-0.000558`, |coef| `0.000558`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000558`, |coef| `0.000558`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.001121` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000870` (lowers CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000841` (lowers CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `0.000668` (raises CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `0.000597` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000459` (raises CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `0.000382` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000379` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `0.000369` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `-0.000362` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_BDOORS`: coefficient `-0.000958` (lowers CT win probability)
- `lag_15__CT_place_MIDDOORS`: coefficient `-0.000798` (lowers CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `-0.000695` (lowers CT win probability)
- `lag_02__T_place_LONGDOORS`: coefficient `-0.000659` (lowers CT win probability)
- `lag_10__T_place_LONGDOORS`: coefficient `-0.000658` (lowers CT win probability)
- `lag_02__CT2__is_scoped`: coefficient `-0.000623` (lowers CT win probability)
- `lag_07__T_place_LONGDOORS`: coefficient `-0.000570` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000562` (lowers CT win probability)
- `lag_00__T_place_SIDE`: coefficient `-0.000558` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000558` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23997`, seconds `11.00`, LSTM delta `-0.1298`

Top all feature movements:
- `lag_03__CT_place_BDOORS`: contribution `-0.009219`
- `lag_04__CT_place_HOLE`: contribution `-0.007759`
- `lag_13__T_flashes_last_5s`: contribution `-0.007621`
- `lag_11__CT_flashes_last_5s`: contribution `-0.007346`
- `lag_11__T_flashes_last_5s`: contribution `-0.005412`

Top utility-only movements:
- `lag_13__T_flashes_last_5s`: contribution `-0.007621`
- `lag_11__CT_flashes_last_5s`: contribution `-0.007346`
- `lag_11__T_flashes_last_5s`: contribution `-0.005412`
- `lag_03__T_flashes_last_5s`: contribution `-0.003464`

### tick `23325`, seconds `0.50`, LSTM delta `-0.0748`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.012323`
- `lag_00__T_flashes_last_5s`: contribution `-0.007882`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002407`
- `lag_01__T_place_TSPAWN`: contribution `-0.002129`
- `lag_00__T_velocity_mean`: contribution `-0.002032`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.012323`
- `lag_00__T_flashes_last_5s`: contribution `-0.007882`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.001304`
- `lag_00__CT4__flash`: contribution `-0.001257`
- `lag_01__T4__utility_total`: contribution `-0.000518`

### tick `23837`, seconds `8.50`, LSTM delta `-0.0325`

Top all feature movements:
- `lag_03__CT_place_BDOORS`: contribution `-0.004610`
- `lag_06__T_flashes_last_5s`: contribution `-0.003346`
- `lag_06__CT_flashes_last_5s`: contribution `-0.003166`
- `lag_08__T_flashes_last_5s`: contribution `-0.001818`
- `lag_02__T_place_LONGDOORS`: contribution `-0.001766`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.003346`
- `lag_06__CT_flashes_last_5s`: contribution `-0.003166`
- `lag_08__T_flashes_last_5s`: contribution `-0.001818`
- `lag_00__CT4__flash`: contribution `+0.000628`
- `lag_02__CT_flash_alpha_mean`: contribution `+0.000449`

### tick `25501`, seconds `34.50`, LSTM delta `-0.0291`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `-0.010797`
- `lag_01__T4__flash_duration`: contribution `-0.002390`
- `lag_08__CT2__is_scoped`: contribution `-0.001846`
- `lag_10__T_place_LONGDOORS`: contribution `-0.001764`
- `lag_12__T3__is_scoped`: contribution `-0.001685`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.002390`

### tick `26365`, seconds `48.00`, LSTM delta `+0.0268`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `+0.010797`
- `lag_03__CT2__is_scoped`: contribution `+0.001843`
- `lag_00__CT2__is_scoped`: contribution `+0.001465`
- `lag_06__CT2__is_scoped`: contribution `+0.001341`
- `lag_00__T_place_LONGA`: contribution `+0.001105`

Top utility-only movements:
- No utility movement among the top local contributors.
