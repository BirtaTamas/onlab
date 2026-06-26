# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `73918`, seconds `41.00`, LSTM `0.7771`, delta `+0.1240`
- tick `74142`, seconds `44.50`, LSTM `0.9493`, delta `+0.0856`
- tick `73886`, seconds `40.50`, LSTM `0.6530`, delta `+0.0813`
- tick `73982`, seconds `42.00`, LSTM `0.8869`, delta `+0.0650`
- tick `71390`, seconds `1.50`, LSTM `0.6775`, delta `-0.0542`
- tick `72414`, seconds `17.50`, LSTM `0.6224`, delta `-0.0504`
- tick `73950`, seconds `41.50`, LSTM `0.8218`, delta `+0.0447`
- tick `71358`, seconds `1.00`, LSTM `0.7318`, delta `+0.0433`
- tick `74014`, seconds `42.50`, LSTM `0.8446`, delta `-0.0422`
- tick `72446`, seconds `18.00`, LSTM `0.5915`, delta `-0.0309`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001151`, |coef| `0.001151`
- `lag_00__CT4__shots_fired`: coefficient `0.001047`, |coef| `0.001047`
- `lag_10__CT_flashes_last_5s`: coefficient `0.001022`, |coef| `0.001022`
- `lag_01__CT4__shots_fired`: coefficient `0.000832`, |coef| `0.000832`
- `lag_03__CT_place_UNKNOWN`: coefficient `-0.000825`, |coef| `0.000825`
- `lag_07__CT4__flash_duration`: coefficient `0.000773`, |coef| `0.000773`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000713`, |coef| `0.000713`
- `lag_09__CT_flashes_last_5s`: coefficient `0.000698`, |coef| `0.000698`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000663`, |coef| `0.000663`
- `lag_00__T_place_TSIDELOWER`: coefficient `0.000663`, |coef| `0.000663`
- `lag_07__CT_flash_duration_sum`: coefficient `0.000652`, |coef| `0.000652`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `-0.000637`, |coef| `0.000637`
- `lag_07__CT3__flash_duration`: coefficient `0.000624`, |coef| `0.000624`
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000617`, |coef| `0.000617`
- `lag_00__CT_kills_last_3s`: coefficient `0.000613`, |coef| `0.000613`

## Top 10 utility ridge features

- `lag_10__CT_flashes_last_5s`: coefficient `0.001022` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.000773` (raises CT win probability)
- `lag_09__CT_flashes_last_5s`: coefficient `0.000698` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.000652` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.000624` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000617` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000564` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000544` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.000528` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000519` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001151` (raises CT win probability)
- `lag_00__CT4__shots_fired`: coefficient `0.001047` (raises CT win probability)
- `lag_01__CT4__shots_fired`: coefficient `0.000832` (raises CT win probability)
- `lag_03__CT_place_UNKNOWN`: coefficient `-0.000825` (lowers CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000713` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000663` (raises CT win probability)
- `lag_00__T_place_TSIDELOWER`: coefficient `0.000663` (raises CT win probability)
- `lag_03__CT_place_TSIDEUPPER`: coefficient `-0.000637` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000613` (raises CT win probability)
- `lag_06__CT4__shots_fired`: coefficient `0.000596` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `73918`, seconds `41.00`, LSTM delta `+0.1240`

Top all feature movements:
- `lag_10__CT_flashes_last_5s`: contribution `+0.011236`
- `lag_00__CT_flashes_last_5s`: contribution `+0.006779`
- `lag_07__CT4__flash_duration`: contribution `+0.006191`
- `lag_07__T_flashed_players`: contribution `+0.004335`
- `lag_07__CT_flash_duration_sum`: contribution `+0.004328`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `+0.011236`
- `lag_00__CT_flashes_last_5s`: contribution `+0.006779`
- `lag_07__CT4__flash_duration`: contribution `+0.006191`
- `lag_07__CT_flash_duration_sum`: contribution `+0.004328`
- `lag_07__CT3__flash_duration`: contribution `+0.004274`

### tick `74142`, seconds `44.50`, LSTM delta `+0.0856`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.005004`
- `lag_03__T3__shots_fired`: contribution `+0.004226`
- `lag_03__T_shots_fired_sum`: contribution `+0.004179`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003998`
- `lag_07__CT_flashes_last_5s`: contribution `+0.002948`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `+0.002948`
- `lag_14__CT4__flash_duration`: contribution `+0.002346`
- `lag_14__T_flash_duration_sum`: contribution `+0.001553`
- `lag_14__CT_flash_duration_sum`: contribution `+0.001524`
- `lag_14__CT3__flash_duration`: contribution `+0.001334`

### tick `73886`, seconds `40.50`, LSTM delta `+0.0813`

Top all feature movements:
- `lag_09__CT_flashes_last_5s`: contribution `+0.007676`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003998`
- `lag_06__T_flashed_players`: contribution `+0.003106`
- `lag_00__CT4__shots_fired`: contribution `+0.002821`
- `lag_06__T_flash_duration_sum`: contribution `+0.002573`

Top utility-only movements:
- `lag_09__CT_flashes_last_5s`: contribution `+0.007676`
- `lag_06__T_flash_duration_sum`: contribution `+0.002573`
- `lag_00__T2__flash_duration`: contribution `+0.002243`
- `lag_06__CT4__flash_duration`: contribution `+0.002225`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002011`

### tick `73982`, seconds `42.00`, LSTM delta `+0.0650`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007997`
- `lag_12__CT_flashes_last_5s`: contribution `+0.003795`
- `lag_00__CT4__shots_fired`: contribution `+0.002821`
- `lag_01__CT4__shots_fired`: contribution `+0.002240`
- `lag_06__T3__duck_amount`: contribution `-0.001947`

Top utility-only movements:
- `lag_12__CT_flashes_last_5s`: contribution `+0.003795`
- `lag_09__CT4__flash_duration`: contribution `+0.001884`
- `lag_02__T4__flash_duration`: contribution `+0.001877`
- `lag_09__CT_flash_duration_sum`: contribution `+0.001439`
- `lag_09__CT3__flash_duration`: contribution `+0.001423`

### tick `71390`, seconds `1.50`, LSTM delta `-0.0542`

Top all feature movements:
- `lag_03__CT_place_UNKNOWN`: contribution `-0.028959`
- `lag_00__CT_place_UNKNOWN`: contribution `+0.010505`
- `lag_01__T_mollies_last_5s`: contribution `-0.010018`
- `lag_00__T_smokes_last_5s`: contribution `-0.005082`
- `lag_01__CT_place_UNKNOWN`: contribution `-0.002168`

Top utility-only movements:
- `lag_01__T_mollies_last_5s`: contribution `-0.010018`
- `lag_00__T_smokes_last_5s`: contribution `-0.005082`
- `lag_00__T_flashes_last_5s`: contribution `-0.001941`
- `lag_00__T3__flash`: contribution `-0.000655`
- `lag_03__CT5__molly`: contribution `-0.000461`
