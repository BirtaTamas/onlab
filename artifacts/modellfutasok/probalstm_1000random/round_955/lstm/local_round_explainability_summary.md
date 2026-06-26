# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `17`

## Largest probability jumps

- tick `148893`, seconds `23.00`, LSTM `0.1921`, delta `-0.0964`
- tick `149373`, seconds `30.50`, LSTM `0.0667`, delta `-0.0722`
- tick `147453`, seconds `0.50`, LSTM `0.2158`, delta `-0.0602`
- tick `148317`, seconds `14.00`, LSTM `0.3646`, delta `+0.0580`
- tick `148285`, seconds `13.50`, LSTM `0.3066`, delta `-0.0462`
- tick `148957`, seconds `24.00`, LSTM `0.1269`, delta `-0.0403`
- tick `147965`, seconds `8.50`, LSTM `0.2822`, delta `+0.0396`
- tick `147485`, seconds `1.00`, LSTM `0.1771`, delta `-0.0388`
- tick `147869`, seconds `7.00`, LSTM `0.2240`, delta `+0.0387`
- tick `149629`, seconds `34.50`, LSTM `0.0435`, delta `-0.0364`

## Top 15 local ridge features

- `lag_11__CT_place_TSIDELOWER`: coefficient `-0.001205`, |coef| `0.001205`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000991`, |coef| `0.000991`
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.000986`, |coef| `0.000986`
- `lag_00__T_flashed_players`: coefficient `-0.000857`, |coef| `0.000857`
- `lag_00__CT1__shots_fired`: coefficient `0.000808`, |coef| `0.000808`
- `lag_01__CT_place_MAINHALL`: coefficient `-0.000762`, |coef| `0.000762`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000733`, |coef| `0.000733`
- `lag_08__CT_place_SIDEENTRANCE`: coefficient `0.000723`, |coef| `0.000723`
- `lag_02__CT_place_MAINHALL`: coefficient `-0.000684`, |coef| `0.000684`
- `lag_00__T_macro_A`: coefficient `-0.000670`, |coef| `0.000670`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.000670`, |coef| `0.000670`
- `lag_00__CT_place_MAINHALL`: coefficient `0.000637`, |coef| `0.000637`
- `lag_00__CT1__flash_duration`: coefficient `0.000627`, |coef| `0.000627`
- `lag_13__CT_place_TSIDELOWER`: coefficient `-0.000608`, |coef| `0.000608`
- `lag_12__CT_place_TSIDELOWER`: coefficient `-0.000590`, |coef| `0.000590`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `0.000627` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000478` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000475` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000433` (raises CT win probability)
- `lag_03__CT_active_smokes`: coefficient `0.000422` (raises CT win probability)
- `lag_01__CT3__flash`: coefficient `-0.000411` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `0.000389` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000375` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000373` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000369` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_TSIDELOWER`: coefficient `-0.001205` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000991` (raises CT win probability)
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.000986` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.000857` (lowers CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `0.000808` (raises CT win probability)
- `lag_01__CT_place_MAINHALL`: coefficient `-0.000762` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000733` (raises CT win probability)
- `lag_08__CT_place_SIDEENTRANCE`: coefficient `0.000723` (raises CT win probability)
- `lag_02__CT_place_MAINHALL`: coefficient `-0.000684` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.000670` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `148893`, seconds `23.00`, LSTM delta `-0.0964`

Top all feature movements:
- `lag_11__CT_place_TSIDELOWER`: contribution `-0.016368`
- `lag_03__CT_place_TSIDELOWER`: contribution `-0.013389`
- `lag_00__T_flashed_players`: contribution `-0.008272`
- `lag_06__CT1__flash_duration`: contribution `-0.002921`
- `lag_11__CT_place_TSIDEUPPER`: contribution `-0.002634`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `-0.002921`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.001488`
- `lag_06__T_A_site_active_infernos`: contribution `-0.001098`

### tick `149373`, seconds `30.50`, LSTM delta `-0.0722`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.017907`
- `lag_00__CT1__shots_fired`: contribution `-0.011107`
- `lag_15__T_flashed_players`: contribution `-0.004062`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.003123`
- `lag_12__T_flashed_players`: contribution `-0.002318`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `147453`, seconds `0.50`, LSTM delta `-0.0602`

Top all feature movements:
- `lag_01__CT_place_MAINHALL`: contribution `-0.006241`
- `lag_01__T_place_TSPAWN`: contribution `-0.002155`
- `lag_00__T_velocity_mean`: contribution `-0.002010`
- `lag_01__CT_place_HOUSE`: contribution `-0.001810`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001513`

Top utility-only movements:
- `lag_01__CT3__flash`: contribution `-0.001239`
- `lag_01__CT3__utility_total`: contribution `-0.000811`
- `lag_01__T1__molly`: contribution `-0.000549`
- `lag_01__T4__flash`: contribution `-0.000517`
- `lag_01__T5__smoke`: contribution `+0.000494`

### tick `148317`, seconds `14.00`, LSTM delta `+0.0580`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.005509`
- `lag_00__CT1__flash_duration`: contribution `+0.003540`
- `lag_08__CT_place_SIDEENTRANCE`: contribution `+0.002910`
- `lag_00__CT1__duck_amount`: contribution `+0.002243`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.002172`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.003540`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.001631`
- `lag_00__CT_flash_duration_sum`: contribution `+0.000846`
- `lag_01__CT3__flash`: contribution `+0.000759`
- `lag_00__CT_active_infernos`: contribution `+0.000734`

### tick `148285`, seconds `13.50`, LSTM delta `-0.0462`

Top all feature movements:
- `lag_00__CT1__duck_amount`: contribution `-0.002243`
- `lag_13__CT_place_HOUSE`: contribution `-0.002035`
- `lag_00__T_place_MAINHALL`: contribution `+0.001887`
- `lag_04__T_place_MAINHALL`: contribution `-0.001421`
- `lag_13__CT_place_TOPOFMID`: contribution `-0.001173`

Top utility-only movements:
- `lag_01__T5__smoke`: contribution `-0.000714`
- `lag_01__CT1__flash_duration`: contribution `+0.000576`
- `lag_00__CT3__flash`: contribution `-0.000556`
