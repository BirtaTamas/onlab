# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `125132`, seconds `69.00`, LSTM `0.8866`, delta `+0.2046`
- tick `122572`, seconds `29.00`, LSTM `0.4775`, delta `-0.1698`
- tick `124460`, seconds `58.50`, LSTM `0.6218`, delta `+0.1522`
- tick `126636`, seconds `92.50`, LSTM `0.9479`, delta `+0.1304`
- tick `122604`, seconds `29.50`, LSTM `0.3666`, delta `-0.1109`
- tick `125484`, seconds `74.50`, LSTM `0.9295`, delta `+0.1045`
- tick `125548`, seconds `75.50`, LSTM `0.8584`, delta `-0.0752`
- tick `122380`, seconds `26.00`, LSTM `0.6709`, delta `+0.0663`
- tick `125708`, seconds `78.00`, LSTM `0.7922`, delta `-0.0568`
- tick `126060`, seconds `83.50`, LSTM `0.8058`, delta `+0.0568`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003214`, |coef| `0.003214`
- `lag_00__CT_kills_last_3s`: coefficient `0.003148`, |coef| `0.003148`
- `lag_00__damage_diff_last_5s`: coefficient `0.002749`, |coef| `0.002749`
- `lag_00__CT_damage_last_5s`: coefficient `0.002190`, |coef| `0.002190`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001970`, |coef| `0.001970`
- `lag_00__T_place_ARCH`: coefficient `-0.001685`, |coef| `0.001685`
- `lag_02__T_place_ARCH`: coefficient `0.001545`, |coef| `0.001545`
- `lag_10__CT2__duck_amount`: coefficient `0.001457`, |coef| `0.001457`
- `lag_01__kill_diff_last_3s`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001395`, |coef| `0.001395`
- `lag_06__CT_place_LIBRARY`: coefficient `0.001273`, |coef| `0.001273`
- `lag_00__T_spread_xy`: coefficient `-0.001265`, |coef| `0.001265`
- `lag_08__CT_flashes_last_5s`: coefficient `-0.001264`, |coef| `0.001264`
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001260`, |coef| `0.001260`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.001223`, |coef| `0.001223`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001395` (lowers CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `-0.001264` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.001194` (raises CT win probability)
- `lag_13__T1__smoke`: coefficient `-0.000956` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.000931` (raises CT win probability)
- `lag_04__T4__molly`: coefficient `-0.000930` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `0.000910` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `-0.000833` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000769` (lowers CT win probability)
- `lag_08__T4__smoke`: coefficient `-0.000765` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003214` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003148` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002749` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002190` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001970` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001685` (lowers CT win probability)
- `lag_02__T_place_ARCH`: coefficient `0.001545` (raises CT win probability)
- `lag_10__CT2__duck_amount`: coefficient `0.001457` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.001419` (raises CT win probability)
- `lag_06__CT_place_LIBRARY`: coefficient `0.001273` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `125132`, seconds `69.00`, LSTM delta `+0.2046`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009090`
- `lag_11__CT_place_LIBRARY`: contribution `+0.008077`
- `lag_00__kill_diff_last_3s`: contribution `+0.007735`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006842`
- `lag_00__damage_diff_last_5s`: contribution `+0.006202`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `+0.003376`

### tick `122572`, seconds `29.00`, LSTM delta `-0.1698`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.015471`
- `lag_00__CT_kills_last_3s`: contribution `-0.009090`
- `lag_00__damage_diff_last_5s`: contribution `-0.008993`
- `lag_05__CT_shots_fired_sum`: contribution `-0.008134`
- `lag_09__CT4__flash_duration`: contribution `-0.007181`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.007181`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.002625`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.002178`

### tick `124460`, seconds `58.50`, LSTM delta `+0.1522`

Top all feature movements:
- `lag_00__T_place_ARCH`: contribution `+0.015676`
- `lag_02__T_place_ARCH`: contribution `+0.014370`
- `lag_00__CT_kills_last_3s`: contribution `+0.009090`
- `lag_06__CT_place_LIBRARY`: contribution `+0.008163`
- `lag_00__kill_diff_last_3s`: contribution `+0.007735`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.002133`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001987`
- `lag_15__T_A_site_active_infernos`: contribution `+0.001983`
- `lag_14__CT1__smoke`: contribution `+0.001588`

### tick `126636`, seconds `92.50`, LSTM delta `+0.1304`

Top all feature movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.013902`
- `lag_00__CT_kills_last_3s`: contribution `+0.009090`
- `lag_00__T_flash_alpha_mean`: contribution `+0.008464`
- `lag_00__kill_diff_last_3s`: contribution `+0.007735`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006842`

Top utility-only movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.013902`
- `lag_00__T_flash_alpha_mean`: contribution `+0.008464`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.002094`

### tick `122604`, seconds `29.50`, LSTM delta `-0.1109`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `-0.007018`
- `lag_01__kill_diff_last_3s`: contribution `-0.006829`
- `lag_06__CT_shots_fired_sum`: contribution `-0.005996`
- `lag_01__damage_diff_last_5s`: contribution `-0.003769`
- `lag_00__T_shots_fired_sum`: contribution `-0.003718`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.007018`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.002180`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.002169`
