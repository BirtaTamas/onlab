# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `61899`, seconds `65.00`, LSTM `0.9139`, delta `+0.2250`
- tick `59851`, seconds `33.00`, LSTM `0.4620`, delta `-0.2215`
- tick `59435`, seconds `26.50`, LSTM `0.5599`, delta `-0.1404`
- tick `59563`, seconds `28.50`, LSTM `0.6241`, delta `+0.0847`
- tick `61643`, seconds `61.00`, LSTM `0.5716`, delta `+0.0735`
- tick `63243`, seconds `86.00`, LSTM `0.8901`, delta `+0.0721`
- tick `61675`, seconds `61.50`, LSTM `0.6320`, delta `+0.0604`
- tick `62923`, seconds `81.00`, LSTM `0.8839`, delta `-0.0587`
- tick `59211`, seconds `23.00`, LSTM `0.6165`, delta `+0.0484`
- tick `59243`, seconds `23.50`, LSTM `0.6600`, delta `+0.0435`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003288`, |coef| `0.003288`
- `lag_00__kill_diff_last_3s`: coefficient `0.002964`, |coef| `0.002964`
- `lag_14__T_place_ARCH`: coefficient `-0.002500`, |coef| `0.002500`
- `lag_13__T_place_ARCH`: coefficient `-0.002403`, |coef| `0.002403`
- `lag_00__CT_kills_last_3s`: coefficient `0.002163`, |coef| `0.002163`
- `lag_00__CT_damage_last_5s`: coefficient `0.002036`, |coef| `0.002036`
- `lag_00__T1__flash`: coefficient `-0.001870`, |coef| `0.001870`
- `lag_03__CT_place_LIBRARY`: coefficient `0.001799`, |coef| `0.001799`
- `lag_14__T_flashed_players`: coefficient `-0.001644`, |coef| `0.001644`
- `lag_00__T1__utility_total`: coefficient `-0.001592`, |coef| `0.001592`
- `lag_00__T_kills_last_3s`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_12__CT_place_RUINS`: coefficient `0.001510`, |coef| `0.001510`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001478`, |coef| `0.001478`
- `lag_01__CT_place_LIBRARY`: coefficient `0.001476`, |coef| `0.001476`
- `lag_04__CT4__flash_duration`: coefficient `0.001388`, |coef| `0.001388`

## Top 10 utility ridge features

- `lag_00__T1__flash`: coefficient `-0.001870` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.001592` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.001388` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001284` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000980` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `-0.000881` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000810` (raises CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000757` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000755` (lowers CT win probability)
- `lag_00__utility_inv_diff`: coefficient `0.000753` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003288` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002964` (raises CT win probability)
- `lag_14__T_place_ARCH`: coefficient `-0.002500` (lowers CT win probability)
- `lag_13__T_place_ARCH`: coefficient `-0.002403` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002163` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002036` (raises CT win probability)
- `lag_03__CT_place_LIBRARY`: coefficient `0.001799` (raises CT win probability)
- `lag_14__T_flashed_players`: coefficient `-0.001644` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001527` (lowers CT win probability)
- `lag_12__CT_place_RUINS`: coefficient `0.001510` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `61899`, seconds `65.00`, LSTM delta `+0.2250`

Top all feature movements:
- `lag_14__T_place_ARCH`: contribution `+0.023259`
- `lag_13__T_place_ARCH`: contribution `+0.022360`
- `lag_01__CT_place_LIBRARY`: contribution `+0.009462`
- `lag_00__damage_diff_last_5s`: contribution `+0.007417`
- `lag_00__kill_diff_last_3s`: contribution `+0.007133`

Top utility-only movements:
- `lag_00__T1__flash`: contribution `+0.005204`
- `lag_00__T1__utility_total`: contribution `+0.003742`

### tick `59851`, seconds `33.00`, LSTM delta `-0.2215`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.012312`
- `lag_04__CT4__flash_duration`: contribution `-0.009450`
- `lag_00__kill_diff_last_3s`: contribution `-0.007133`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.005676`
- `lag_00__T_kills_last_3s`: contribution `-0.004839`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `-0.009450`
- `lag_00__T1__flash`: contribution `-0.002602`

### tick `59435`, seconds `26.50`, LSTM delta `-0.1404`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `-0.018501`
- `lag_13__T_place_BALCONY`: contribution `-0.010974`
- `lag_00__damage_diff_last_5s`: contribution `-0.007417`
- `lag_00__kill_diff_last_3s`: contribution `-0.007133`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.005676`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.002720`

### tick `59563`, seconds `28.50`, LSTM delta `+0.0847`

Top all feature movements:
- `lag_15__T_place_BALCONY`: contribution `+0.016240`
- `lag_00__kill_diff_last_3s`: contribution `+0.007133`
- `lag_00__CT_kills_last_3s`: contribution `+0.006245`
- `lag_07__CT4__is_walking`: contribution `+0.002939`
- `lag_12__CT4__is_walking`: contribution `-0.002833`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.002194`

### tick `61643`, seconds `61.00`, LSTM delta `+0.0735`

Top all feature movements:
- `lag_06__T_place_ARCH`: contribution `+0.009888`
- `lag_00__damage_diff_last_5s`: contribution `+0.007417`
- `lag_05__T_place_ARCH`: contribution `+0.007350`
- `lag_00__kill_diff_last_3s`: contribution `+0.007133`
- `lag_00__CT_kills_last_3s`: contribution `+0.006245`

Top utility-only movements:
- No utility movement among the top local contributors.
