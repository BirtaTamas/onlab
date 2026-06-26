# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `36083`, seconds `76.50`, LSTM `0.8272`, delta `+0.3840`
- tick `33459`, seconds `35.50`, LSTM `0.3940`, delta `+0.2863`
- tick `36755`, seconds `87.00`, LSTM `0.5895`, delta `-0.2514`
- tick `31859`, seconds `10.50`, LSTM `0.2399`, delta `-0.2032`
- tick `36915`, seconds `89.50`, LSTM `0.2356`, delta `-0.1503`
- tick `33587`, seconds `37.50`, LSTM `0.4032`, delta `-0.1359`
- tick `33395`, seconds `34.50`, LSTM `0.1473`, delta `+0.0926`
- tick `31891`, seconds `11.00`, LSTM `0.1520`, delta `-0.0878`
- tick `36051`, seconds `76.00`, LSTM `0.4432`, delta `+0.0838`
- tick `36883`, seconds `89.00`, LSTM `0.3859`, delta `-0.0790`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004673`, |coef| `0.004673`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.004289`, |coef| `0.004289`
- `lag_00__damage_diff_last_5s`: coefficient `0.004196`, |coef| `0.004196`
- `lag_00__CT_kills_last_3s`: coefficient `0.004054`, |coef| `0.004054`
- `lag_09__CT_place_TRUCK`: coefficient `-0.003661`, |coef| `0.003661`
- `lag_10__CT_place_TRUCK`: coefficient `-0.003209`, |coef| `0.003209`
- `lag_00__T5__is_scoped`: coefficient `-0.003054`, |coef| `0.003054`
- `lag_00__CT2__duck_amount`: coefficient `0.003020`, |coef| `0.003020`
- `lag_12__T1__duck_amount`: coefficient `0.003013`, |coef| `0.003013`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002981`, |coef| `0.002981`
- `lag_00__CT_closest_enemy_dist`: coefficient `0.002825`, |coef| `0.002825`
- `lag_08__CT_place_TRUCK`: coefficient `-0.002680`, |coef| `0.002680`
- `lag_02__T5__duck_amount`: coefficient `0.002604`, |coef| `0.002604`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002593`, |coef| `0.002593`
- `lag_00__CT_damage_last_5s`: coefficient `0.002578`, |coef| `0.002578`

## Top 10 utility ridge features

- `lag_00__T5__smoke`: coefficient `-0.002125` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001496` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001485` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `0.001406` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001330` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `0.001188` (raises CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `-0.001168` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001105` (raises CT win probability)
- `lag_01__T1__molly`: coefficient `0.001072` (raises CT win probability)
- `lag_14__CT_B_site_active_smokes`: coefficient `-0.000921` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004673` (raises CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.004289` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004196` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004054` (raises CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `-0.003661` (lowers CT win probability)
- `lag_10__CT_place_TRUCK`: coefficient `-0.003209` (lowers CT win probability)
- `lag_00__T5__is_scoped`: coefficient `-0.003054` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.003020` (raises CT win probability)
- `lag_12__T1__duck_amount`: coefficient `0.003013` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002981` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `36083`, seconds `76.50`, LSTM delta `+0.3840`

Top all feature movements:
- `lag_09__CT_place_TRUCK`: contribution `+0.023616`
- `lag_00__closest_enemy_dist_diff`: contribution `+0.015341`
- `lag_00__T5__is_scoped`: contribution `+0.014569`
- `lag_12__T1__duck_amount`: contribution `+0.011799`
- `lag_00__CT_kills_last_3s`: contribution `+0.011704`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33459`, seconds `35.50`, LSTM delta `+0.2863`

Top all feature movements:
- `lag_04__T_place_TRUCK`: contribution `+0.036174`
- `lag_02__T_place_TRUCK`: contribution `+0.029842`
- `lag_00__CT_kills_last_3s`: contribution `+0.011704`
- `lag_00__kill_diff_last_3s`: contribution `+0.011249`
- `lag_03__CT1__duck_amount`: contribution `+0.009539`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `+0.005090`

### tick `36755`, seconds `87.00`, LSTM delta `-0.2514`

Top all feature movements:
- `lag_10__CT_place_TRUCK`: contribution `-0.020696`
- `lag_00__CT_duck_amount_mean`: contribution `-0.015079`
- `lag_00__CT2__duck_amount`: contribution `-0.011505`
- `lag_00__kill_diff_last_3s`: contribution `-0.011249`
- `lag_00__damage_diff_last_5s`: contribution `-0.009467`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31859`, seconds `10.50`, LSTM delta `-0.2032`

Top all feature movements:
- `lag_00__T5__is_scoped`: contribution `+0.014569`
- `lag_00__kill_diff_last_3s`: contribution `-0.011249`
- `lag_00__damage_diff_last_5s`: contribution `-0.010225`
- `lag_04__CT_place_UNDERPASS`: contribution `-0.010068`
- `lag_08__CT2__duck_amount`: contribution `-0.008758`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `-0.004804`
- `lag_02__T_A_site_active_infernos`: contribution `-0.002565`

### tick `36915`, seconds `89.50`, LSTM delta `-0.1503`

Top all feature movements:
- `lag_00__T_place_SCAFFOLDING`: contribution `-0.085733`
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.007873`
- `lag_07__CT2__duck_amount`: contribution `-0.005870`
- `lag_07__CT5__duck_amount`: contribution `+0.004935`
- `lag_05__CT1__duck_amount`: contribution `-0.003984`

Top utility-only movements:
- `lag_01__T1__molly`: contribution `-0.002374`
