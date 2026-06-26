# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `83608`, seconds `80.00`, LSTM `0.1646`, delta `-0.3613`
- tick `83640`, seconds `80.50`, LSTM `0.0505`, delta `-0.1141`
- tick `82776`, seconds `67.00`, LSTM `0.3665`, delta `+0.0709`
- tick `78520`, seconds `0.50`, LSTM `0.1619`, delta `-0.0670`
- tick `83512`, seconds `78.50`, LSTM `0.5184`, delta `-0.0636`
- tick `82968`, seconds `70.00`, LSTM `0.4670`, delta `+0.0564`
- tick `82104`, seconds `56.50`, LSTM `0.1872`, delta `+0.0441`
- tick `83288`, seconds `75.00`, LSTM `0.5269`, delta `+0.0413`
- tick `83416`, seconds `77.00`, LSTM `0.5558`, delta `+0.0368`
- tick `79960`, seconds `23.00`, LSTM `0.2168`, delta `+0.0368`

## Top 15 local ridge features

- `lag_11__T_place_QUAD`: coefficient `-0.002118`, |coef| `0.002118`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002098`, |coef| `0.002098`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001851`, |coef| `0.001851`
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.001158`, |coef| `0.001158`
- `lag_12__T_place_QUAD`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_00__T_place_BALCONY`: coefficient `-0.000986`, |coef| `0.000986`
- `lag_00__damage_diff_last_5s`: coefficient `0.000970`, |coef| `0.000970`
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.000969`, |coef| `0.000969`
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000951`, |coef| `0.000951`
- `lag_09__CT4__flash_duration`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.000916`, |coef| `0.000916`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000912`, |coef| `0.000912`
- `lag_09__CT2__flash_duration`: coefficient `-0.000900`, |coef| `0.000900`
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.000880`, |coef| `0.000880`
- `lag_00__CT_place_RUINS`: coefficient `0.000835`, |coef| `0.000835`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002098` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001851` (raises CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.001158` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.000969` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000951` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000948` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.000916` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000912` (raises CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000900` (lowers CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.000880` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_QUAD`: coefficient `-0.002118` (lowers CT win probability)
- `lag_12__T_place_QUAD`: coefficient `-0.001036` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.000986` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000970` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.000835` (raises CT win probability)
- `lag_04__T_place_TOPOFMID`: coefficient `0.000827` (raises CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `0.000778` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000762` (lowers CT win probability)
- `lag_07__T_place_QUAD`: coefficient `0.000759` (raises CT win probability)
- `lag_03__T_place_BACKALLEY`: coefficient `-0.000749` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `83608`, seconds `80.00`, LSTM delta `-0.3613`

Top all feature movements:
- `lag_11__T_place_QUAD`: contribution `-0.102011`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.022857`
- `lag_07__T_place_QUAD`: contribution `-0.018274`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.016541`
- `lag_03__T_place_QUAD`: contribution `-0.016371`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.022857`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.016541`
- `lag_09__CT2__flash_duration`: contribution `-0.006753`
- `lag_09__CT4__flash_duration`: contribution `-0.006532`
- `lag_02__CT2__flash_duration`: contribution `-0.006178`

### tick `83640`, seconds `80.50`, LSTM delta `-0.1141`

Top all feature movements:
- `lag_12__T_place_QUAD`: contribution `-0.049897`
- `lag_07__T_place_QUAD`: contribution `+0.018274`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.010359`
- `lag_08__T_place_QUAD`: contribution `+0.009240`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.008149`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.010359`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.008149`
- `lag_03__CT2__flash_duration`: contribution `-0.005170`
- `lag_10__CT4__flash_duration`: contribution `-0.004206`
- `lag_03__T3__flash_duration`: contribution `-0.003933`

### tick `82776`, seconds `67.00`, LSTM delta `+0.0709`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.018008`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.013032`
- `lag_12__CT5__duck_amount`: contribution `+0.002624`
- `lag_01__CT2__duck_amount`: contribution `+0.001958`
- `lag_05__CT5__duck_amount`: contribution `+0.001837`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.018008`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.013032`

### tick `78520`, seconds `0.50`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `-0.002663`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002619`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002571`
- `lag_01__centroid_distance_xy`: contribution `-0.002333`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002255`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000980`
- `lag_01__T_smoke_inv`: contribution `-0.000923`
- `lag_01__T4__molly`: contribution `-0.000757`
- `lag_01__utility_inv_diff`: contribution `-0.000756`
- `lag_01__T3__smoke`: contribution `-0.000621`

### tick `83512`, seconds `78.50`, LSTM delta `-0.0636`

Top all feature movements:
- `lag_08__T_place_QUAD`: contribution `-0.018479`
- `lag_03__T_place_QUAD`: contribution `+0.016371`
- `lag_00__T_place_QUAD`: contribution `-0.005032`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.004140`
- `lag_13__utility_damage_diff_last_5s`: contribution `-0.003309`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.004140`
- `lag_13__utility_damage_diff_last_5s`: contribution `-0.003309`
- `lag_13__CT_utility_damage_last_5s`: contribution `-0.003244`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.002653`
- `lag_06__CT2__flash_duration`: contribution `-0.002457`
