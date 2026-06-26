# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `21373`, seconds `0.50`, LSTM `0.0252`, delta `-0.0347`
- tick `21405`, seconds `1.00`, LSTM `0.0190`, delta `-0.0062`
- tick `21917`, seconds `9.00`, LSTM `0.0223`, delta `+0.0042`
- tick `22333`, seconds `15.50`, LSTM `0.0172`, delta `-0.0041`
- tick `22013`, seconds `10.50`, LSTM `0.0221`, delta `-0.0026`
- tick `24157`, seconds `44.00`, LSTM `0.0079`, delta `+0.0026`
- tick `21725`, seconds `6.00`, LSTM `0.0212`, delta `+0.0026`
- tick `21949`, seconds `9.50`, LSTM `0.0249`, delta `+0.0026`
- tick `22397`, seconds `16.50`, LSTM `0.0129`, delta `-0.0023`
- tick `21597`, seconds `4.00`, LSTM `0.0208`, delta `+0.0023`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000304`, |coef| `0.000304`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000286`, |coef| `0.000286`
- `lag_00__T_velocity_mean`: coefficient `-0.000247`, |coef| `0.000247`
- `lag_01__utility_inv_diff`: coefficient `0.000236`, |coef| `0.000236`
- `lag_01__smoke_inv_diff`: coefficient `0.000217`, |coef| `0.000217`
- `lag_01__molly_inv_diff`: coefficient `0.000190`, |coef| `0.000190`
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000188`, |coef| `0.000188`
- `lag_00__CT_velocity_mean`: coefficient `-0.000188`, |coef| `0.000188`
- `lag_01__armor_diff`: coefficient `0.000182`, |coef| `0.000182`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000179`, |coef| `0.000179`
- `lag_01__T2__has_bomb`: coefficient `-0.000176`, |coef| `0.000176`
- `lag_01__flash_inv_diff`: coefficient `0.000175`, |coef| `0.000175`
- `lag_01__equip_diff`: coefficient `0.000167`, |coef| `0.000167`
- `lag_01__CT_armor_sum`: coefficient `0.000166`, |coef| `0.000166`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000161`, |coef| `0.000161`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000236` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000217` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000190` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000175` (raises CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000155` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000153` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000152` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000152` (lowers CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000146` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000132` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000304` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000286` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000247` (lowers CT win probability)
- `lag_01__T_round_start_equip_sum`: coefficient `-0.000188` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000188` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000182` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000179` (lowers CT win probability)
- `lag_01__T2__has_bomb`: coefficient `-0.000176` (lowers CT win probability)
- `lag_01__equip_diff`: coefficient `0.000167` (raises CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000166` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `21373`, seconds `0.50`, LSTM delta `-0.0347`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001455`
- `lag_01__T_place_TSPAWN`: contribution `-0.001266`
- `lag_00__T_velocity_mean`: contribution `-0.000869`
- `lag_01__utility_inv_diff`: contribution `-0.000830`
- `lag_01__smoke_inv_diff`: contribution `-0.000692`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000830`
- `lag_01__smoke_inv_diff`: contribution `-0.000692`
- `lag_01__molly_inv_diff`: contribution `-0.000530`
- `lag_01__flash_inv_diff`: contribution `-0.000467`
- `lag_01__T_utility_inv`: contribution `-0.000369`

### tick `21405`, seconds `1.00`, LSTM delta `-0.0062`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000562`
- `lag_02__T_place_TSPAWN`: contribution `-0.000477`
- `lag_02__utility_inv_diff`: contribution `-0.000315`
- `lag_02__smoke_inv_diff`: contribution `-0.000278`
- `lag_02__molly_inv_diff`: contribution `-0.000211`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000315`
- `lag_02__smoke_inv_diff`: contribution `-0.000278`
- `lag_02__molly_inv_diff`: contribution `-0.000211`
- `lag_02__flash_inv_diff`: contribution `-0.000162`
- `lag_02__T1__utility_total`: contribution `-0.000138`

### tick `21917`, seconds `9.00`, LSTM delta `+0.0042`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `+0.000617`
- `lag_12__CT_flashes_last_5s`: contribution `+0.000398`
- `lag_12__T_flashes_last_5s`: contribution `+0.000271`
- `lag_04__CT_place_ARAMP`: contribution `+0.000244`
- `lag_05__CT_place_SHORTSTAIRS`: contribution `+0.000230`

Top utility-only movements:
- `lag_12__CT_flashes_last_5s`: contribution `+0.000398`
- `lag_12__T_flashes_last_5s`: contribution `+0.000271`
- `lag_02__CT_flashes_last_5s`: contribution `+0.000115`
- `lag_02__T_flashes_last_5s`: contribution `+0.000070`

### tick `22333`, seconds `15.50`, LSTM delta `-0.0041`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.000617`
- `lag_15__CT_flashes_last_5s`: contribution `-0.000395`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.000319`
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `-0.000282`
- `lag_15__T_flashes_last_5s`: contribution `-0.000269`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `-0.000395`
- `lag_15__T_flashes_last_5s`: contribution `-0.000269`

### tick `22013`, seconds `10.50`, LSTM delta `-0.0026`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.000617`
- `lag_15__CT_flashes_last_5s`: contribution `+0.000395`
- `lag_12__CT_place_ARAMP`: contribution `-0.000299`
- `lag_05__CT_flashes_last_5s`: contribution `-0.000282`
- `lag_15__T_flashes_last_5s`: contribution `+0.000269`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.000395`
- `lag_05__CT_flashes_last_5s`: contribution `-0.000282`
- `lag_15__T_flashes_last_5s`: contribution `+0.000269`
- `lag_05__T_flashes_last_5s`: contribution `-0.000194`
