# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `45255`, seconds `0.50`, LSTM `0.0181`, delta `-0.0361`
- tick `46311`, seconds `17.00`, LSTM `0.0287`, delta `-0.0300`
- tick `45735`, seconds `8.00`, LSTM `0.0320`, delta `-0.0162`
- tick `45767`, seconds `8.50`, LSTM `0.0195`, delta `-0.0126`
- tick `46343`, seconds `17.50`, LSTM `0.0168`, delta `-0.0119`
- tick `46247`, seconds `16.00`, LSTM `0.0578`, delta `+0.0108`
- tick `45607`, seconds `6.00`, LSTM `0.0290`, delta `+0.0101`
- tick `45639`, seconds `6.50`, LSTM `0.0385`, delta `+0.0095`
- tick `45991`, seconds `12.00`, LSTM `0.0278`, delta `+0.0087`
- tick `45671`, seconds `7.00`, LSTM `0.0467`, delta `+0.0082`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000497`, |coef| `0.000497`
- `lag_00__CT_place_ADMIN`: coefficient `0.000279`, |coef| `0.000279`
- `lag_03__CT_place_OBSERVATION`: coefficient `0.000275`, |coef| `0.000275`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000252`, |coef| `0.000252`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000250`, |coef| `0.000250`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000238`, |coef| `0.000238`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000236`, |coef| `0.000236`
- `lag_01__centroid_distance_xy`: coefficient `-0.000218`, |coef| `0.000218`
- `lag_03__CT_place_HELL`: coefficient `0.000189`, |coef| `0.000189`
- `lag_01__smoke_inv_diff`: coefficient `0.000185`, |coef| `0.000185`
- `lag_00__T_velocity_mean`: coefficient `-0.000183`, |coef| `0.000183`
- `lag_06__CT_place_OBSERVATION`: coefficient `-0.000183`, |coef| `0.000183`
- `lag_00__CT_velocity_mean`: coefficient `-0.000178`, |coef| `0.000178`
- `lag_01__T_flashes_last_5s`: coefficient `-0.000176`, |coef| `0.000176`
- `lag_01__utility_inv_diff`: coefficient `0.000162`, |coef| `0.000162`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000497` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000185` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000176` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000162` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000131` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000129` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000128` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000128` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000105` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000101` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_ADMIN`: coefficient `0.000279` (raises CT win probability)
- `lag_03__CT_place_OBSERVATION`: coefficient `0.000275` (raises CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000252` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000250` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000238` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000236` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000218` (lowers CT win probability)
- `lag_03__CT_place_HELL`: coefficient `0.000189` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000183` (lowers CT win probability)
- `lag_06__CT_place_OBSERVATION`: coefficient `-0.000183` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `45255`, seconds `0.50`, LSTM delta `-0.0361`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.004508`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001206`
- `lag_01__T_place_TSPAWN`: contribution `-0.001106`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001034`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001004`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.004508`
- `lag_01__smoke_inv_diff`: contribution `-0.000588`
- `lag_01__utility_inv_diff`: contribution `-0.000428`
- `lag_01__T1__utility_total`: contribution `-0.000297`
- `lag_01__T_smoke_inv`: contribution `-0.000291`

### tick `46311`, seconds `17.00`, LSTM delta `-0.0300`

Top all feature movements:
- `lag_03__CT_place_OBSERVATION`: contribution `-0.004782`
- `lag_06__CT_place_OBSERVATION`: contribution `-0.003182`
- `lag_01__CT_place_OBSERVATION`: contribution `-0.002222`
- `lag_05__CT_place_OBSERVATION`: contribution `-0.001790`
- `lag_09__CT_place_CONTROL`: contribution `-0.001337`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45735`, seconds `8.00`, LSTM delta `-0.0162`

Top all feature movements:
- `lag_00__CT_place_ADMIN`: contribution `-0.003874`
- `lag_03__CT_place_HELL`: contribution `-0.002050`
- `lag_03__CT_place_ADMIN`: contribution `-0.001479`
- `lag_06__CT_place_HELL`: contribution `-0.001254`
- `lag_15__T_flashes_last_5s`: contribution `-0.000856`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `-0.000856`
- `lag_05__T_flashes_last_5s`: contribution `-0.000604`

### tick `45767`, seconds `8.50`, LSTM delta `-0.0126`

Top all feature movements:
- `lag_00__CT_place_ADMIN`: contribution `-0.001937`
- `lag_04__CT_place_ADMIN`: contribution `-0.001505`
- `lag_01__CT_place_ADMIN`: contribution `-0.001325`
- `lag_03__CT_place_HELL`: contribution `-0.001025`
- `lag_02__CT_place_HELL`: contribution `-0.000790`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.000313`

### tick `46343`, seconds `17.50`, LSTM delta `-0.0119`

Top all feature movements:
- `lag_06__CT_place_OBSERVATION`: contribution `-0.003182`
- `lag_04__CT_place_OBSERVATION`: contribution `-0.002639`
- `lag_02__CT_place_OBSERVATION`: contribution `-0.000794`
- `lag_01__CT_place_CONTROL`: contribution `-0.000758`
- `lag_10__CT_place_CONTROL`: contribution `-0.000711`

Top utility-only movements:
- No utility movement among the top local contributors.
