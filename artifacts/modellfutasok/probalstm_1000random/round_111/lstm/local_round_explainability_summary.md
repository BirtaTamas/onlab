# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `5734`, seconds `0.50`, LSTM `0.0130`, delta `-0.0317`
- tick `10246`, seconds `71.00`, LSTM `0.0096`, delta `-0.0207`
- tick `10214`, seconds `70.50`, LSTM `0.0303`, delta `+0.0182`
- tick `10086`, seconds `68.50`, LSTM `0.0120`, delta `-0.0172`
- tick `5766`, seconds `1.00`, LSTM `0.0075`, delta `-0.0055`
- tick `9542`, seconds `60.00`, LSTM `0.0236`, delta `-0.0052`
- tick `9478`, seconds `59.00`, LSTM `0.0270`, delta `+0.0050`
- tick `9286`, seconds `56.00`, LSTM `0.0190`, delta `+0.0038`
- tick `10150`, seconds `69.50`, LSTM `0.0091`, delta `-0.0037`
- tick `9926`, seconds `66.00`, LSTM `0.0242`, delta `-0.0036`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000326`, |coef| `0.000326`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000304`, |coef| `0.000304`
- `lag_00__T_velocity_mean`: coefficient `-0.000260`, |coef| `0.000260`
- `lag_00__CT_velocity_mean`: coefficient `-0.000246`, |coef| `0.000246`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000208`, |coef| `0.000208`
- `lag_14__T_place_PALACEALLEY`: coefficient `0.000207`, |coef| `0.000207`
- `lag_00__kill_diff_last_3s`: coefficient `0.000195`, |coef| `0.000195`
- `lag_01__armor_diff`: coefficient `0.000193`, |coef| `0.000193`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000191`, |coef| `0.000191`
- `lag_01__centroid_distance_xy`: coefficient `-0.000188`, |coef| `0.000188`
- `lag_00__T_kills_last_3s`: coefficient `-0.000185`, |coef| `0.000185`
- `lag_01__smoke_inv_diff`: coefficient `0.000183`, |coef| `0.000183`
- `lag_01__CT_armor_sum`: coefficient `0.000163`, |coef| `0.000163`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000150`, |coef| `0.000150`
- `lag_02__CT_place_CTSPAWN`: coefficient `-0.000150`, |coef| `0.000150`

## Top 10 utility ridge features

- `lag_01__smoke_inv_diff`: coefficient `0.000183` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000132` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000119` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000111` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000109` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000097` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000094` (raises CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000093` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.000092` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000091` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000326` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000304` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000260` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000246` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000208` (lowers CT win probability)
- `lag_14__T_place_PALACEALLEY`: coefficient `0.000207` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000195` (raises CT win probability)
- `lag_01__armor_diff`: coefficient `0.000193` (raises CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000191` (lowers CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000188` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5734`, seconds `0.50`, LSTM delta `-0.0317`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001560`
- `lag_01__T_place_TSPAWN`: contribution `-0.001347`
- `lag_00__T_velocity_mean`: contribution `-0.000948`
- `lag_00__CT_velocity_mean`: contribution `-0.000842`
- `lag_01__smoke_inv_diff`: contribution `-0.000583`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000583`
- `lag_01__utility_inv_diff`: contribution `-0.000289`
- `lag_01__T_smoke_inv`: contribution `-0.000270`

### tick `10246`, seconds `71.00`, LSTM delta `-0.0207`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.000653`
- `lag_00__T_kills_last_3s`: contribution `-0.000587`
- `lag_01__T3__flash_duration`: contribution `-0.000580`
- `lag_00__T_shots_fired_sum`: contribution `-0.000564`
- `lag_01__CT3__flash_duration`: contribution `-0.000493`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.000580`
- `lag_01__CT3__flash_duration`: contribution `-0.000493`
- `lag_01__CT1__flash_duration`: contribution `-0.000480`
- `lag_01__CT4__flash_duration`: contribution `-0.000475`
- `lag_01__CT_flash_duration_sum`: contribution `-0.000396`

### tick `10214`, seconds `70.50`, LSTM delta `+0.0182`

Top all feature movements:
- `lag_00__CT_flashed_players`: contribution `+0.000869`
- `lag_04__T_shots_fired_sum`: contribution `+0.000653`
- `lag_00__CT_flash_duration_sum`: contribution `+0.000560`
- `lag_00__T3__flash_duration`: contribution `+0.000542`
- `lag_00__CT5__flash_duration`: contribution `+0.000512`

Top utility-only movements:
- `lag_00__CT_flash_duration_sum`: contribution `+0.000560`
- `lag_00__T3__flash_duration`: contribution `+0.000542`
- `lag_00__CT5__flash_duration`: contribution `+0.000512`
- `lag_00__CT3__flash_duration`: contribution `+0.000463`
- `lag_00__CT1__flash_duration`: contribution `+0.000452`

### tick `10086`, seconds `68.50`, LSTM delta `-0.0172`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.000790`
- `lag_14__T_place_PALACEALLEY`: contribution `-0.000722`
- `lag_00__T_kills_last_3s`: contribution `-0.000587`
- `lag_08__CT2__duck_amount`: contribution `-0.000521`
- `lag_09__CT2__duck_amount`: contribution `-0.000485`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5766`, seconds `1.00`, LSTM delta `-0.0055`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000717`
- `lag_02__T_place_TSPAWN`: contribution `-0.000606`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000247`
- `lag_02__armor_diff`: contribution `-0.000228`
- `lag_02__CT_closest_enemy_dist`: contribution `-0.000220`

Top utility-only movements:
- `lag_00__T4__smoke`: contribution `-0.000168`
- `lag_02__smoke_inv_diff`: contribution `-0.000156`
- `lag_02__utility_inv_diff`: contribution `-0.000101`
