# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `23604`, seconds `0.50`, LSTM `0.0222`, delta `-0.0383`
- tick `24244`, seconds `10.50`, LSTM `0.0144`, delta `-0.0154`
- tick `24212`, seconds `10.00`, LSTM `0.0298`, delta `-0.0091`
- tick `24180`, seconds `9.50`, LSTM `0.0389`, delta `+0.0082`
- tick `23636`, seconds `1.00`, LSTM `0.0164`, delta `-0.0058`
- tick `23796`, seconds `3.50`, LSTM `0.0235`, delta `+0.0045`
- tick `24916`, seconds `21.00`, LSTM `0.0100`, delta `+0.0042`
- tick `24148`, seconds `9.00`, LSTM `0.0307`, delta `+0.0040`
- tick `24020`, seconds `7.00`, LSTM `0.0226`, delta `-0.0038`
- tick `24116`, seconds `8.50`, LSTM `0.0267`, delta `+0.0036`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000365`, |coef| `0.000365`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000342`, |coef| `0.000342`
- `lag_00__T_velocity_mean`: coefficient `-0.000265`, |coef| `0.000265`
- `lag_01__CT_place_HOLE`: coefficient `0.000246`, |coef| `0.000246`
- `lag_01__utility_inv_diff`: coefficient `0.000234`, |coef| `0.000234`
- `lag_01__armor_diff`: coefficient `0.000230`, |coef| `0.000230`
- `lag_01__smoke_inv_diff`: coefficient `0.000228`, |coef| `0.000228`
- `lag_00__CT_velocity_mean`: coefficient `-0.000226`, |coef| `0.000226`
- `lag_01__molly_inv_diff`: coefficient `0.000218`, |coef| `0.000218`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000204`, |coef| `0.000204`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000197`, |coef| `0.000197`
- `lag_01__CT_armor_sum`: coefficient `0.000192`, |coef| `0.000192`
- `lag_01__T5__utility_total`: coefficient `-0.000189`, |coef| `0.000189`
- `lag_01__centroid_distance_xy`: coefficient `-0.000187`, |coef| `0.000187`
- `lag_01__T_smoke_inv`: coefficient `-0.000184`, |coef| `0.000184`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000234` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000228` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000218` (raises CT win probability)
- `lag_01__T5__utility_total`: coefficient `-0.000189` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000184` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000180` (raises CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000175` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000174` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000167` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000145` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000365` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000342` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000265` (lowers CT win probability)
- `lag_01__CT_place_HOLE`: coefficient `0.000246` (raises CT win probability)
- `lag_01__armor_diff`: coefficient `0.000230` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000226` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000204` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000197` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000192` (raises CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000187` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23604`, seconds `0.50`, LSTM delta `-0.0383`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001746`
- `lag_01__T_place_TSPAWN`: contribution `-0.001516`
- `lag_00__T_velocity_mean`: contribution `-0.000794`
- `lag_00__CT_velocity_mean`: contribution `-0.000741`
- `lag_01__utility_inv_diff`: contribution `-0.000668`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000668`
- `lag_01__molly_inv_diff`: contribution `-0.000609`
- `lag_01__smoke_inv_diff`: contribution `-0.000580`
- `lag_01__T5__utility_total`: contribution `-0.000437`
- `lag_01__T_smoke_inv`: contribution `-0.000420`

### tick `24244`, seconds `10.50`, LSTM delta `-0.0154`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `-0.002747`
- `lag_03__CT_place_HOLE`: contribution `-0.001319`
- `lag_14__CT_place_MIDDOORS`: contribution `-0.000605`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.000575`
- `lag_02__CT_place_BDOORS`: contribution `-0.000573`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.000575`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.000323`

### tick `24212`, seconds `10.00`, LSTM delta `-0.0091`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.001608`
- `lag_04__CT_place_BDOORS`: contribution `-0.000661`
- `lag_02__CT_place_HOLE`: contribution `-0.000631`
- `lag_02__CT_place_BDOORS`: contribution `-0.000573`
- `lag_07__CT_place_BDOORS`: contribution `-0.000507`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.000271`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.000138`

### tick `24180`, seconds `9.50`, LSTM delta `+0.0082`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.002747`
- `lag_04__CT_place_BDOORS`: contribution `+0.000661`
- `lag_03__CT_place_BDOORS`: contribution `-0.000457`
- `lag_05__CT_place_BDOORS`: contribution `+0.000443`
- `lag_00__T_velocity_mean`: contribution `+0.000393`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.000256`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.000235`

### tick `23636`, seconds `1.00`, LSTM delta `-0.0058`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.000624`
- `lag_02__T_place_TSPAWN`: contribution `-0.000576`
- `lag_02__utility_inv_diff`: contribution `-0.000258`
- `lag_02__armor_diff`: contribution `-0.000256`
- `lag_02__smoke_inv_diff`: contribution `-0.000231`

Top utility-only movements:
- `lag_02__utility_inv_diff`: contribution `-0.000258`
- `lag_02__smoke_inv_diff`: contribution `-0.000231`
- `lag_02__molly_inv_diff`: contribution `-0.000226`
- `lag_02__T5__utility_total`: contribution `-0.000169`
- `lag_02__T_smoke_inv`: contribution `-0.000166`
