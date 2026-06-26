# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `16520`, seconds `47.00`, LSTM `0.1665`, delta `+0.1074`
- tick `15496`, seconds `31.00`, LSTM `0.0313`, delta `-0.0646`
- tick `16072`, seconds `40.00`, LSTM `0.0273`, delta `-0.0601`
- tick `13544`, seconds `0.50`, LSTM `0.0680`, delta `-0.0562`
- tick `16008`, seconds `39.00`, LSTM `0.0818`, delta `+0.0530`
- tick `14632`, seconds `17.50`, LSTM `0.1177`, delta `-0.0379`
- tick `16616`, seconds `48.50`, LSTM `0.2549`, delta `+0.0371`
- tick `17480`, seconds `62.00`, LSTM `0.3553`, delta `+0.0362`
- tick `17064`, seconds `55.50`, LSTM `0.2691`, delta `-0.0357`
- tick `15976`, seconds `38.50`, LSTM `0.0288`, delta `-0.0335`

## Top 15 local ridge features

- `lag_00__T_duck_amount_mean`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_00__T1__duck_amount`: coefficient `-0.001435`, |coef| `0.001435`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001398`, |coef| `0.001398`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_09__T_duck_amount_mean`: coefficient `0.001173`, |coef| `0.001173`
- `lag_06__CT_duck_amount_mean`: coefficient `0.001163`, |coef| `0.001163`
- `lag_12__T_shots_fired_sum`: coefficient `-0.001064`, |coef| `0.001064`
- `lag_00__kill_diff_last_3s`: coefficient `0.001020`, |coef| `0.001020`
- `lag_03__T_duck_amount_mean`: coefficient `-0.001005`, |coef| `0.001005`
- `lag_15__CT_velocity_mean`: coefficient `-0.000990`, |coef| `0.000990`
- `lag_00__CT3__is_walking`: coefficient `-0.000965`, |coef| `0.000965`
- `lag_11__kill_diff_last_3s`: coefficient `0.000859`, |coef| `0.000859`
- `lag_03__T1__duck_amount`: coefficient `-0.000846`, |coef| `0.000846`
- `lag_07__CT_velocity_mean`: coefficient `0.000827`, |coef| `0.000827`
- `lag_10__T_duck_amount_mean`: coefficient `0.000822`, |coef| `0.000822`

## Top 10 utility ridge features

- `lag_15__CT_smokes_last_5s`: coefficient `-0.000784` (lowers CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `0.000727` (raises CT win probability)
- `lag_06__CT_smokes_last_5s`: coefficient `0.000718` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.000717` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000650` (lowers CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `-0.000577` (lowers CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000576` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000563` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000554` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000530` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_duck_amount_mean`: coefficient `-0.002268` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001435` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001398` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001297` (lowers CT win probability)
- `lag_09__T_duck_amount_mean`: coefficient `0.001173` (raises CT win probability)
- `lag_06__CT_duck_amount_mean`: coefficient `0.001163` (raises CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `-0.001064` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001020` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.001005` (lowers CT win probability)
- `lag_15__CT_velocity_mean`: coefficient `-0.000990` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16520`, seconds `47.00`, LSTM delta `+0.1074`

Top all feature movements:
- `lag_12__T_shots_fired_sum`: contribution `+0.007978`
- `lag_13__CT_place_BALCONY`: contribution `+0.005268`
- `lag_07__T3__flash_duration`: contribution `+0.004271`
- `lag_12__T1__shots_fired`: contribution `+0.003267`
- `lag_15__CT_place_BALCONY`: contribution `+0.003052`

Top utility-only movements:
- `lag_07__T3__flash_duration`: contribution `+0.004271`
- `lag_09__T_A_site_active_infernos`: contribution `+0.001934`

### tick `15496`, seconds `31.00`, LSTM delta `-0.0646`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.005835`
- `lag_00__T1__duck_amount`: contribution `-0.005620`
- `lag_10__T5__shots_fired`: contribution `-0.003115`
- `lag_00__T_duck_amount_mean`: contribution `-0.002638`
- `lag_00__kill_diff_last_3s`: contribution `-0.002454`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16072`, seconds `40.00`, LSTM delta `-0.0601`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.004863`
- `lag_00__kill_diff_last_3s`: contribution `-0.002454`
- `lag_00__T_kills_last_3s`: contribution `-0.002442`
- `lag_04__CT_flashed_players`: contribution `-0.002414`
- `lag_00__T_damage_last_5s`: contribution `-0.002298`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `-0.002197`
- `lag_09__T_A_site_active_infernos`: contribution `-0.001934`

### tick `13544`, seconds `0.50`, LSTM delta `-0.0562`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003624`
- `lag_01__T_place_TSPAWN`: contribution `-0.003166`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002631`
- `lag_01__centroid_distance_xy`: contribution `-0.002506`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002392`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001458`
- `lag_01__T3__utility_total`: contribution `-0.001345`
- `lag_01__T3__flash`: contribution `-0.001288`
- `lag_01__molly_inv_diff`: contribution `-0.001261`
- `lag_01__T_utility_inv`: contribution `-0.001031`

### tick `16008`, seconds `39.00`, LSTM delta `+0.0530`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.004863`
- `lag_00__kill_diff_last_3s`: contribution `+0.002454`
- `lag_00__CT_duck_amount_mean`: contribution `+0.002033`
- `lag_01__CT_flashed_players`: contribution `+0.001964`
- `lag_10__T_kills_last_3s`: contribution `+0.001921`

Top utility-only movements:
- No utility movement among the top local contributors.
