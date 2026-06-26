# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `12830`, seconds `7.50`, LSTM `0.8859`, delta `+0.0352`
- tick `17598`, seconds `82.00`, LSTM `0.9736`, delta `+0.0257`
- tick `12382`, seconds `0.50`, LSTM `0.9051`, delta `+0.0256`
- tick `14398`, seconds `32.00`, LSTM `0.9214`, delta `-0.0223`
- tick `13598`, seconds `19.50`, LSTM `0.9376`, delta `-0.0199`
- tick `12510`, seconds `2.50`, LSTM `0.8523`, delta `-0.0198`
- tick `12990`, seconds `10.00`, LSTM `0.9119`, delta `+0.0197`
- tick `15198`, seconds `44.50`, LSTM `0.9592`, delta `+0.0178`
- tick `13022`, seconds `10.50`, LSTM `0.9291`, delta `+0.0172`
- tick `14846`, seconds `39.00`, LSTM `0.9471`, delta `+0.0162`

## Top 15 local ridge features

- `lag_00__T_place_DECK`: coefficient `0.000586`, |coef| `0.000586`
- `lag_00__T_flashes_last_5s`: coefficient `-0.000561`, |coef| `0.000561`
- `lag_00__T_place_KITCHEN`: coefficient `0.000549`, |coef| `0.000549`
- `lag_00__CT_kills_last_3s`: coefficient `0.000495`, |coef| `0.000495`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000471`, |coef| `0.000471`
- `lag_00__kill_diff_last_3s`: coefficient `0.000431`, |coef| `0.000431`
- `lag_00__damage_diff_last_5s`: coefficient `0.000415`, |coef| `0.000415`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000413`, |coef| `0.000413`
- `lag_04__CT_place_RUINS`: coefficient `0.000412`, |coef| `0.000412`
- `lag_08__T_place_SECONDMID`: coefficient `0.000375`, |coef| `0.000375`
- `lag_01__CT_place_LIBRARY`: coefficient `-0.000366`, |coef| `0.000366`
- `lag_05__CT_place_RUINS`: coefficient `0.000366`, |coef| `0.000366`
- `lag_00__CT_damage_last_5s`: coefficient `0.000364`, |coef| `0.000364`
- `lag_01__CT_place_CTSPAWN`: coefficient `0.000344`, |coef| `0.000344`
- `lag_00__CT_place_LIBRARY`: coefficient `-0.000342`, |coef| `0.000342`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.000561` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000248` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000206` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000205` (raises CT win probability)
- `lag_09__T_flashes_last_5s`: coefficient `-0.000185` (lowers CT win probability)
- `lag_01__CT_smoke_inv`: coefficient `0.000172` (raises CT win probability)
- `lag_01__CT1__molly`: coefficient `0.000168` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.000168` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `-0.000168` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000161` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_DECK`: coefficient `0.000586` (raises CT win probability)
- `lag_00__T_place_KITCHEN`: coefficient `0.000549` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000495` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000471` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000431` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000415` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.000413` (raises CT win probability)
- `lag_04__CT_place_RUINS`: coefficient `0.000412` (raises CT win probability)
- `lag_08__T_place_SECONDMID`: coefficient `0.000375` (raises CT win probability)
- `lag_01__CT_place_LIBRARY`: coefficient `-0.000366` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `12830`, seconds `7.50`, LSTM delta `+0.0352`

Top all feature movements:
- `lag_00__T_place_UPSTAIRS`: contribution `+0.006963`
- `lag_00__T_flashes_last_5s`: contribution `+0.005080`
- `lag_04__CT_place_RUINS`: contribution `+0.001441`
- `lag_05__T_place_LOWERMID`: contribution `+0.001414`
- `lag_10__T_flashes_last_5s`: contribution `+0.001335`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.005080`
- `lag_10__T_flashes_last_5s`: contribution `+0.001335`

### tick `17598`, seconds `82.00`, LSTM delta `+0.0257`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.001637`
- `lag_00__CT_kills_last_3s`: contribution `+0.001429`
- `lag_05__CT_place_RUINS`: contribution `+0.001278`
- `lag_08__T_place_SECONDMID`: contribution `+0.001227`
- `lag_00__kill_diff_last_3s`: contribution `+0.001038`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `12382`, seconds `0.50`, LSTM delta `+0.0256`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `+0.001647`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.001064`
- `lag_01__T_place_TSPAWN`: contribution `+0.001031`
- `lag_01__T_closest_enemy_dist`: contribution `+0.000888`
- `lag_01__centroid_distance_xy`: contribution `+0.000879`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `+0.000592`
- `lag_01__smoke_inv_diff`: contribution `+0.000528`
- `lag_01__molly_inv_diff`: contribution `+0.000430`
- `lag_01__CT_smoke_inv`: contribution `+0.000398`
- `lag_01__CT1__molly`: contribution `+0.000334`

### tick `14398`, seconds `32.00`, LSTM delta `-0.0223`

Top all feature movements:
- `lag_00__T_place_DECK`: contribution `-0.014213`
- `lag_00__T_place_SECONDMID`: contribution `-0.000927`
- `lag_00__CT3__is_walking`: contribution `-0.000606`
- `lag_13__CT4__duck_amount`: contribution `-0.000604`
- `lag_07__CT4__duck_amount`: contribution `-0.000564`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13598`, seconds `19.50`, LSTM delta `-0.0199`

Top all feature movements:
- `lag_00__T_place_KITCHEN`: contribution `-0.017548`
- `lag_00__T_place_DECK`: contribution `+0.014213`
- `lag_03__T_place_KITCHEN`: contribution `-0.006242`
- `lag_03__T_place_UPSTAIRS`: contribution `-0.003000`
- `lag_11__CT_place_QUAD`: contribution `-0.001233`

Top utility-only movements:
- No utility movement among the top local contributors.
