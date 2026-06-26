# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `11945`, seconds `7.00`, LSTM `0.1342`, delta `-0.0547`
- tick `11977`, seconds `7.50`, LSTM `0.0907`, delta `-0.0436`
- tick `11625`, seconds `2.00`, LSTM `0.0635`, delta `+0.0392`
- tick `11913`, seconds `6.50`, LSTM `0.1890`, delta `+0.0363`
- tick `11657`, seconds `2.50`, LSTM `0.0911`, delta `+0.0277`
- tick `11529`, seconds `0.50`, LSTM `0.0284`, delta `-0.0224`
- tick `13001`, seconds `23.50`, LSTM `0.0304`, delta `-0.0184`
- tick `13801`, seconds `36.00`, LSTM `0.0371`, delta `+0.0179`
- tick `11689`, seconds `3.00`, LSTM `0.1075`, delta `+0.0164`
- tick `11817`, seconds `5.00`, LSTM `0.1434`, delta `+0.0157`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000610`, |coef| `0.000610`
- `lag_01__CT_smokes_last_5s`: coefficient `0.000384`, |coef| `0.000384`
- `lag_03__CT_smokes_last_5s`: coefficient `0.000308`, |coef| `0.000308`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000278`, |coef| `0.000278`
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000275`, |coef| `0.000275`
- `lag_07__CT_smokes_last_5s`: coefficient `0.000272`, |coef| `0.000272`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000250`, |coef| `0.000250`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000249`, |coef| `0.000249`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000247`, |coef| `0.000247`
- `lag_01__CT_place_HELL`: coefficient `0.000223`, |coef| `0.000223`
- `lag_01__centroid_distance_xy`: coefficient `-0.000220`, |coef| `0.000220`
- `lag_04__CT_place_HELL`: coefficient `-0.000219`, |coef| `0.000219`
- `lag_00__T_velocity_mean`: coefficient `-0.000212`, |coef| `0.000212`
- `lag_07__CT_place_GARAGE`: coefficient `0.000211`, |coef| `0.000211`
- `lag_06__CT_smokes_last_5s`: coefficient `0.000201`, |coef| `0.000201`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000610` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000384` (raises CT win probability)
- `lag_03__CT_smokes_last_5s`: coefficient `0.000308` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000275` (lowers CT win probability)
- `lag_07__CT_smokes_last_5s`: coefficient `0.000272` (raises CT win probability)
- `lag_06__CT_smokes_last_5s`: coefficient `0.000201` (raises CT win probability)
- `lag_09__CT_smokes_last_5s`: coefficient `0.000193` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000183` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000171` (raises CT win probability)
- `lag_05__CT_smokes_last_5s`: coefficient `0.000148` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000278` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000250` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000249` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000247` (lowers CT win probability)
- `lag_01__CT_place_HELL`: coefficient `0.000223` (raises CT win probability)
- `lag_01__centroid_distance_xy`: coefficient `-0.000220` (lowers CT win probability)
- `lag_04__CT_place_HELL`: coefficient `-0.000219` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000212` (lowers CT win probability)
- `lag_07__CT_place_GARAGE`: coefficient `0.000211` (raises CT win probability)
- `lag_03__CT_place_HELL`: coefficient `0.000197` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `11945`, seconds `7.00`, LSTM delta `-0.0547`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.021083`
- `lag_10__CT_smokes_last_5s`: contribution `-0.009499`
- `lag_03__CT_smokes_last_5s`: contribution `-0.005324`
- `lag_09__CT_smokes_last_5s`: contribution `+0.003330`
- `lag_04__CT_place_HELL`: contribution `-0.002378`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `-0.021083`
- `lag_10__CT_smokes_last_5s`: contribution `-0.009499`
- `lag_03__CT_smokes_last_5s`: contribution `-0.005324`
- `lag_09__CT_smokes_last_5s`: contribution `+0.003330`
- `lag_13__CT_smokes_last_5s`: contribution `-0.000582`

### tick `11977`, seconds `7.50`, LSTM delta `-0.0436`

Top all feature movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.013279`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010542`
- `lag_10__CT_smokes_last_5s`: contribution `-0.004750`
- `lag_11__CT_smokes_last_5s`: contribution `-0.004346`
- `lag_14__CT_smokes_last_5s`: contribution `+0.001590`

Top utility-only movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.013279`
- `lag_00__CT_smokes_last_5s`: contribution `-0.010542`
- `lag_10__CT_smokes_last_5s`: contribution `-0.004750`
- `lag_11__CT_smokes_last_5s`: contribution `-0.004346`
- `lag_14__CT_smokes_last_5s`: contribution `+0.001590`

### tick `11625`, seconds `2.00`, LSTM delta `+0.0392`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.021083`
- `lag_03__CT_smokes_last_5s`: contribution `+0.005324`
- `lag_04__CT_place_CTSPAWN`: contribution `+0.000458`
- `lag_04__CT_closest_enemy_dist`: contribution `+0.000416`
- `lag_04__T_closest_enemy_dist`: contribution `+0.000358`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.021083`
- `lag_03__CT_smokes_last_5s`: contribution `+0.005324`
- `lag_01__T3__smoke`: contribution `+0.000275`
- `lag_04__utility_inv_diff`: contribution `+0.000174`
- `lag_04__molly_inv_diff`: contribution `+0.000157`

### tick `11913`, seconds `6.50`, LSTM delta `+0.0363`

Top all feature movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.006660`
- `lag_03__CT_place_HELL`: contribution `+0.002138`
- `lag_12__CT_smokes_last_5s`: contribution `+0.001733`
- `lag_08__CT_smokes_last_5s`: contribution `+0.001649`
- `lag_02__CT_smokes_last_5s`: contribution `+0.001286`

Top utility-only movements:
- `lag_09__CT_smokes_last_5s`: contribution `+0.006660`
- `lag_12__CT_smokes_last_5s`: contribution `+0.001733`
- `lag_08__CT_smokes_last_5s`: contribution `+0.001649`
- `lag_02__CT_smokes_last_5s`: contribution `+0.001286`
- `lag_13__utility_inv_diff`: contribution `+0.000278`

### tick `11657`, seconds `2.50`, LSTM delta `+0.0277`

Top all feature movements:
- `lag_01__CT_smokes_last_5s`: contribution `+0.013279`
- `lag_00__CT_smokes_last_5s`: contribution `+0.010542`
- `lag_04__CT_smokes_last_5s`: contribution `+0.000541`
- `lag_05__T_place_TSPAWN`: contribution `+0.000355`
- `lag_01__bomb_events_last_5s`: contribution `+0.000266`

Top utility-only movements:
- `lag_01__CT_smokes_last_5s`: contribution `+0.013279`
- `lag_00__CT_smokes_last_5s`: contribution `+0.010542`
- `lag_04__CT_smokes_last_5s`: contribution `+0.000541`
- `lag_01__smoke_inv_diff`: contribution `+0.000219`
- `lag_01__CT4__smoke`: contribution `+0.000153`
