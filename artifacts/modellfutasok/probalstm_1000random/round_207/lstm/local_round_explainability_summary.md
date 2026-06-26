# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-b8-bo3--nzkpOWiS4qFgkFOwM8Hun/legacy-vs-b8-m2-ancient.csv`
- round_num: `19`

## Largest probability jumps

- tick `142890`, seconds `45.50`, LSTM `0.7138`, delta `+0.2968`
- tick `141994`, seconds `31.50`, LSTM `0.3829`, delta `-0.2474`
- tick `142570`, seconds `40.50`, LSTM `0.4033`, delta `+0.1926`
- tick `141098`, seconds `17.50`, LSTM `0.7719`, delta `+0.1913`
- tick `141450`, seconds `23.00`, LSTM `0.5596`, delta `-0.1567`
- tick `143658`, seconds `57.50`, LSTM `0.6622`, delta `-0.1503`
- tick `141162`, seconds `18.50`, LSTM `0.6426`, delta `-0.1262`
- tick `142218`, seconds `35.00`, LSTM `0.2300`, delta `-0.1240`
- tick `140682`, seconds `11.00`, LSTM `0.5345`, delta `+0.0926`
- tick `142826`, seconds `44.50`, LSTM `0.4466`, delta `-0.0805`

## Top 15 local ridge features

- `lag_06__CT_place_TSIDELOWER`: coefficient `0.003629`, |coef| `0.003629`
- `lag_10__CT_place_TUNNEL`: coefficient `-0.003455`, |coef| `0.003455`
- `lag_00__kill_diff_last_3s`: coefficient `0.003375`, |coef| `0.003375`
- `lag_02__CT_place_TSIDELOWER`: coefficient `-0.002878`, |coef| `0.002878`
- `lag_09__T1__flash_duration`: coefficient `-0.002810`, |coef| `0.002810`
- `lag_00__CT3__flash_duration`: coefficient `0.002564`, |coef| `0.002564`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002369`, |coef| `0.002369`
- `lag_09__T_flash_duration_sum`: coefficient `-0.002363`, |coef| `0.002363`
- `lag_00__T_kills_last_3s`: coefficient `-0.002354`, |coef| `0.002354`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `0.002229`, |coef| `0.002229`
- `lag_10__CT_place_TSPAWN`: coefficient `0.002222`, |coef| `0.002222`
- `lag_09__T_flashed_players`: coefficient `-0.002106`, |coef| `0.002106`
- `lag_12__CT_place_WATER`: coefficient `-0.001998`, |coef| `0.001998`
- `lag_01__T_bomb_zone_count`: coefficient `-0.001984`, |coef| `0.001984`
- `lag_11__T3__duck_amount`: coefficient `0.001967`, |coef| `0.001967`

## Top 10 utility ridge features

- `lag_09__T1__flash_duration`: coefficient `-0.002810` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.002564` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.002363` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `-0.001917` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001599` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.001298` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.001292` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001265` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `0.001246` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.001229` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_place_TSIDELOWER`: coefficient `0.003629` (raises CT win probability)
- `lag_10__CT_place_TUNNEL`: coefficient `-0.003455` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003375` (raises CT win probability)
- `lag_02__CT_place_TSIDELOWER`: coefficient `-0.002878` (lowers CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002369` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002354` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `0.002229` (raises CT win probability)
- `lag_10__CT_place_TSPAWN`: coefficient `0.002222` (raises CT win probability)
- `lag_09__T_flashed_players`: coefficient `-0.002106` (lowers CT win probability)
- `lag_12__CT_place_WATER`: coefficient `-0.001998` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `142890`, seconds `45.50`, LSTM delta `+0.2968`

Top all feature movements:
- `lag_06__CT_place_TSIDELOWER`: contribution `+0.049293`
- `lag_02__CT_place_TSIDELOWER`: contribution `+0.039092`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.010877`
- `lag_00__kill_diff_last_3s`: contribution `+0.008123`
- `lag_05__CT4__flash_duration`: contribution `+0.007789`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.007789`

### tick `141994`, seconds `31.50`, LSTM delta `-0.2474`

Top all feature movements:
- `lag_10__CT_place_TUNNEL`: contribution `-0.055500`
- `lag_00__CT3__flash_duration`: contribution `-0.020493`
- `lag_10__CT_place_TSPAWN`: contribution `-0.016633`
- `lag_11__CT3__flash_duration`: contribution `-0.010373`
- `lag_15__T_place_WATER`: contribution `-0.010244`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.020493`
- `lag_11__CT3__flash_duration`: contribution `-0.010373`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004525`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.004085`
- `lag_01__CT_B_site_active_infernos`: contribution `-0.004067`

### tick `142570`, seconds `40.50`, LSTM delta `+0.1926`

Top all feature movements:
- `lag_12__CT_place_WATER`: contribution `+0.012144`
- `lag_01__T_bomb_zone_count`: contribution `+0.011549`
- `lag_00__kill_diff_last_3s`: contribution `+0.008123`
- `lag_09__T3__flash_duration`: contribution `+0.007593`
- `lag_11__T2__duck_amount`: contribution `+0.005830`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `+0.007593`
- `lag_09__T_flash_duration_sum`: contribution `+0.003901`

### tick `141098`, seconds `17.50`, LSTM delta `+0.1913`

Top all feature movements:
- `lag_04__CT4__shots_fired`: contribution `+0.012671`
- `lag_10__CT_shots_fired_sum`: contribution `+0.011011`
- `lag_04__CT_shots_fired_sum`: contribution `+0.010007`
- `lag_00__kill_diff_last_3s`: contribution `+0.008123`
- `lag_00__T5__flash_duration`: contribution `+0.007267`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.007267`
- `lag_15__CT1__flash_duration`: contribution `+0.006934`
- `lag_07__T5__flash_duration`: contribution `+0.006746`
- `lag_12__CT4__flash_duration`: contribution `+0.006300`
- `lag_07__CT2__flash_duration`: contribution `+0.006164`

### tick `141450`, seconds `23.00`, LSTM delta `-0.1567`

Top all feature movements:
- `lag_15__CT_shots_fired_sum`: contribution `-0.013965`
- `lag_15__CT4__shots_fired`: contribution `-0.008280`
- `lag_00__kill_diff_last_3s`: contribution `-0.008123`
- `lag_12__CT4__flash_duration`: contribution `-0.007822`
- `lag_00__T_kills_last_3s`: contribution `-0.007458`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.007822`
- `lag_11__T5__flash_duration`: contribution `-0.006654`
- `lag_12__CT_flash_duration_sum`: contribution `-0.003110`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.002335`
