# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `30332`, seconds `104.00`, LSTM `0.2248`, delta `-0.5337`
- tick `30300`, seconds `103.50`, LSTM `0.7585`, delta `+0.4676`
- tick `29948`, seconds `98.00`, LSTM `0.5169`, delta `-0.2444`
- tick `29916`, seconds `97.50`, LSTM `0.7613`, delta `+0.2298`
- tick `30140`, seconds `101.00`, LSTM `0.2350`, delta `-0.1994`
- tick `30172`, seconds `101.50`, LSTM `0.3953`, delta `+0.1603`
- tick `30268`, seconds `103.00`, LSTM `0.2909`, delta `-0.0913`
- tick `30204`, seconds `102.00`, LSTM `0.3153`, delta `-0.0800`
- tick `29052`, seconds `84.00`, LSTM `0.4533`, delta `-0.0773`
- tick `30236`, seconds `102.50`, LSTM `0.3822`, delta `+0.0669`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003991`, |coef| `0.003991`
- `lag_01__T_place_STAIRS`: coefficient `0.003348`, |coef| `0.003348`
- `lag_00__T_place_STAIRS`: coefficient `-0.003185`, |coef| `0.003185`
- `lag_11__T_bomb_zone_count`: coefficient `0.003022`, |coef| `0.003022`
- `lag_12__CT_shots_fired_sum`: coefficient `0.002770`, |coef| `0.002770`
- `lag_03__T_place_STAIRS`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_02__T_place_STAIRS`: coefficient `0.002206`, |coef| `0.002206`
- `lag_04__T_place_STAIRS`: coefficient `0.002156`, |coef| `0.002156`
- `lag_07__CT_shots_fired_sum`: coefficient `-0.002091`, |coef| `0.002091`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002031`, |coef| `0.002031`
- `lag_08__T_bomb_zone_count`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_12__T_place_STAIRS`: coefficient `-0.001951`, |coef| `0.001951`
- `lag_00__kill_diff_last_3s`: coefficient `0.001950`, |coef| `0.001950`
- `lag_04__CT_shots_fired_sum`: coefficient `0.001893`, |coef| `0.001893`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001799`, |coef| `0.001799`

## Top 10 utility ridge features

- `lag_06__CT4__flash_duration`: coefficient `-0.001519` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.000971` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `0.000969` (raises CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `0.000952` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000810` (lowers CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.000803` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000797` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000784` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.000781` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.003991` (raises CT win probability)
- `lag_01__T_place_STAIRS`: coefficient `0.003348` (raises CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `-0.003185` (lowers CT win probability)
- `lag_11__T_bomb_zone_count`: coefficient `0.003022` (raises CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `0.002770` (raises CT win probability)
- `lag_03__T_place_STAIRS`: coefficient `-0.002268` (lowers CT win probability)
- `lag_02__T_place_STAIRS`: coefficient `0.002206` (raises CT win probability)
- `lag_04__T_place_STAIRS`: coefficient `0.002156` (raises CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.002091` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002031` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `30332`, seconds `104.00`, LSTM delta `-0.5337`

Top all feature movements:
- `lag_01__T_place_STAIRS`: contribution `-0.064095`
- `lag_03__T_place_STAIRS`: contribution `-0.043418`
- `lag_13__T_place_STAIRS`: contribution `-0.030969`
- `lag_12__CT_shots_fired_sum`: contribution `-0.026941`
- `lag_11__T_bomb_zone_count`: contribution `-0.017589`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30300`, seconds `103.50`, LSTM delta `+0.4676`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.060969`
- `lag_02__T_place_STAIRS`: contribution `+0.042237`
- `lag_12__T_place_STAIRS`: contribution `+0.037352`
- `lag_11__T_bomb_zone_count`: contribution `+0.017589`
- `lag_12__CT_shots_fired_sum`: contribution `+0.017319`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.012030`

### tick `29948`, seconds `98.00`, LSTM delta `-0.2444`

Top all feature movements:
- `lag_01__T_place_STAIRS`: contribution `-0.064095`
- `lag_00__CT_shots_fired_sum`: contribution `-0.038819`
- `lag_05__T_place_STAIRS`: contribution `-0.020077`
- `lag_01__CT_shots_fired_sum`: contribution `-0.010145`
- `lag_14__CT4__flash_duration`: contribution `-0.007566`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `-0.007566`

### tick `29916`, seconds `97.50`, LSTM delta `+0.2298`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.060969`
- `lag_04__T_place_STAIRS`: contribution `+0.041279`
- `lag_00__CT_shots_fired_sum`: contribution `+0.024955`
- `lag_11__CT_place_JUNGLE`: contribution `+0.006032`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005636`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `+0.004940`
- `lag_08__CT4__flash_duration`: contribution `+0.004692`

### tick `30140`, seconds `101.00`, LSTM delta `-0.1994`

Top all feature movements:
- `lag_07__T_place_STAIRS`: contribution `-0.015741`
- `lag_07__CT_shots_fired_sum`: contribution `-0.013077`
- `lag_06__CT_shots_fired_sum`: contribution `-0.011535`
- `lag_06__T_bomb_zone_count`: contribution `-0.010475`
- `lag_00__kill_diff_last_3s`: contribution `-0.009387`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.006189`
