# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `101862`, seconds `93.00`, LSTM `0.7720`, delta `+0.2925`
- tick `101734`, seconds `91.00`, LSTM `0.6024`, delta `-0.1547`
- tick `100390`, seconds `70.00`, LSTM `0.8989`, delta `+0.1544`
- tick `101702`, seconds `90.50`, LSTM `0.7571`, delta `-0.0867`
- tick `98310`, seconds `37.50`, LSTM `0.8300`, delta `+0.0751`
- tick `99590`, seconds `57.50`, LSTM `0.7147`, delta `-0.0658`
- tick `101798`, seconds `92.00`, LSTM `0.5188`, delta `-0.0521`
- tick `101894`, seconds `93.50`, LSTM `0.7273`, delta `-0.0446`
- tick `100294`, seconds `68.50`, LSTM `0.7515`, delta `+0.0412`
- tick `101830`, seconds `92.50`, LSTM `0.4795`, delta `-0.0393`

## Top 15 local ridge features

- `lag_00__T_bomb_zone_count`: coefficient `-0.001744`, |coef| `0.001744`
- `lag_07__CT_place_BACKOFA`: coefficient `-0.001638`, |coef| `0.001638`
- `lag_01__CT_place_STORAGEROOM`: coefficient `-0.001449`, |coef| `0.001449`
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001382`, |coef| `0.001382`
- `lag_03__T_duck_amount_mean`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_15__T_place_RESTROOM`: coefficient `0.001332`, |coef| `0.001332`
- `lag_03__CT_place_STORAGEROOM`: coefficient `-0.001318`, |coef| `0.001318`
- `lag_06__CT_place_STORAGEROOM`: coefficient `-0.001306`, |coef| `0.001306`
- `lag_10__CT_place_STAIRS`: coefficient `-0.001298`, |coef| `0.001298`
- `lag_01__CT_place_LOBBY`: coefficient `0.001205`, |coef| `0.001205`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001203`, |coef| `0.001203`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001195`, |coef| `0.001195`
- `lag_00__T_place_RESTROOM`: coefficient `-0.001175`, |coef| `0.001175`
- `lag_11__CT_place_STAIRS`: coefficient `-0.001120`, |coef| `0.001120`
- `lag_05__CT_place_BACKOFA`: coefficient `-0.001115`, |coef| `0.001115`

## Top 10 utility ridge features

- `lag_09__CT5__flash_duration`: coefficient `-0.001013` (lowers CT win probability)
- `lag_12__T5__flash_duration`: coefficient `-0.000834` (lowers CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000619` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000600` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000574` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.000559` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.000554` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.000551` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000532` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000515` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_bomb_zone_count`: coefficient `-0.001744` (lowers CT win probability)
- `lag_07__CT_place_BACKOFA`: coefficient `-0.001638` (lowers CT win probability)
- `lag_01__CT_place_STORAGEROOM`: coefficient `-0.001449` (lowers CT win probability)
- `lag_09__CT_place_STORAGEROOM`: coefficient `-0.001382` (lowers CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.001378` (lowers CT win probability)
- `lag_15__T_place_RESTROOM`: coefficient `0.001332` (raises CT win probability)
- `lag_03__CT_place_STORAGEROOM`: coefficient `-0.001318` (lowers CT win probability)
- `lag_06__CT_place_STORAGEROOM`: coefficient `-0.001306` (lowers CT win probability)
- `lag_10__CT_place_STAIRS`: coefficient `-0.001298` (lowers CT win probability)
- `lag_01__CT_place_LOBBY`: coefficient `0.001205` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `101862`, seconds `93.00`, LSTM delta `+0.2925`

Top all feature movements:
- `lag_01__CT_place_STORAGEROOM`: contribution `+0.030998`
- `lag_07__CT_place_BACKOFA`: contribution `+0.015820`
- `lag_07__CT_place_STORAGEROOM`: contribution `+0.013226`
- `lag_05__CT_place_STORAGEROOM`: contribution `+0.011726`
- `lag_05__CT_place_BACKOFA`: contribution `+0.010770`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `+0.005739`

### tick `101734`, seconds `91.00`, LSTM delta `-0.1547`

Top all feature movements:
- `lag_01__CT_place_STORAGEROOM`: contribution `-0.030998`
- `lag_03__CT_place_STORAGEROOM`: contribution `-0.028186`
- `lag_07__CT_place_BACKOFA`: contribution `-0.015820`
- `lag_05__CT_place_BACKOFA`: contribution `-0.010770`
- `lag_10__CT_place_STAIRS`: contribution `-0.010101`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.003124`
- `lag_15__CT5__flash_duration`: contribution `-0.002912`

### tick `100390`, seconds `70.00`, LSTM delta `+0.1544`

Top all feature movements:
- `lag_15__T_place_RESTROOM`: contribution `+0.025699`
- `lag_03__T_place_RESTROOM`: contribution `+0.017809`
- `lag_08__T_place_RESTROOM`: contribution `+0.013476`
- `lag_09__CT_place_BRIDGE`: contribution `+0.010206`
- `lag_06__CT_place_BRIDGE`: contribution `+0.005788`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101702`, seconds `90.50`, LSTM delta `-0.0867`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.010154`
- `lag_00__CT_place_STORAGEROOM`: contribution `-0.009120`
- `lag_09__CT_place_STAIRS`: contribution `-0.008207`
- `lag_00__CT_place_BACKOFA`: contribution `-0.006992`
- `lag_06__CT_place_BACKOFA`: contribution `-0.006145`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `-0.002302`

### tick `98310`, seconds `37.50`, LSTM delta `+0.0751`

Top all feature movements:
- `lag_09__CT_place_STORAGEROOM`: contribution `+0.029572`
- `lag_05__CT_place_BACKOFA`: contribution `+0.010770`
- `lag_09__CT_place_BACKOFA`: contribution `+0.009786`
- `lag_12__CT_place_STORAGEROOM`: contribution `+0.008484`
- `lag_00__CT_kills_last_3s`: contribution `+0.002968`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.001618`
- `lag_10__T4__flash_duration`: contribution `+0.001399`
- `lag_10__T1__flash_duration`: contribution `+0.001288`
- `lag_10__CT4__flash_duration`: contribution `-0.001262`
