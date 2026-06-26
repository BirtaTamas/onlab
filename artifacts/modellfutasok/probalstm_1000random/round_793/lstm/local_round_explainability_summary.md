# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `3`

## Largest probability jumps

- tick `16405`, seconds `92.50`, LSTM `0.1317`, delta `-0.1851`
- tick `16277`, seconds `90.50`, LSTM `0.3625`, delta `-0.1789`
- tick `13397`, seconds `45.50`, LSTM `0.2304`, delta `-0.1632`
- tick `15285`, seconds `75.00`, LSTM `0.4758`, delta `+0.1454`
- tick `16437`, seconds `93.00`, LSTM `0.2738`, delta `+0.1421`
- tick `16469`, seconds `93.50`, LSTM `0.1564`, delta `-0.1174`
- tick `15509`, seconds `78.50`, LSTM `0.4860`, delta `-0.0774`
- tick `15541`, seconds `79.00`, LSTM `0.4100`, delta `-0.0761`
- tick `15861`, seconds `84.00`, LSTM `0.4061`, delta `+0.0702`
- tick `14901`, seconds `69.00`, LSTM `0.3591`, delta `+0.0669`

## Top 15 local ridge features

- `lag_00__CT_place_BRIDGE`: coefficient `0.002387`, |coef| `0.002387`
- `lag_03__T_place_BRICKS`: coefficient `-0.001931`, |coef| `0.001931`
- `lag_05__CT_place_OUTSIDELONG`: coefficient `-0.001677`, |coef| `0.001677`
- `lag_00__CT1__duck_amount`: coefficient `-0.001673`, |coef| `0.001673`
- `lag_05__T_place_BRICKS`: coefficient `-0.001569`, |coef| `0.001569`
- `lag_02__CT_place_BRIDGE`: coefficient `0.001529`, |coef| `0.001529`
- `lag_01__CT_place_BRIDGE`: coefficient `0.001442`, |coef| `0.001442`
- `lag_07__T1__flash_duration`: coefficient `-0.001360`, |coef| `0.001360`
- `lag_07__T_flashed_players`: coefficient `-0.001353`, |coef| `0.001353`
- `lag_00__kill_diff_last_3s`: coefficient `0.001306`, |coef| `0.001306`
- `lag_00__damage_diff_last_5s`: coefficient `0.001245`, |coef| `0.001245`
- `lag_00__CT3__is_walking`: coefficient `-0.001202`, |coef| `0.001202`
- `lag_00__T_place_STREET`: coefficient `-0.001186`, |coef| `0.001186`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001179`, |coef| `0.001179`
- `lag_10__T3__flash_duration`: coefficient `-0.001176`, |coef| `0.001176`

## Top 10 utility ridge features

- `lag_07__T1__flash_duration`: coefficient `-0.001360` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001176` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000972` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.000930` (lowers CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `-0.000880` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000865` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000859` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000798` (lowers CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `-0.000769` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BRIDGE`: coefficient `0.002387` (raises CT win probability)
- `lag_03__T_place_BRICKS`: coefficient `-0.001931` (lowers CT win probability)
- `lag_05__CT_place_OUTSIDELONG`: coefficient `-0.001677` (lowers CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `-0.001673` (lowers CT win probability)
- `lag_05__T_place_BRICKS`: coefficient `-0.001569` (lowers CT win probability)
- `lag_02__CT_place_BRIDGE`: coefficient `0.001529` (raises CT win probability)
- `lag_01__CT_place_BRIDGE`: coefficient `0.001442` (raises CT win probability)
- `lag_07__T_flashed_players`: coefficient `-0.001353` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001306` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001245` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `16405`, seconds `92.50`, LSTM delta `-0.1851`

Top all feature movements:
- `lag_01__T_place_BRICKS`: contribution `-0.063088`
- `lag_09__CT_place_OUTSIDELONG`: contribution `-0.011605`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.009145`
- `lag_11__T_flashed_players`: contribution `-0.006199`
- `lag_11__T1__flash_duration`: contribution `-0.004630`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `-0.004630`
- `lag_11__T_flash_duration_sum`: contribution `-0.004051`
- `lag_09__CT5__flash_duration`: contribution `-0.002392`
- `lag_09__T5__flash_duration`: contribution `-0.002238`
- `lag_09__CT_flash_duration_sum`: contribution `-0.001992`

### tick `16277`, seconds `90.50`, LSTM delta `-0.1789`

Top all feature movements:
- `lag_05__CT_place_OUTSIDELONG`: contribution `-0.017007`
- `lag_07__T_flashed_players`: contribution `-0.010444`
- `lag_07__T1__flash_duration`: contribution `-0.010233`
- `lag_05__CT5__flash_duration`: contribution `-0.005437`
- `lag_08__T_bomb_zone_count`: contribution `-0.005309`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `-0.010233`
- `lag_05__CT5__flash_duration`: contribution `-0.005437`
- `lag_05__T5__flash_duration`: contribution `-0.004996`
- `lag_07__T_flash_duration_sum`: contribution `-0.003540`
- `lag_05__CT_flash_duration_sum`: contribution `-0.002641`

### tick `13397`, seconds `45.50`, LSTM delta `-0.1632`

Top all feature movements:
- `lag_00__CT_place_BRIDGE`: contribution `-0.027366`
- `lag_11__CT_place_BRIDGE`: contribution `-0.011183`
- `lag_07__CT_place_BRICKS`: contribution `-0.009221`
- `lag_00__T_shots_fired_sum`: contribution `-0.007580`
- `lag_00__T3__shots_fired`: contribution `-0.006625`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.006552`
- `lag_00__CT4__flash_duration`: contribution `-0.004518`
- `lag_08__T4__flash_duration`: contribution `-0.003694`

### tick `15285`, seconds `75.00`, LSTM delta `+0.1454`

Top all feature movements:
- `lag_15__CT_place_BRIDGE`: contribution `+0.012714`
- `lag_10__T3__flash_duration`: contribution `+0.007326`
- `lag_00__T_place_STREET`: contribution `+0.006521`
- `lag_12__T_place_STREET`: contribution `+0.005999`
- `lag_12__T_place_TSTAIRS`: contribution `+0.004294`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `+0.007326`
- `lag_08__T_B_site_active_smokes`: contribution `+0.002225`

### tick `16437`, seconds `93.00`, LSTM delta `+0.1421`

Top all feature movements:
- `lag_00__T_place_BRICKS`: contribution `+0.060088`
- `lag_02__T_place_BRICKS`: contribution `+0.043333`
- `lag_10__CT_place_OUTSIDELONG`: contribution `+0.006977`
- `lag_11__CT2__duck_amount`: contribution `+0.004226`
- `lag_01__CT_place_OUTSIDELONG`: contribution `-0.003791`

Top utility-only movements:
- No utility movement among the top local contributors.
