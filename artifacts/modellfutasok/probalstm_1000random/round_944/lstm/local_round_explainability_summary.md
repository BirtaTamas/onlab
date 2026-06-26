# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `23706`, seconds `16.00`, LSTM `0.6711`, delta `+0.2394`
- tick `24474`, seconds `28.00`, LSTM `0.6312`, delta `+0.1663`
- tick `26106`, seconds `53.50`, LSTM `0.9179`, delta `+0.0984`
- tick `24026`, seconds `21.00`, LSTM `0.5474`, delta `-0.0715`
- tick `24570`, seconds `29.50`, LSTM `0.7514`, delta `+0.0711`
- tick `24506`, seconds `28.50`, LSTM `0.6968`, delta `+0.0656`
- tick `24378`, seconds `26.50`, LSTM `0.4324`, delta `-0.0586`
- tick `24602`, seconds `30.00`, LSTM `0.8039`, delta `+0.0525`
- tick `24442`, seconds `27.50`, LSTM `0.4649`, delta `+0.0510`
- tick `23898`, seconds `19.00`, LSTM `0.6422`, delta `-0.0303`

## Top 15 local ridge features

- `lag_13__T_he_last_5s`: coefficient `-0.002119`, |coef| `0.002119`
- `lag_14__CT3__flash_duration`: coefficient `-0.001990`, |coef| `0.001990`
- `lag_14__CT2__flash_duration`: coefficient `-0.001769`, |coef| `0.001769`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001715`, |coef| `0.001715`
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001645`, |coef| `0.001645`
- `lag_07__CT1__shots_fired`: coefficient `-0.001630`, |coef| `0.001630`
- `lag_03__T1__duck_amount`: coefficient `-0.001549`, |coef| `0.001549`
- `lag_00__CT_kills_last_3s`: coefficient `0.001373`, |coef| `0.001373`
- `lag_14__CT_flashed_players`: coefficient `-0.001308`, |coef| `0.001308`
- `lag_00__T1__duck_amount`: coefficient `-0.001299`, |coef| `0.001299`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001294`, |coef| `0.001294`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001272`, |coef| `0.001272`
- `lag_00__kill_diff_last_3s`: coefficient `0.001209`, |coef| `0.001209`
- `lag_04__T2__flash_duration`: coefficient `0.001198`, |coef| `0.001198`

## Top 10 utility ridge features

- `lag_13__T_he_last_5s`: coefficient `-0.002119` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.001990` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001769` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001711` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.001198` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `-0.001032` (lowers CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.000974` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000968` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.000923` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.000918` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001715` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001645` (raises CT win probability)
- `lag_07__CT1__shots_fired`: coefficient `-0.001630` (lowers CT win probability)
- `lag_03__T1__duck_amount`: coefficient `-0.001549` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001373` (raises CT win probability)
- `lag_14__CT_flashed_players`: coefficient `-0.001308` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001299` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001294` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001272` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001209` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `23706`, seconds `16.00`, LSTM delta `+0.2394`

Top all feature movements:
- `lag_13__T_he_last_5s`: contribution `+0.027662`
- `lag_07__CT1__shots_fired`: contribution `+0.015502`
- `lag_07__CT_shots_fired_sum`: contribution `+0.009837`
- `lag_00__T_shots_fired_sum`: contribution `+0.007764`
- `lag_04__T2__flash_duration`: contribution `+0.007074`

Top utility-only movements:
- `lag_13__T_he_last_5s`: contribution `+0.027662`
- `lag_04__T2__flash_duration`: contribution `+0.007074`
- `lag_03__CT3__flash_duration`: contribution `+0.005630`
- `lag_09__CT5__flash_duration`: contribution `+0.004547`
- `lag_04__CT2__flash_duration`: contribution `+0.003232`

### tick `24474`, seconds `28.00`, LSTM delta `+0.1663`

Top all feature movements:
- `lag_14__CT3__flash_duration`: contribution `+0.014950`
- `lag_14__CT2__flash_duration`: contribution `+0.010936`
- `lag_14__CT_flash_duration_sum`: contribution `+0.010550`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.008368`
- `lag_14__CT_flashed_players`: contribution `+0.005730`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `+0.014950`
- `lag_14__CT2__flash_duration`: contribution `+0.010936`
- `lag_14__CT_flash_duration_sum`: contribution `+0.010550`

### tick `26106`, seconds `53.50`, LSTM delta `+0.0984`

Top all feature movements:
- `lag_13__CT_place_TSIDELOWER`: contribution `+0.011279`
- `lag_03__T1__duck_amount`: contribution `+0.006066`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005715`
- `lag_00__CT5__flash_duration`: contribution `+0.004805`
- `lag_00__CT_kills_last_3s`: contribution `+0.003965`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.004805`
- `lag_02__T5__flash_duration`: contribution `+0.002806`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001943`

### tick `24026`, seconds `21.00`, LSTM delta `-0.0715`

Top all feature movements:
- `lag_14__CT2__flash_duration`: contribution `-0.008738`
- `lag_09__CT_shots_fired_sum`: contribution `-0.008008`
- `lag_04__T2__flash_duration`: contribution `-0.007074`
- `lag_13__CT3__flash_duration`: contribution `-0.005973`
- `lag_14__CT_flash_duration_sum`: contribution `-0.003836`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `-0.008738`
- `lag_04__T2__flash_duration`: contribution `-0.007074`
- `lag_13__CT3__flash_duration`: contribution `-0.005973`
- `lag_14__CT_flash_duration_sum`: contribution `-0.003836`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003371`

### tick `24570`, seconds `29.50`, LSTM delta `+0.0711`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.010006`
- `lag_03__T1__duck_amount`: contribution `+0.006066`
- `lag_07__CT_place_HOUSE`: contribution `+0.002990`
- `lag_14__CT_place_HOUSE`: contribution `+0.002597`
- `lag_06__T5__duck_amount`: contribution `+0.002132`

Top utility-only movements:
- No utility movement among the top local contributors.
