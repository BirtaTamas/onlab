# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `25228`, seconds `46.50`, LSTM `0.0358`, delta `-0.1323`
- tick `23660`, seconds `22.00`, LSTM `0.3630`, delta `-0.0842`
- tick `24940`, seconds `42.00`, LSTM `0.3101`, delta `-0.0792`
- tick `23788`, seconds `24.00`, LSTM `0.3832`, delta `+0.0701`
- tick `23596`, seconds `21.00`, LSTM `0.4608`, delta `-0.0539`
- tick `24844`, seconds `40.50`, LSTM `0.4140`, delta `-0.0480`
- tick `24972`, seconds `42.50`, LSTM `0.2653`, delta `-0.0448`
- tick `23692`, seconds `22.50`, LSTM `0.3189`, delta `-0.0441`
- tick `23948`, seconds `26.50`, LSTM `0.4638`, delta `+0.0409`
- tick `25804`, seconds `55.50`, LSTM `0.0264`, delta `-0.0362`

## Top 15 local ridge features

- `lag_07__CT_shots_fired_sum`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.001263`, |coef| `0.001263`
- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.001136`, |coef| `0.001136`
- `lag_00__T_place_CONTROL`: coefficient `-0.000950`, |coef| `0.000950`
- `lag_00__T_burning_players`: coefficient `-0.000853`, |coef| `0.000853`
- `lag_06__T_place_CONTROL`: coefficient `-0.000844`, |coef| `0.000844`
- `lag_00__T_kills_last_3s`: coefficient `-0.000817`, |coef| `0.000817`
- `lag_07__CT3__shots_fired`: coefficient `-0.000805`, |coef| `0.000805`
- `lag_00__CT2__flash_duration`: coefficient `-0.000765`, |coef| `0.000765`
- `lag_09__CT2__flash_duration`: coefficient `-0.000763`, |coef| `0.000763`
- `lag_09__T_place_CONTROL`: coefficient `-0.000763`, |coef| `0.000763`
- `lag_03__T_place_TROPHY`: coefficient `0.000748`, |coef| `0.000748`
- `lag_05__CT_shots_fired_sum`: coefficient `-0.000727`, |coef| `0.000727`
- `lag_01__T_burning_players`: coefficient `-0.000715`, |coef| `0.000715`
- `lag_00__T_damage_last_5s`: coefficient `-0.000714`, |coef| `0.000714`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `-0.000765` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.000763` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000714` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000656` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000607` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `0.000584` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000580` (raises CT win probability)
- `lag_00__T4__utility_total`: coefficient `0.000549` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000543` (raises CT win probability)
- `lag_00__T5__molly`: coefficient `0.000516` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_shots_fired_sum`: coefficient `-0.001274` (lowers CT win probability)
- `lag_09__CT_place_LOCKERROOM`: coefficient `-0.001263` (lowers CT win probability)
- `lag_10__CT_place_LOCKERROOM`: coefficient `-0.001136` (lowers CT win probability)
- `lag_00__T_place_CONTROL`: coefficient `-0.000950` (lowers CT win probability)
- `lag_00__T_burning_players`: coefficient `-0.000853` (lowers CT win probability)
- `lag_06__T_place_CONTROL`: coefficient `-0.000844` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000817` (lowers CT win probability)
- `lag_07__CT3__shots_fired`: coefficient `-0.000805` (lowers CT win probability)
- `lag_09__T_place_CONTROL`: coefficient `-0.000763` (lowers CT win probability)
- `lag_03__T_place_TROPHY`: coefficient `0.000748` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25228`, seconds `46.50`, LSTM delta `-0.1323`

Top all feature movements:
- `lag_06__T_place_CONTROL`: contribution `-0.005997`
- `lag_09__CT2__flash_duration`: contribution `-0.005471`
- `lag_09__T_place_CONTROL`: contribution `-0.005419`
- `lag_03__T_place_TROPHY`: contribution `-0.004745`
- `lag_12__T_place_CONTROL`: contribution `-0.004510`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.005471`
- `lag_00__CT1__utility_total`: contribution `-0.001848`

### tick `23660`, seconds `22.00`, LSTM delta `-0.0842`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.009179`
- `lag_03__CT3__shots_fired`: contribution `-0.005546`
- `lag_06__CT3__flash_duration`: contribution `-0.004015`
- `lag_07__CT_shots_fired_sum`: contribution `-0.003541`
- `lag_05__CT_shots_fired_sum`: contribution `-0.002526`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.004015`
- `lag_02__CT3__flash_duration`: contribution `-0.002361`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.001915`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001566`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.001181`

### tick `24940`, seconds `42.00`, LSTM delta `-0.0792`

Top all feature movements:
- `lag_09__CT_place_LOCKERROOM`: contribution `-0.015727`
- `lag_00__T_place_CONTROL`: contribution `-0.006749`
- `lag_00__CT2__flash_duration`: contribution `-0.005482`
- `lag_03__T_place_TROPHY`: contribution `-0.004745`
- `lag_03__CT_place_MINI`: contribution `-0.002599`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.005482`

### tick `23788`, seconds `24.00`, LSTM delta `+0.0701`

Top all feature movements:
- `lag_07__CT_shots_fired_sum`: contribution `+0.016820`
- `lag_07__CT3__shots_fired`: contribution `+0.007870`
- `lag_06__CT3__flash_duration`: contribution `+0.004015`
- `lag_00__T_kills_last_3s`: contribution `+0.002590`
- `lag_05__CT_shots_fired_sum`: contribution `-0.002526`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.004015`
- `lag_11__T_B_site_active_infernos`: contribution `+0.001388`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.001181`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.001165`

### tick `23596`, seconds `21.00`, LSTM delta `-0.0539`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.007390`
- `lag_01__CT3__shots_fired`: contribution `-0.005088`
- `lag_00__T_kills_last_3s`: contribution `-0.002590`
- `lag_03__CT_shots_fired_sum`: contribution `+0.002416`
- `lag_05__CT_shots_fired_sum`: contribution `-0.002020`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.001661`
- `lag_04__CT3__flash_duration`: contribution `-0.001546`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.001347`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.001151`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.001022`
