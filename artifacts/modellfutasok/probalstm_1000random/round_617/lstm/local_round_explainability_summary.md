# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `61682`, seconds `28.00`, LSTM `0.1787`, delta `-0.1613`
- tick `61746`, seconds `29.00`, LSTM `0.0636`, delta `-0.1480`
- tick `61554`, seconds `26.00`, LSTM `0.4015`, delta `+0.1084`
- tick `61394`, seconds `23.50`, LSTM `0.3031`, delta `-0.0958`
- tick `61874`, seconds `31.00`, LSTM `0.0542`, delta `-0.0766`
- tick `61778`, seconds `29.50`, LSTM `0.1266`, delta `+0.0630`
- tick `59922`, seconds `0.50`, LSTM `0.2571`, delta `-0.0565`
- tick `61906`, seconds `31.50`, LSTM `0.1073`, delta `+0.0531`
- tick `62290`, seconds `37.50`, LSTM `0.0178`, delta `-0.0478`
- tick `59954`, seconds `1.00`, LSTM `0.3046`, delta `+0.0475`

## Top 15 local ridge features

- `lag_00__T3__shots_fired`: coefficient `-0.001522`, |coef| `0.001522`
- `lag_04__T_shots_fired_sum`: coefficient `0.001263`, |coef| `0.001263`
- `lag_07__T3__shots_fired`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_06__CT_place_JUNGLE`: coefficient `-0.001047`, |coef| `0.001047`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_01__CT_place_JUNGLE`: coefficient `0.000941`, |coef| `0.000941`
- `lag_05__T_shots_fired_sum`: coefficient `-0.000913`, |coef| `0.000913`
- `lag_11__T3__shots_fired`: coefficient `-0.000910`, |coef| `0.000910`
- `lag_00__T_kills_last_3s`: coefficient `-0.000888`, |coef| `0.000888`
- `lag_06__CT_place_STAIRS`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_08__T3__shots_fired`: coefficient `-0.000862`, |coef| `0.000862`
- `lag_04__T3__shots_fired`: coefficient `0.000848`, |coef| `0.000848`
- `lag_07__T_shots_fired_sum`: coefficient `-0.000833`, |coef| `0.000833`
- `lag_00__CT_place_STAIRS`: coefficient `0.000818`, |coef| `0.000818`
- `lag_09__T3__shots_fired`: coefficient `-0.000758`, |coef| `0.000758`

## Top 10 utility ridge features

- `lag_01__T_mollies_last_5s`: coefficient `-0.000609` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.000512` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000502` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000457` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000433` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000426` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000426` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.000424` (raises CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.000403` (raises CT win probability)
- `lag_14__T_smokes_last_5s`: coefficient `-0.000388` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T3__shots_fired`: coefficient `-0.001522` (lowers CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `0.001263` (raises CT win probability)
- `lag_07__T3__shots_fired`: coefficient `-0.001237` (lowers CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `-0.001047` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000992` (lowers CT win probability)
- `lag_01__CT_place_JUNGLE`: coefficient `0.000941` (raises CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `-0.000913` (lowers CT win probability)
- `lag_11__T3__shots_fired`: coefficient `-0.000910` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000888` (lowers CT win probability)
- `lag_06__CT_place_STAIRS`: coefficient `-0.000874` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `61682`, seconds `28.00`, LSTM delta `-0.1613`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.023666`
- `lag_04__T3__shots_fired`: contribution `-0.012844`
- `lag_06__CT_place_STAIRS`: contribution `-0.006802`
- `lag_00__T_shots_fired_sum`: contribution `-0.006694`
- `lag_15__CT_place_JUNGLE`: contribution `-0.004261`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61746`, seconds `29.00`, LSTM delta `-0.1480`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.016363`
- `lag_06__T_shots_fired_sum`: contribution `-0.011235`
- `lag_00__CT_place_STAIRS`: contribution `-0.006366`
- `lag_00__T_kills_last_3s`: contribution `-0.005625`
- `lag_03__T2__is_scoped`: contribution `-0.005610`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61554`, seconds `26.00`, LSTM delta `+0.1084`

Top all feature movements:
- `lag_00__T3__shots_fired`: contribution `+0.023038`
- `lag_00__T_shots_fired_sum`: contribution `+0.018594`
- `lag_06__CT_place_JUNGLE`: contribution `+0.006719`
- `lag_04__T_shots_fired_sum`: contribution `+0.004733`
- `lag_05__T_shots_fired_sum`: contribution `-0.003422`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `+0.001305`

### tick `61394`, seconds `23.50`, LSTM delta `-0.0958`

Top all feature movements:
- `lag_06__CT_place_JUNGLE`: contribution `-0.006719`
- `lag_01__CT_place_JUNGLE`: contribution `-0.006036`
- `lag_00__T3__shots_fired`: contribution `-0.004608`
- `lag_00__T_shots_fired_sum`: contribution `-0.003719`
- `lag_00__CT3__is_scoped`: contribution `-0.003360`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.001524`
- `lag_11__T_B_site_active_infernos`: contribution `-0.001223`

### tick `61874`, seconds `31.00`, LSTM delta `-0.0766`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.020826`
- `lag_10__T_shots_fired_sum`: contribution `+0.009856`
- `lag_05__T_shots_fired_sum`: contribution `-0.007529`
- `lag_10__T3__shots_fired`: contribution `+0.005437`
- `lag_00__T3__shots_fired`: contribution `-0.004608`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.001870`
