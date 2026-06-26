# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `81107`, seconds `63.50`, LSTM `0.0994`, delta `-0.2951`
- tick `77779`, seconds `11.50`, LSTM `0.4647`, delta `-0.1634`
- tick `81075`, seconds `63.00`, LSTM `0.3945`, delta `+0.1316`
- tick `77811`, seconds `12.00`, LSTM `0.3941`, delta `-0.0706`
- tick `77939`, seconds `14.00`, LSTM `0.3964`, delta `+0.0546`
- tick `77683`, seconds `10.00`, LSTM `0.6600`, delta `-0.0454`
- tick `77907`, seconds `13.50`, LSTM `0.3418`, delta `-0.0443`
- tick `80307`, seconds `51.00`, LSTM `0.3740`, delta `-0.0430`
- tick `80595`, seconds `55.50`, LSTM `0.3791`, delta `-0.0390`
- tick `81619`, seconds `71.50`, LSTM `0.0564`, delta `-0.0390`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001748`, |coef| `0.001748`
- `lag_15__CT_place_EXTENDEDA`: coefficient `0.001595`, |coef| `0.001595`
- `lag_01__CT5__shots_fired`: coefficient `-0.001540`, |coef| `0.001540`
- `lag_04__T4__flash_duration`: coefficient `0.001499`, |coef| `0.001499`
- `lag_00__T_kills_last_3s`: coefficient `-0.001493`, |coef| `0.001493`
- `lag_00__CT5__duck_amount`: coefficient `-0.001492`, |coef| `0.001492`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001457`, |coef| `0.001457`
- `lag_08__CT_place_EXTENDEDA`: coefficient `-0.001316`, |coef| `0.001316`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001268`, |coef| `0.001268`
- `lag_00__CT_money_sum`: coefficient `0.001255`, |coef| `0.001255`
- `lag_00__CT_start_balance_sum`: coefficient `0.001240`, |coef| `0.001240`
- `lag_01__T5__shots_fired`: coefficient `-0.001232`, |coef| `0.001232`
- `lag_00__kill_diff_last_3s`: coefficient `0.001220`, |coef| `0.001220`
- `lag_10__CT_place_EXTENDEDA`: coefficient `0.001184`, |coef| `0.001184`
- `lag_13__CT_flashed_players`: coefficient `-0.001179`, |coef| `0.001179`

## Top 10 utility ridge features

- `lag_04__T4__flash_duration`: coefficient `0.001499` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.000963` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000923` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000879` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000855` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.000842` (lowers CT win probability)
- `lag_06__T2__molly`: coefficient `0.000790` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000785` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000776` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000704` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001748` (raises CT win probability)
- `lag_15__CT_place_EXTENDEDA`: coefficient `0.001595` (raises CT win probability)
- `lag_01__CT5__shots_fired`: coefficient `-0.001540` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001493` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `-0.001492` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001457` (lowers CT win probability)
- `lag_08__CT_place_EXTENDEDA`: coefficient `-0.001316` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001268` (lowers CT win probability)
- `lag_00__CT_money_sum`: coefficient `0.001255` (raises CT win probability)
- `lag_00__CT_start_balance_sum`: coefficient `0.001240` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `81107`, seconds `63.50`, LSTM delta `-0.2951`

Top all feature movements:
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.008956`
- `lag_04__T4__flash_duration`: contribution `-0.008150`
- `lag_08__CT_place_EXTENDEDA`: contribution `-0.007387`
- `lag_10__CT_place_EXTENDEDA`: contribution `-0.006647`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006074`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `-0.008150`

### tick `77779`, seconds `11.50`, LSTM delta `-0.1634`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.011650`
- `lag_00__T_shots_fired_sum`: contribution `-0.009832`
- `lag_01__T_shots_fired_sum`: contribution `-0.004754`
- `lag_00__T_kills_last_3s`: contribution `-0.004730`
- `lag_14__T_place_OUTSIDETUNNEL`: contribution `-0.003984`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.003768`
- `lag_08__CT3__flash_duration`: contribution `-0.002420`

### tick `81075`, seconds `63.00`, LSTM delta `+0.1316`

Top all feature movements:
- `lag_15__CT_place_EXTENDEDA`: contribution `+0.008956`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006074`
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.005864`
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.005361`
- `lag_00__CT5__duck_amount`: contribution `+0.004817`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.002183`

### tick `77811`, seconds `12.00`, LSTM delta `-0.0706`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.010925`
- `lag_01__T_shots_fired_sum`: contribution `-0.008558`
- `lag_01__T5__shots_fired`: contribution `-0.004546`
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `-0.003539`
- `lag_02__T_shots_fired_sum`: contribution `-0.003301`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `-0.002342`
- `lag_02__T1__flash_duration`: contribution `-0.001530`

### tick `77939`, seconds `14.00`, LSTM delta `+0.0546`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.025127`
- `lag_01__T_shots_fired_sum`: contribution `+0.020919`
- `lag_01__T5__shots_fired`: contribution `+0.020455`
- `lag_02__T_shots_fired_sum`: contribution `-0.006602`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005582`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.002756`
