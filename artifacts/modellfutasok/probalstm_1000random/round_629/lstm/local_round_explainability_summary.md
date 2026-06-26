# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `17`

## Largest probability jumps

- tick `136447`, seconds `75.50`, LSTM `0.6986`, delta `-0.2313`
- tick `133759`, seconds `33.50`, LSTM `0.9129`, delta `+0.1104`
- tick `132991`, seconds `21.50`, LSTM `0.8084`, delta `+0.0797`
- tick `136671`, seconds `79.00`, LSTM `0.6030`, delta `-0.0572`
- tick `136511`, seconds `76.50`, LSTM `0.6085`, delta `-0.0561`
- tick `136799`, seconds `81.00`, LSTM `0.6213`, delta `+0.0543`
- tick `136639`, seconds `78.50`, LSTM `0.6603`, delta `+0.0538`
- tick `136351`, seconds `74.00`, LSTM `0.9191`, delta `-0.0505`
- tick `133823`, seconds `34.50`, LSTM `0.9631`, delta `+0.0368`
- tick `131807`, seconds `3.00`, LSTM `0.7078`, delta `-0.0344`

## Top 15 local ridge features

- `lag_13__T5__flash_duration`: coefficient `0.002701`, |coef| `0.002701`
- `lag_00__kill_diff_last_3s`: coefficient `0.002083`, |coef| `0.002083`
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.002078`, |coef| `0.002078`
- `lag_00__T_kills_last_3s`: coefficient `-0.001833`, |coef| `0.001833`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001654`, |coef| `0.001654`
- `lag_15__CT4__duck_amount`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_10__CT1__duck_amount`: coefficient `0.001536`, |coef| `0.001536`
- `lag_10__T_velocity_mean`: coefficient `0.001440`, |coef| `0.001440`
- `lag_03__T_kills_last_3s`: coefficient `-0.001415`, |coef| `0.001415`
- `lag_15__T_A_site_active_infernos`: coefficient `0.001377`, |coef| `0.001377`
- `lag_01__CT_A_site_active_infernos`: coefficient `0.001375`, |coef| `0.001375`
- `lag_04__CT_place_HEAVEN`: coefficient `0.001359`, |coef| `0.001359`
- `lag_00__CT_place_MIDDLE`: coefficient `0.001349`, |coef| `0.001349`
- `lag_00__CT1__alive`: coefficient `0.001267`, |coef| `0.001267`
- `lag_03__CT3__alive`: coefficient `0.001266`, |coef| `0.001266`

## Top 10 utility ridge features

- `lag_13__T5__flash_duration`: coefficient `0.002701` (raises CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.002078` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.001377` (raises CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.001375` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.001241` (raises CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001101` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `0.001100` (raises CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.001091` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.000964` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.000929` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002083` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001833` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001654` (lowers CT win probability)
- `lag_15__CT4__duck_amount`: coefficient `-0.001637` (lowers CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `0.001536` (raises CT win probability)
- `lag_10__T_velocity_mean`: coefficient `0.001440` (raises CT win probability)
- `lag_03__T_kills_last_3s`: coefficient `-0.001415` (lowers CT win probability)
- `lag_04__CT_place_HEAVEN`: coefficient `0.001359` (raises CT win probability)
- `lag_00__CT_place_MIDDLE`: coefficient `0.001349` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001267` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `136447`, seconds `75.50`, LSTM delta `-0.2313`

Top all feature movements:
- `lag_13__T5__flash_duration`: contribution `-0.020021`
- `lag_04__CT_place_HEAVEN`: contribution `-0.007337`
- `lag_12__CT_A_site_active_infernos`: contribution `-0.007332`
- `lag_12__CT_place_HEAVEN`: contribution `-0.006499`
- `lag_15__CT4__duck_amount`: contribution `-0.006013`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.020021`
- `lag_12__CT_A_site_active_infernos`: contribution `-0.007332`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.004852`
- `lag_15__T_A_site_active_infernos`: contribution `-0.004099`
- `lag_13__T_flash_duration_sum`: contribution `-0.003314`

### tick `133759`, seconds `33.50`, LSTM delta `+0.1104`

Top all feature movements:
- `lag_09__CT_place_BACKOFB`: contribution `+0.013838`
- `lag_00__kill_diff_last_3s`: contribution `+0.005013`
- `lag_00__damage_diff_last_5s`: contribution `+0.004705`
- `lag_10__CT_place_WALKWAY`: contribution `+0.004254`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004096`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `132991`, seconds `21.50`, LSTM delta `+0.0797`

Top all feature movements:
- `lag_02__T2__flash_duration`: contribution `+0.008800`
- `lag_00__kill_diff_last_3s`: contribution `+0.005013`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002926`
- `lag_04__CT_shots_fired_sum`: contribution `+0.002812`
- `lag_00__CT_kills_last_3s`: contribution `+0.002390`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.008800`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.002356`
- `lag_02__T_flash_duration_sum`: contribution `+0.001984`
- `lag_00__T2__flash_duration`: contribution `-0.001877`
- `lag_09__CT2__molly`: contribution `+0.001250`

### tick `136671`, seconds `79.00`, LSTM delta `-0.0572`

Top all feature movements:
- `lag_04__CT_place_MAIN`: contribution `-0.006774`
- `lag_06__T_bomb_zone_count`: contribution `-0.005847`
- `lag_11__CT_place_HEAVEN`: contribution `+0.004428`
- `lag_10__CT_place_WALKWAY`: contribution `-0.004254`
- `lag_03__T_velocity_mean`: contribution `+0.004132`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `136511`, seconds `76.50`, LSTM delta `-0.0561`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `-0.006195`
- `lag_14__CT_place_WALKWAY`: contribution `-0.003572`
- `lag_15__T5__flash_duration`: contribution `-0.003024`
- `lag_12__CT1__duck_amount`: contribution `+0.002866`
- `lag_04__CT_shots_fired_sum`: contribution `-0.002343`

Top utility-only movements:
- `lag_15__T5__flash_duration`: contribution `-0.003024`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.002309`
