# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `11772`, seconds `75.00`, LSTM `0.2297`, delta `-0.2565`
- tick `10780`, seconds `59.50`, LSTM `0.1975`, delta `-0.2238`
- tick `10908`, seconds `61.50`, LSTM `0.4079`, delta `+0.1949`
- tick `9308`, seconds `36.50`, LSTM `0.7362`, delta `+0.1560`
- tick `11932`, seconds `77.50`, LSTM `0.3341`, delta `+0.1473`
- tick `11996`, seconds `78.50`, LSTM `0.2076`, delta `-0.1296`
- tick `9596`, seconds `41.00`, LSTM `0.5326`, delta `-0.1217`
- tick `10876`, seconds `61.00`, LSTM `0.2130`, delta `+0.1082`
- tick `10748`, seconds `59.00`, LSTM `0.4213`, delta `-0.0750`
- tick `10844`, seconds `60.50`, LSTM `0.1049`, delta `-0.0716`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004426`, |coef| `0.004426`
- `lag_00__T_kills_last_3s`: coefficient `-0.003790`, |coef| `0.003790`
- `lag_00__T_place_TROPHY`: coefficient `-0.003080`, |coef| `0.003080`
- `lag_10__CT_place_MINI`: coefficient `-0.002963`, |coef| `0.002963`
- `lag_15__CT_place_SECRET`: coefficient `0.002890`, |coef| `0.002890`
- `lag_06__CT2__duck_amount`: coefficient `0.002785`, |coef| `0.002785`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002741`, |coef| `0.002741`
- `lag_01__T_shots_fired_sum`: coefficient `-0.002640`, |coef| `0.002640`
- `lag_15__T_A_site_active_infernos`: coefficient `0.002361`, |coef| `0.002361`
- `lag_10__CT_place_LOCKERROOM`: coefficient `0.002356`, |coef| `0.002356`
- `lag_15__T_B_site_active_infernos`: coefficient `0.002246`, |coef| `0.002246`
- `lag_09__CT_place_MINI`: coefficient `-0.002108`, |coef| `0.002108`
- `lag_00__CT1__alive`: coefficient `0.002074`, |coef| `0.002074`
- `lag_03__T2__duck_amount`: coefficient `0.002050`, |coef| `0.002050`
- `lag_05__T2__duck_amount`: coefficient `-0.001953`, |coef| `0.001953`

## Top 10 utility ridge features

- `lag_15__T_A_site_active_infernos`: coefficient `0.002361` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.002246` (raises CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.001674` (raises CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `0.001142` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.001022` (raises CT win probability)
- `lag_15__active_infernos_total`: coefficient `0.000981` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000975` (raises CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `0.000970` (raises CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `0.000764` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000738` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004426` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003790` (lowers CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `-0.003080` (lowers CT win probability)
- `lag_10__CT_place_MINI`: coefficient `-0.002963` (lowers CT win probability)
- `lag_15__CT_place_SECRET`: coefficient `0.002890` (raises CT win probability)
- `lag_06__CT2__duck_amount`: coefficient `0.002785` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002741` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.002640` (lowers CT win probability)
- `lag_10__CT_place_LOCKERROOM`: coefficient `0.002356` (raises CT win probability)
- `lag_09__CT_place_MINI`: coefficient `-0.002108` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `11772`, seconds `75.00`, LSTM delta `-0.2565`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.012007`
- `lag_00__kill_diff_last_3s`: contribution `-0.010653`
- `lag_06__CT2__duck_amount`: contribution `-0.010609`
- `lag_03__T2__duck_amount`: contribution `-0.007838`
- `lag_15__T_A_site_active_infernos`: contribution `-0.007028`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `-0.007028`
- `lag_15__T_B_site_active_infernos`: contribution `-0.006351`
- `lag_15__T_active_infernos`: contribution `-0.003487`

### tick `10780`, seconds `59.50`, LSTM delta `-0.2238`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `-0.019532`
- `lag_10__CT_place_MINI`: contribution `-0.018167`
- `lag_00__T_kills_last_3s`: contribution `-0.012007`
- `lag_00__kill_diff_last_3s`: contribution `-0.010653`
- `lag_06__CT_place_MINI`: contribution `-0.009483`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10908`, seconds `61.50`, LSTM delta `+0.1949`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.019532`
- `lag_10__CT_place_MINI`: contribution `+0.018167`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_04__T_place_VENDING`: contribution `+0.006279`
- `lag_00__T_place_CONTROL`: contribution `+0.006208`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9308`, seconds `36.50`, LSTM delta `+0.1560`

Top all feature movements:
- `lag_10__CT_place_LOCKERROOM`: contribution `+0.029328`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_06__CT2__duck_amount`: contribution `+0.010609`
- `lag_00__CT_kills_last_3s`: contribution `+0.005356`
- `lag_01__T4__is_walking`: contribution `+0.004283`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11932`, seconds `77.50`, LSTM delta `+0.1473`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.027706`
- `lag_01__T2__shots_fired`: contribution `+0.015349`
- `lag_00__kill_diff_last_3s`: contribution `+0.010653`
- `lag_11__CT4__duck_amount`: contribution `+0.006331`
- `lag_00__CT_kills_last_3s`: contribution `+0.005356`

Top utility-only movements:
- No utility movement among the top local contributors.
