# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `10`

## Largest probability jumps

- tick `64338`, seconds `20.00`, LSTM `0.1277`, delta `-0.3455`
- tick `66162`, seconds `48.50`, LSTM `0.0359`, delta `-0.1307`
- tick `64434`, seconds `21.50`, LSTM `0.1942`, delta `+0.0694`
- tick `64306`, seconds `19.50`, LSTM `0.4732`, delta `-0.0511`
- tick `64626`, seconds `24.50`, LSTM `0.2548`, delta `-0.0493`
- tick `64274`, seconds `19.00`, LSTM `0.5244`, delta `-0.0450`
- tick `64466`, seconds `22.00`, LSTM `0.2324`, delta `+0.0381`
- tick `64594`, seconds `24.00`, LSTM `0.3041`, delta `+0.0367`
- tick `64786`, seconds `27.00`, LSTM `0.2485`, delta `-0.0328`
- tick `64530`, seconds `23.00`, LSTM `0.2597`, delta `+0.0311`

## Top 15 local ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.001858`, |coef| `0.001858`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001801`, |coef| `0.001801`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001627`, |coef| `0.001627`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001613`, |coef| `0.001613`
- `lag_02__T1__shots_fired`: coefficient `-0.001455`, |coef| `0.001455`
- `lag_00__T_kills_last_3s`: coefficient `-0.001387`, |coef| `0.001387`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.001382`, |coef| `0.001382`
- `lag_06__T5__flash_duration`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_12__CT_flashed_players`: coefficient `-0.001298`, |coef| `0.001298`
- `lag_06__CT1__flash_duration`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_01__CT2__shots_fired`: coefficient `-0.001234`, |coef| `0.001234`
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001161`, |coef| `0.001161`
- `lag_06__T_place_TUNNEL`: coefficient `-0.001130`, |coef| `0.001130`
- `lag_00__CT4__flash_duration`: coefficient `0.001116`, |coef| `0.001116`
- `lag_00__T1__flash_duration`: coefficient `0.001079`, |coef| `0.001079`

## Top 10 utility ridge features

- `lag_06__T5__flash_duration`: coefficient `-0.001342` (lowers CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.001237` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001116` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001079` (raises CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.000975` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000954` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.000846` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000840` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000784` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000752` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.001858` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001801` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001627` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001613` (lowers CT win probability)
- `lag_02__T1__shots_fired`: coefficient `-0.001455` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001387` (lowers CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.001382` (raises CT win probability)
- `lag_12__CT_flashed_players`: coefficient `-0.001298` (lowers CT win probability)
- `lag_01__CT2__shots_fired`: coefficient `-0.001234` (lowers CT win probability)
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001161` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `64338`, seconds `20.00`, LSTM delta `-0.3455`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.012092`
- `lag_01__T_shots_fired_sum`: contribution `-0.009758`
- `lag_12__CT_flashed_players`: contribution `-0.008526`
- `lag_06__T5__flash_duration`: contribution `-0.008516`
- `lag_06__CT1__flash_duration`: contribution `-0.007977`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `-0.008516`
- `lag_06__CT1__flash_duration`: contribution `-0.007977`
- `lag_00__CT4__flash_duration`: contribution `-0.006186`
- `lag_00__T1__flash_duration`: contribution `-0.005643`
- `lag_09__T1__flash_duration`: contribution `-0.004989`

### tick `66162`, seconds `48.50`, LSTM delta `-0.1307`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.013537`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.008725`
- `lag_02__CT_place_TSIDEUPPER`: contribution `-0.008046`
- `lag_00__T_shots_fired_sum`: contribution `-0.004837`
- `lag_00__T_kills_last_3s`: contribution `-0.004394`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `64434`, seconds `21.50`, LSTM delta `+0.0694`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.036226`
- `lag_02__T1__shots_fired`: contribution `+0.015656`
- `lag_03__T_shots_fired_sum`: contribution `-0.006285`
- `lag_02__T1__duck_amount`: contribution `+0.004149`
- `lag_11__T1__duck_amount`: contribution `-0.003181`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.002887`
- `lag_09__T5__flash_duration`: contribution `+0.002425`
- `lag_05__CT_active_infernos`: contribution `+0.001357`
- `lag_12__CT4__flash_duration`: contribution `-0.001331`

### tick `64306`, seconds `19.50`, LSTM delta `-0.0511`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.009673`
- `lag_01__T_shots_fired_sum`: contribution `-0.006099`
- `lag_02__T_shots_fired_sum`: contribution `-0.004180`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003193`
- `lag_15__T_burning_players`: contribution `-0.003080`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.002192`
- `lag_05__CT1__flash_duration`: contribution `-0.001916`
- `lag_08__T1__flash_duration`: contribution `-0.001669`

### tick `64626`, seconds `24.50`, LSTM delta `-0.0493`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.014587`
- `lag_09__T1__flash_duration`: contribution `+0.004989`
- `lag_02__T_shots_fired_sum`: contribution `-0.004180`
- `lag_01__T_shots_fired_sum`: contribution `+0.003659`
- `lag_08__T1__shots_fired`: contribution `-0.003304`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.004989`
- `lag_15__T5__flash_duration`: contribution `-0.002032`
- `lag_15__CT1__flash_duration`: contribution `-0.001715`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001649`
- `lag_11__T_B_site_active_infernos`: contribution `-0.001435`
