# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `53749`, seconds `90.50`, LSTM `0.4779`, delta `-0.2606`
- tick `52789`, seconds `75.50`, LSTM `0.8397`, delta `+0.2222`
- tick `52085`, seconds `64.50`, LSTM `0.5110`, delta `-0.2129`
- tick `52053`, seconds `64.00`, LSTM `0.7238`, delta `-0.1788`
- tick `52437`, seconds `70.00`, LSTM `0.7694`, delta `+0.1680`
- tick `53493`, seconds `86.50`, LSTM `0.7463`, delta `-0.1622`
- tick `53781`, seconds `91.00`, LSTM `0.3335`, delta `-0.1444`
- tick `52117`, seconds `65.00`, LSTM `0.6543`, delta `+0.1434`
- tick `51765`, seconds `59.50`, LSTM `0.9095`, delta `+0.1081`
- tick `52149`, seconds `65.50`, LSTM `0.5479`, delta `-0.1064`

## Top 15 local ridge features

- `lag_00__CT_place_VENTS`: coefficient `0.003875`, |coef| `0.003875`
- `lag_15__CT_place_VENTS`: coefficient `-0.003342`, |coef| `0.003342`
- `lag_00__kill_diff_last_3s`: coefficient `0.002817`, |coef| `0.002817`
- `lag_08__CT_place_VENTS`: coefficient `0.002760`, |coef| `0.002760`
- `lag_00__damage_diff_last_5s`: coefficient `0.002524`, |coef| `0.002524`
- `lag_10__CT_shots_fired_sum`: coefficient `-0.002466`, |coef| `0.002466`
- `lag_00__T_kills_last_3s`: coefficient `-0.002432`, |coef| `0.002432`
- `lag_01__CT_place_VENTS`: coefficient `0.002390`, |coef| `0.002390`
- `lag_12__CT_place_VENTS`: coefficient `-0.002105`, |coef| `0.002105`
- `lag_11__CT_place_VENTS`: coefficient `-0.001949`, |coef| `0.001949`
- `lag_00__T4__is_walking`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_13__CT_place_ADMIN`: coefficient `0.001850`, |coef| `0.001850`
- `lag_10__CT_place_HELL`: coefficient `-0.001836`, |coef| `0.001836`
- `lag_00__T_damage_last_5s`: coefficient `-0.001790`, |coef| `0.001790`
- `lag_00__CT2__flash_duration`: coefficient `0.001780`, |coef| `0.001780`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.001780` (raises CT win probability)
- `lag_08__CT5__utility_total`: coefficient `0.001503` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.001435` (raises CT win probability)
- `lag_08__CT5__molly`: coefficient `0.001284` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001239` (raises CT win probability)
- `lag_08__CT5__smoke`: coefficient `0.001163` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001153` (raises CT win probability)
- `lag_09__CT5__utility_total`: coefficient `0.001151` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001126` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.001109` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_VENTS`: coefficient `0.003875` (raises CT win probability)
- `lag_15__CT_place_VENTS`: coefficient `-0.003342` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002817` (raises CT win probability)
- `lag_08__CT_place_VENTS`: coefficient `0.002760` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002524` (raises CT win probability)
- `lag_10__CT_shots_fired_sum`: coefficient `-0.002466` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002432` (lowers CT win probability)
- `lag_01__CT_place_VENTS`: coefficient `0.002390` (raises CT win probability)
- `lag_12__CT_place_VENTS`: coefficient `-0.002105` (lowers CT win probability)
- `lag_11__CT_place_VENTS`: coefficient `-0.001949` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `53749`, seconds `90.50`, LSTM delta `-0.2606`

Top all feature movements:
- `lag_00__CT_place_VENTS`: contribution `-0.032517`
- `lag_15__CT_place_VENTS`: contribution `-0.028044`
- `lag_08__CT_place_VENTS`: contribution `-0.023160`
- `lag_11__CT_place_VENTS`: contribution `-0.016357`
- `lag_13__CT_place_ADMIN`: contribution `-0.012853`

Top utility-only movements:
- `lag_08__CT5__utility_total`: contribution `-0.004260`
- `lag_00__CT1__molly`: contribution `-0.003572`
- `lag_08__CT5__molly`: contribution `-0.003185`

### tick `52789`, seconds `75.50`, LSTM delta `+0.2222`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `+0.034260`
- `lag_06__T_place_DECON`: contribution `+0.027346`
- `lag_13__T_place_MINI`: contribution `+0.023380`
- `lag_12__T_place_VENTS`: contribution `+0.015337`
- `lag_08__T_place_MINI`: contribution `+0.012755`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52085`, seconds `64.50`, LSTM delta `-0.2129`

Top all feature movements:
- `lag_07__CT_place_LOCKERROOM`: contribution `-0.016758`
- `lag_04__CT2__is_scoped`: contribution `-0.010462`
- `lag_10__CT_shots_fired_sum`: contribution `-0.008565`
- `lag_06__CT_shots_fired_sum`: contribution `-0.007705`
- `lag_00__T_kills_last_3s`: contribution `-0.007704`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.003921`
- `lag_10__CT2__flash_duration`: contribution `-0.003061`

### tick `52053`, seconds `64.00`, LSTM delta `-0.1788`

Top all feature movements:
- `lag_06__CT_place_LOCKERROOM`: contribution `-0.018483`
- `lag_01__T_place_MINI`: contribution `-0.013862`
- `lag_05__CT_place_ADMIN`: contribution `-0.010597`
- `lag_00__damage_diff_last_5s`: contribution `-0.010532`
- `lag_00__CT2__flash_duration`: contribution `-0.008893`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.008893`
- `lag_09__CT2__flash_duration`: contribution `-0.003861`

### tick `52437`, seconds `70.00`, LSTM delta `+0.1680`

Top all feature movements:
- `lag_13__T_place_MINI`: contribution `+0.023380`
- `lag_05__T_place_VENTS`: contribution `+0.014192`
- `lag_02__CT_place_LOCKERROOM`: contribution `+0.012801`
- `lag_09__T_place_MINI`: contribution `+0.011694`
- `lag_04__CT2__is_scoped`: contribution `+0.010462`

Top utility-only movements:
- No utility movement among the top local contributors.
