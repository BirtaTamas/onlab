# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `48399`, seconds `55.50`, LSTM `0.0730`, delta `-0.2847`
- tick `48143`, seconds `51.50`, LSTM `0.1003`, delta `-0.2572`
- tick `47247`, seconds `37.50`, LSTM `0.2204`, delta `-0.2559`
- tick `48335`, seconds `54.50`, LSTM `0.3783`, delta `+0.2545`
- tick `47919`, seconds `48.00`, LSTM `0.2000`, delta `+0.1432`
- tick `47791`, seconds `46.00`, LSTM `0.0930`, delta `-0.0890`
- tick `48015`, seconds `49.50`, LSTM `0.3024`, delta `+0.0818`
- tick `48303`, seconds `54.00`, LSTM `0.1238`, delta `+0.0657`
- tick `47215`, seconds `37.00`, LSTM `0.4763`, delta `-0.0623`
- tick `45519`, seconds `10.50`, LSTM `0.4116`, delta `-0.0470`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002600`, |coef| `0.002600`
- `lag_00__T_kills_last_3s`: coefficient `-0.002593`, |coef| `0.002593`
- `lag_07__T_place_SNIPERSNEST`: coefficient `0.002340`, |coef| `0.002340`
- `lag_00__kill_diff_last_3s`: coefficient `0.002310`, |coef| `0.002310`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002009`, |coef| `0.002009`
- `lag_13__T_place_SNIPERSNEST`: coefficient `-0.002001`, |coef| `0.002001`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_14__CT_place_CONNECTOR`: coefficient `-0.001909`, |coef| `0.001909`
- `lag_01__CT_place_CONNECTOR`: coefficient `0.001797`, |coef| `0.001797`
- `lag_12__CT4__duck_amount`: coefficient `-0.001778`, |coef| `0.001778`
- `lag_07__T_shots_fired_sum`: coefficient `0.001655`, |coef| `0.001655`
- `lag_15__T3__duck_amount`: coefficient `0.001617`, |coef| `0.001617`
- `lag_07__T3__shots_fired`: coefficient `0.001592`, |coef| `0.001592`
- `lag_00__CT1__alive`: coefficient `0.001542`, |coef| `0.001542`
- `lag_11__T_place_SNIPERSNEST`: coefficient `-0.001538`, |coef| `0.001538`

## Top 10 utility ridge features

- `lag_06__T4__smoke`: coefficient `0.001299` (raises CT win probability)
- `lag_03__T4__molly`: coefficient `0.001282` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001171` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001163` (lowers CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000978` (lowers CT win probability)
- `lag_05__T4__smoke`: coefficient `0.000840` (raises CT win probability)
- `lag_02__T4__molly`: coefficient `0.000820` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000803` (lowers CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `0.000798` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000787` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002600` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002593` (lowers CT win probability)
- `lag_07__T_place_SNIPERSNEST`: coefficient `0.002340` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002310` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002009` (raises CT win probability)
- `lag_13__T_place_SNIPERSNEST`: coefficient `-0.002001` (lowers CT win probability)
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.001936` (lowers CT win probability)
- `lag_14__CT_place_CONNECTOR`: coefficient `-0.001909` (lowers CT win probability)
- `lag_01__CT_place_CONNECTOR`: coefficient `0.001797` (raises CT win probability)
- `lag_12__CT4__duck_amount`: coefficient `-0.001778` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `48399`, seconds `55.50`, LSTM delta `-0.2847`

Top all feature movements:
- `lag_15__T_place_SNIPERSNEST`: contribution `-0.023779`
- `lag_15__T_shots_fired_sum`: contribution `-0.021255`
- `lag_00__CT_shots_fired_sum`: contribution `-0.020940`
- `lag_15__T3__shots_fired`: contribution `-0.018478`
- `lag_00__T_shots_fired_sum`: contribution `-0.011696`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.008929`

### tick `48143`, seconds `51.50`, LSTM delta `-0.2572`

Top all feature movements:
- `lag_07__T_place_SNIPERSNEST`: contribution `-0.041586`
- `lag_11__T_place_SNIPERSNEST`: contribution `-0.027326`
- `lag_07__T_shots_fired_sum`: contribution `-0.024817`
- `lag_07__T3__shots_fired`: contribution `-0.019279`
- `lag_00__T_kills_last_3s`: contribution `-0.008216`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `-0.002792`

### tick `47247`, seconds `37.50`, LSTM delta `-0.2559`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008216`
- `lag_14__CT_place_CONNECTOR`: contribution `-0.006827`
- `lag_12__CT4__duck_amount`: contribution `-0.006529`
- `lag_01__CT_place_CONNECTOR`: contribution `-0.006424`
- `lag_01__T_shots_fired_sum`: contribution `-0.005627`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.003486`

### tick `48335`, seconds `54.50`, LSTM delta `+0.2545`

Top all feature movements:
- `lag_13__T_place_SNIPERSNEST`: contribution `+0.035556`
- `lag_13__T_shots_fired_sum`: contribution `+0.021034`
- `lag_13__T3__shots_fired`: contribution `+0.016309`
- `lag_01__CT_place_LADDER`: contribution `+0.013231`
- `lag_00__kill_diff_last_3s`: contribution `+0.011121`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `+0.004353`
- `lag_08__CT4__flash_duration`: contribution `+0.002792`

### tick `47919`, seconds `48.00`, LSTM delta `+0.1432`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.038988`
- `lag_00__T_place_SNIPERSNEST`: contribution `+0.034401`
- `lag_04__T_place_SNIPERSNEST`: contribution `+0.019045`
- `lag_00__T3__shots_fired`: contribution `+0.012140`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005584`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.003486`
