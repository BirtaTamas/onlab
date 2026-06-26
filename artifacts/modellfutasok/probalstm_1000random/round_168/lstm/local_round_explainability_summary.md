# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `18`

## Largest probability jumps

- tick `150405`, seconds `16.00`, LSTM `0.7713`, delta `+0.1577`
- tick `150725`, seconds `21.00`, LSTM `0.8918`, delta `+0.1315`
- tick `152933`, seconds `55.50`, LSTM `0.9388`, delta `+0.0552`
- tick `149893`, seconds `8.00`, LSTM `0.6582`, delta `-0.0290`
- tick `149989`, seconds `9.50`, LSTM `0.6742`, delta `+0.0288`
- tick `150917`, seconds `24.00`, LSTM `0.8677`, delta `+0.0264`
- tick `151525`, seconds `33.50`, LSTM `0.8680`, delta `+0.0242`
- tick `150437`, seconds `16.50`, LSTM `0.7472`, delta `-0.0241`
- tick `151365`, seconds `31.00`, LSTM `0.8212`, delta `-0.0237`
- tick `150245`, seconds `13.50`, LSTM `0.6430`, delta `-0.0218`

## Top 15 local ridge features

- `lag_12__CT1__is_scoped`: coefficient `0.001236`, |coef| `0.001236`
- `lag_04__CT_place_GARAGE`: coefficient `0.001072`, |coef| `0.001072`
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000918`, |coef| `0.000918`
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000875`, |coef| `0.000875`
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000839`, |coef| `0.000839`
- `lag_00__CT_kills_last_3s`: coefficient `0.000832`, |coef| `0.000832`
- `lag_09__CT_place_HEAVEN`: coefficient `0.000807`, |coef| `0.000807`
- `lag_13__T_place_OBSERVATION`: coefficient `0.000782`, |coef| `0.000782`
- `lag_08__T_place_TROPHY`: coefficient `0.000729`, |coef| `0.000729`
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.000728`, |coef| `0.000728`
- `lag_02__CT1__is_scoped`: coefficient `0.000726`, |coef| `0.000726`
- `lag_06__CT_place_HELL`: coefficient `-0.000723`, |coef| `0.000723`
- `lag_01__CT1__duck_amount`: coefficient `0.000708`, |coef| `0.000708`
- `lag_00__kill_diff_last_3s`: coefficient `0.000694`, |coef| `0.000694`

## Top 10 utility ridge features

- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000971` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000875` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000839` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.000728` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000641` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `-0.000583` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `0.000572` (raises CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.000557` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000479` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000474` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT1__is_scoped`: coefficient `0.001236` (raises CT win probability)
- `lag_04__CT_place_GARAGE`: coefficient `0.001072` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000918` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000832` (raises CT win probability)
- `lag_09__CT_place_HEAVEN`: coefficient `0.000807` (raises CT win probability)
- `lag_13__T_place_OBSERVATION`: coefficient `0.000782` (raises CT win probability)
- `lag_08__T_place_TROPHY`: coefficient `0.000729` (raises CT win probability)
- `lag_02__CT1__is_scoped`: coefficient `0.000726` (raises CT win probability)
- `lag_06__CT_place_HELL`: coefficient `-0.000723` (lowers CT win probability)
- `lag_01__CT1__duck_amount`: coefficient `0.000708` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `150405`, seconds `16.00`, LSTM delta `+0.1577`

Top all feature movements:
- `lag_04__CT_place_GARAGE`: contribution `+0.007707`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.006670`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.006176`
- `lag_12__CT1__is_scoped`: contribution `+0.005292`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.003932`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `+0.006670`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.006176`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.003932`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.003377`
- `lag_02__CT_active_infernos`: contribution `+0.002686`

### tick `150725`, seconds `21.00`, LSTM delta `+0.1315`

Top all feature movements:
- `lag_12__CT_A_site_active_infernos`: contribution `+0.005923`
- `lag_12__CT1__is_scoped`: contribution `+0.005292`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.005000`
- `lag_08__T_place_TROPHY`: contribution `+0.004621`
- `lag_09__CT_place_HEAVEN`: contribution `+0.004355`

Top utility-only movements:
- `lag_12__CT_A_site_active_infernos`: contribution `+0.005923`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.005000`
- `lag_12__CT_active_infernos`: contribution `+0.002565`

### tick `152933`, seconds `55.50`, LSTM delta `+0.0552`

Top all feature movements:
- `lag_13__T_place_OBSERVATION`: contribution `+0.013242`
- `lag_00__T_place_OBSERVATION`: contribution `+0.006401`
- `lag_09__CT_place_HEAVEN`: contribution `+0.004355`
- `lag_02__T_place_TROPHY`: contribution `-0.002568`
- `lag_03__T_place_CONTROL`: contribution `+0.002488`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149893`, seconds `8.00`, LSTM delta `-0.0290`

Top all feature movements:
- `lag_06__CT_place_HELL`: contribution `-0.011770`
- `lag_00__CT_place_ADMIN`: contribution `-0.003831`
- `lag_02__CT_place_HEAVEN`: contribution `-0.002355`
- `lag_06__CT_place_OUTSIDE`: contribution `-0.001906`
- `lag_05__CT_place_HELL`: contribution `-0.001065`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149989`, seconds `9.50`, LSTM delta `+0.0288`

Top all feature movements:
- `lag_06__CT_place_HELL`: contribution `+0.007846`
- `lag_09__CT_place_HELL`: contribution `+0.007025`
- `lag_02__CT_A_site_active_infernos`: contribution `-0.003088`
- `lag_06__CT_place_ADMIN`: contribution `+0.002462`
- `lag_02__CT_place_HEAVEN`: contribution `+0.002355`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `-0.003088`
- `lag_02__CT_active_infernos`: contribution `-0.001343`
- `lag_02__active_infernos_total`: contribution `-0.000546`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.000410`
