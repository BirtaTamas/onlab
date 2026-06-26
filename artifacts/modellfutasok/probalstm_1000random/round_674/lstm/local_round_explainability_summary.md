# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `22940`, seconds `27.50`, LSTM `0.2157`, delta `-0.1723`
- tick `23132`, seconds `30.50`, LSTM `0.0966`, delta `-0.1657`
- tick `23004`, seconds `28.50`, LSTM `0.2835`, delta `+0.0906`
- tick `23100`, seconds `30.00`, LSTM `0.2623`, delta `-0.0728`
- tick `21212`, seconds `0.50`, LSTM `0.3097`, delta `-0.0502`
- tick `23292`, seconds `33.00`, LSTM `0.0237`, delta `-0.0463`
- tick `21340`, seconds `2.50`, LSTM `0.2377`, delta `-0.0409`
- tick `21436`, seconds `4.00`, LSTM `0.1559`, delta `-0.0388`
- tick `23036`, seconds `29.00`, LSTM `0.3218`, delta `+0.0383`
- tick `22044`, seconds `13.50`, LSTM `0.3004`, delta `+0.0343`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001625`, |coef| `0.001625`
- `lag_02__CT1__flash_duration`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000948`, |coef| `0.000948`
- `lag_07__T_flashes_last_5s`: coefficient `-0.000840`, |coef| `0.000840`
- `lag_07__T_place_TSIDELOWER`: coefficient `0.000835`, |coef| `0.000835`
- `lag_00__CT1__flash_duration`: coefficient `0.000808`, |coef| `0.000808`
- `lag_01__T4__shots_fired`: coefficient `0.000772`, |coef| `0.000772`
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000771`, |coef| `0.000771`
- `lag_00__T5__shots_fired`: coefficient `0.000753`, |coef| `0.000753`
- `lag_03__T4__shots_fired`: coefficient `-0.000715`, |coef| `0.000715`
- `lag_07__T_place_RAMP`: coefficient `-0.000710`, |coef| `0.000710`
- `lag_02__T2__flash_duration`: coefficient `-0.000709`, |coef| `0.000709`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.000666`, |coef| `0.000666`
- `lag_03__CT5__shots_fired`: coefficient `-0.000625`, |coef| `0.000625`
- `lag_05__CT1__flash_duration`: coefficient `0.000623`, |coef| `0.000623`

## Top 10 utility ridge features

- `lag_02__CT1__flash_duration`: coefficient `-0.001233` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `-0.000840` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000808` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.000709` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000623` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000608` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000592` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `-0.000547` (lowers CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `0.000534` (raises CT win probability)
- `lag_04__CT_he_last_5s`: coefficient `-0.000505` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.001625` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000948` (raises CT win probability)
- `lag_07__T_place_TSIDELOWER`: coefficient `0.000835` (raises CT win probability)
- `lag_01__T4__shots_fired`: coefficient `0.000772` (raises CT win probability)
- `lag_02__CT_place_UNKNOWN`: coefficient `0.000771` (raises CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.000753` (raises CT win probability)
- `lag_03__T4__shots_fired`: coefficient `-0.000715` (lowers CT win probability)
- `lag_07__T_place_RAMP`: coefficient `-0.000710` (lowers CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.000666` (lowers CT win probability)
- `lag_03__CT5__shots_fired`: coefficient `-0.000625` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22940`, seconds `27.50`, LSTM delta `-0.1723`

Top all feature movements:
- `lag_02__CT1__flash_duration`: contribution `-0.010200`
- `lag_07__T_flashes_last_5s`: contribution `-0.007614`
- `lag_00__CT1__flash_duration`: contribution `-0.006295`
- `lag_01__T4__shots_fired`: contribution `-0.005724`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.005004`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.010200`
- `lag_07__T_flashes_last_5s`: contribution `-0.007614`
- `lag_00__CT1__flash_duration`: contribution `-0.006295`
- `lag_02__T2__flash_duration`: contribution `-0.003998`
- `lag_06__T3__flash_duration`: contribution `-0.003592`

### tick `23132`, seconds `30.50`, LSTM delta `-0.1657`

Top all feature movements:
- `lag_00__T5__shots_fired`: contribution `-0.008796`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006587`
- `lag_13__T_flashes_last_5s`: contribution `-0.005361`
- `lag_03__T_shots_fired_sum`: contribution `-0.004900`
- `lag_03__T_flashes_last_5s`: contribution `-0.004840`

Top utility-only movements:
- `lag_13__T_flashes_last_5s`: contribution `-0.005361`
- `lag_03__T_flashes_last_5s`: contribution `-0.004840`
- `lag_06__CT1__flash_duration`: contribution `-0.004735`
- `lag_08__CT1__flash_duration`: contribution `-0.004526`
- `lag_12__T3__flash_duration`: contribution `-0.003476`

### tick `23004`, seconds `28.50`, LSTM delta `+0.0906`

Top all feature movements:
- `lag_02__CT1__flash_duration`: contribution `+0.009603`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007905`
- `lag_07__T_place_TSIDELOWER`: contribution `+0.006259`
- `lag_03__T4__shots_fired`: contribution `+0.005303`
- `lag_09__T_flashes_last_5s`: contribution `+0.004270`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.009603`
- `lag_09__T_flashes_last_5s`: contribution `+0.004270`
- `lag_08__T3__flash_duration`: contribution `+0.003331`
- `lag_04__CT1__flash_duration`: contribution `+0.002616`
- `lag_04__T2__flash_duration`: contribution `+0.001952`

### tick `23100`, seconds `30.00`, LSTM delta `-0.0728`

Top all feature movements:
- `lag_05__CT1__flash_duration`: contribution `-0.004854`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004611`
- `lag_06__T4__shots_fired`: contribution `-0.003275`
- `lag_03__T_shots_fired_sum`: contribution `+0.003267`
- `lag_02__T_shots_fired_sum`: contribution `-0.002817`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.004854`
- `lag_12__T_flashes_last_5s`: contribution `-0.002650`
- `lag_11__T3__flash_duration`: contribution `-0.001985`
- `lag_07__CT1__flash_duration`: contribution `-0.001965`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.001621`

### tick `21212`, seconds `0.50`, LSTM delta `-0.0502`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.016864`
- `lag_00__CT_he_last_5s`: contribution `-0.008851`
- `lag_00__T_velocity_mean`: contribution `-0.001235`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000751`
- `lag_00__CT_velocity_mean`: contribution `-0.000681`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.008851`
- `lag_01__molly_inv_diff`: contribution `-0.000368`
- `lag_01__T1__utility_total`: contribution `-0.000283`
- `lag_01__T_smoke_inv`: contribution `-0.000251`
- `lag_01__T4__smoke`: contribution `-0.000243`
