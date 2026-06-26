# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `7198`, seconds `29.00`, LSTM `0.9588`, delta `+0.0530`
- tick `7134`, seconds `28.00`, LSTM `0.8973`, delta `-0.0382`
- tick `7038`, seconds `26.50`, LSTM `0.9259`, delta `+0.0355`
- tick `5374`, seconds `0.50`, LSTM `0.9153`, delta `+0.0228`
- tick `7070`, seconds `27.00`, LSTM `0.9480`, delta `+0.0220`
- tick `6974`, seconds `25.50`, LSTM `0.8863`, delta `+0.0176`
- tick `6750`, seconds `22.00`, LSTM `0.9106`, delta `+0.0168`
- tick `5438`, seconds `1.50`, LSTM `0.9361`, delta `+0.0165`
- tick `6558`, seconds `19.00`, LSTM `0.9200`, delta `-0.0148`
- tick `5982`, seconds `10.00`, LSTM `0.9469`, delta `+0.0146`

## Top 15 local ridge features

- `lag_07__T_place_TSIDELOWER`: coefficient `-0.000774`, |coef| `0.000774`
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.000623`, |coef| `0.000623`
- `lag_00__CT3__is_walking`: coefficient `-0.000521`, |coef| `0.000521`
- `lag_00__T_place_TSIDEUPPER`: coefficient `-0.000520`, |coef| `0.000520`
- `lag_06__T_place_TSIDELOWER`: coefficient `-0.000472`, |coef| `0.000472`
- `lag_00__damage_diff_last_5s`: coefficient `0.000451`, |coef| `0.000451`
- `lag_11__T_place_TSIDELOWER`: coefficient `-0.000438`, |coef| `0.000438`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000433`, |coef| `0.000433`
- `lag_10__CT3__is_walking`: coefficient `-0.000429`, |coef| `0.000429`
- `lag_08__CT5__is_walking`: coefficient `-0.000424`, |coef| `0.000424`
- `lag_05__T_place_TSIDELOWER`: coefficient `-0.000417`, |coef| `0.000417`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000403`, |coef| `0.000403`
- `lag_01__T4__duck_amount`: coefficient `-0.000403`, |coef| `0.000403`
- `lag_00__T_walking_count`: coefficient `-0.000398`, |coef| `0.000398`
- `lag_12__T_place_TSIDELOWER`: coefficient `-0.000394`, |coef| `0.000394`

## Top 10 utility ridge features

- `lag_06__CT_B_site_active_infernos`: coefficient `0.000214` (raises CT win probability)
- `lag_02__CT3__flash`: coefficient `-0.000178` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000174` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000165` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `0.000148` (raises CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000145` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000141` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000141` (raises CT win probability)
- `lag_06__utility_damage_diff_last_5s`: coefficient `0.000136` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `0.000132` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_TSIDELOWER`: coefficient `-0.000774` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.000623` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000521` (lowers CT win probability)
- `lag_00__T_place_TSIDEUPPER`: coefficient `-0.000520` (lowers CT win probability)
- `lag_06__T_place_TSIDELOWER`: coefficient `-0.000472` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000451` (raises CT win probability)
- `lag_11__T_place_TSIDELOWER`: coefficient `-0.000438` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000433` (raises CT win probability)
- `lag_10__CT3__is_walking`: coefficient `-0.000429` (lowers CT win probability)
- `lag_08__CT5__is_walking`: coefficient `-0.000424` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `7198`, seconds `29.00`, LSTM delta `+0.0530`

Top all feature movements:
- `lag_11__T_place_TSIDELOWER`: contribution `+0.001642`
- `lag_12__T_place_TSIDELOWER`: contribution `+0.001476`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001399`
- `lag_02__CT_place_SIDEENTRANCE`: contribution `+0.001382`
- `lag_00__T_place_TSIDEUPPER`: contribution `+0.001311`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7134`, seconds `28.00`, LSTM delta `-0.0382`

Top all feature movements:
- `lag_01__T4__duck_amount`: contribution `-0.001489`
- `lag_09__T_place_TSIDELOWER`: contribution `-0.001404`
- `lag_01__CT_shots_fired_sum`: contribution `-0.001204`
- `lag_10__CT3__is_walking`: contribution `-0.001024`
- `lag_08__CT5__is_walking`: contribution `-0.001015`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7038`, seconds `26.50`, LSTM delta `+0.0355`

Top all feature movements:
- `lag_07__T_place_TSIDELOWER`: contribution `+0.002902`
- `lag_06__T_place_TSIDELOWER`: contribution `+0.001769`
- `lag_00__CT3__is_walking`: contribution `+0.001243`
- `lag_08__CT5__is_walking`: contribution `+0.001015`
- `lag_07__T_place_TSIDEUPPER`: contribution `+0.000993`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5374`, seconds `0.50`, LSTM delta `+0.0228`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `+0.012410`
- `lag_00__T_velocity_mean`: contribution `+0.001267`
- `lag_01__T_velocity_mean`: contribution `-0.000505`
- `lag_00__CT_velocity_mean`: contribution `+0.000497`
- `lag_01__CT_closest_enemy_dist`: contribution `+0.000253`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `+0.000165`

### tick `7070`, seconds `27.00`, LSTM delta `+0.0220`

Top all feature movements:
- `lag_07__T_place_TSIDELOWER`: contribution `+0.002902`
- `lag_10__CT3__is_walking`: contribution `+0.001024`
- `lag_07__T_place_TSIDEUPPER`: contribution `+0.000993`
- `lag_15__T_place_RUINS`: contribution `+0.000953`
- `lag_01__CT_shots_fired_sum`: contribution `+0.000903`

Top utility-only movements:
- No utility movement among the top local contributors.
