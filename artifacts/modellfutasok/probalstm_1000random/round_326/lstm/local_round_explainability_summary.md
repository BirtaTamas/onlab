# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `130473`, seconds `0.50`, LSTM `0.0150`, delta `-0.0279`
- tick `131721`, seconds `20.00`, LSTM `0.0198`, delta `-0.0065`
- tick `131145`, seconds `11.00`, LSTM `0.0245`, delta `+0.0063`
- tick `131177`, seconds `11.50`, LSTM `0.0304`, delta `+0.0059`
- tick `131785`, seconds `21.00`, LSTM `0.0112`, delta `-0.0045`
- tick `131401`, seconds `15.00`, LSTM `0.0260`, delta `-0.0042`
- tick `130505`, seconds `1.00`, LSTM `0.0110`, delta `-0.0040`
- tick `131753`, seconds `20.50`, LSTM `0.0158`, delta `-0.0040`
- tick `131081`, seconds `10.00`, LSTM `0.0174`, delta `+0.0033`
- tick `131465`, seconds `16.00`, LSTM `0.0293`, delta `+0.0032`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000245`, |coef| `0.000245`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000241`, |coef| `0.000241`
- `lag_00__CT_velocity_mean`: coefficient `-0.000196`, |coef| `0.000196`
- `lag_01__utility_inv_diff`: coefficient `0.000157`, |coef| `0.000157`
- `lag_01__armor_diff`: coefficient `0.000156`, |coef| `0.000156`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000148`, |coef| `0.000148`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000147`, |coef| `0.000147`
- `lag_01__T1__duck_amount`: coefficient `-0.000140`, |coef| `0.000140`
- `lag_01__CT_armor_sum`: coefficient `0.000138`, |coef| `0.000138`
- `lag_01__T3__has_bomb`: coefficient `-0.000137`, |coef| `0.000137`
- `lag_00__T_velocity_mean`: coefficient `-0.000136`, |coef| `0.000136`
- `lag_01__molly_inv_diff`: coefficient `0.000136`, |coef| `0.000136`
- `lag_01__centroid_distance_xy`: coefficient `-0.000133`, |coef| `0.000133`
- `lag_00__T1__smoke`: coefficient `0.000132`, |coef| `0.000132`
- `lag_01__equip_diff`: coefficient `0.000129`, |coef| `0.000129`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000157` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000136` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000132` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000127` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000118` (raises CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000111` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000108` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000094` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000094` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000083` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000245` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000241` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000196` (lowers CT win probability)
- `lag_01__armor_diff`: coefficient `0.000156` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000148` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000147` (lowers CT win probability)
- `lag_01__T1__duck_amount`: coefficient `-0.000140` (lowers CT win probability)
- `lag_01__CT_armor_sum`: coefficient `0.000138` (raises CT win probability)
- `lag_01__T3__has_bomb`: coefficient `-0.000137` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000136` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `130473`, seconds `0.50`, LSTM delta `-0.0279`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001170`
- `lag_01__T_place_TSPAWN`: contribution `-0.001067`
- `lag_00__CT_velocity_mean`: contribution `-0.000685`
- `lag_01__T1__duck_amount`: contribution `-0.000501`
- `lag_01__utility_inv_diff`: contribution `-0.000484`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000484`
- `lag_01__molly_inv_diff`: contribution `-0.000379`
- `lag_01__smoke_inv_diff`: contribution `-0.000300`
- `lag_01__flash_inv_diff`: contribution `-0.000287`
- `lag_00__T1__smoke`: contribution `-0.000285`

### tick `131721`, seconds `20.00`, LSTM delta `-0.0065`

Top all feature movements:
- `lag_04__CT_place_TOPOFMID`: contribution `-0.001027`
- `lag_04__CT_place_MIDDLE`: contribution `-0.000918`
- `lag_05__CT_place_TOPOFMID`: contribution `-0.000363`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.000311`
- `lag_05__CT_place_MIDDLE`: contribution `-0.000298`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131145`, seconds `11.00`, LSTM delta `+0.0063`

Top all feature movements:
- `lag_04__CT_place_MIDDLE`: contribution `+0.000306`
- `lag_09__CT_place_SNIPERSNEST`: contribution `+0.000300`
- `lag_06__CT_place_MIDDLE`: contribution `+0.000258`
- `lag_07__CT_place_MIDDLE`: contribution `+0.000238`
- `lag_02__CT_place_SNIPERSNEST`: contribution `+0.000208`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131177`, seconds `11.50`, LSTM delta `+0.0059`

Top all feature movements:
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.000363`
- `lag_05__CT_place_MIDDLE`: contribution `+0.000298`
- `lag_07__CT_place_MIDDLE`: contribution `+0.000238`
- `lag_01__T_place_TSPAWN`: contribution `+0.000229`
- `lag_01__CT_walking_count`: contribution `+0.000207`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131785`, seconds `21.00`, LSTM delta `-0.0045`

Top all feature movements:
- `lag_00__CT_place_SIDEALLEY`: contribution `-0.000991`
- `lag_06__CT_place_TOPOFMID`: contribution `-0.000861`
- `lag_06__CT_place_MIDDLE`: contribution `-0.000773`
- `lag_05__CT_place_TOPOFMID`: contribution `-0.000363`
- `lag_04__CT_place_TOPOFMID`: contribution `+0.000342`

Top utility-only movements:
- No utility movement among the top local contributors.
