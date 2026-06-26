# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `3`

## Largest probability jumps

- tick `15973`, seconds `0.50`, LSTM `0.0185`, delta `-0.0311`
- tick `21957`, seconds `94.00`, LSTM `0.0436`, delta `-0.0304`
- tick `21829`, seconds `92.00`, LSTM `0.0511`, delta `+0.0301`
- tick `16965`, seconds `16.00`, LSTM `0.0254`, delta `-0.0146`
- tick `22021`, seconds `95.00`, LSTM `0.0318`, delta `-0.0131`
- tick `19525`, seconds `56.00`, LSTM `0.0094`, delta `-0.0118`
- tick `21861`, seconds `92.50`, LSTM `0.0616`, delta `+0.0105`
- tick `21893`, seconds `93.00`, LSTM `0.0716`, delta `+0.0100`
- tick `18405`, seconds `38.50`, LSTM `0.0136`, delta `-0.0088`
- tick `16773`, seconds `13.00`, LSTM `0.0446`, delta `+0.0082`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000483`, |coef| `0.000483`
- `lag_00__CT_kills_last_3s`: coefficient `0.000434`, |coef| `0.000434`
- `lag_03__CT3__is_walking`: coefficient `0.000329`, |coef| `0.000329`
- `lag_14__CT_place_TOPOFMID`: coefficient `0.000317`, |coef| `0.000317`
- `lag_01__CT_kills_last_3s`: coefficient `0.000310`, |coef| `0.000310`
- `lag_00__CT_velocity_mean`: coefficient `-0.000306`, |coef| `0.000306`
- `lag_14__T_B_site_active_infernos`: coefficient `0.000287`, |coef| `0.000287`
- `lag_01__kill_diff_last_3s`: coefficient `0.000287`, |coef| `0.000287`
- `lag_12__CT4__duck_amount`: coefficient `0.000286`, |coef| `0.000286`
- `lag_07__T2__is_walking`: coefficient `0.000277`, |coef| `0.000277`
- `lag_04__CT3__duck_amount`: coefficient `-0.000272`, |coef| `0.000272`
- `lag_00__damage_diff_last_5s`: coefficient `0.000259`, |coef| `0.000259`
- `lag_05__CT_place_RUINS`: coefficient `0.000257`, |coef| `0.000257`
- `lag_07__CT3__is_walking`: coefficient `-0.000251`, |coef| `0.000251`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000243`, |coef| `0.000243`

## Top 10 utility ridge features

- `lag_14__T_B_site_active_infernos`: coefficient `0.000287` (raises CT win probability)
- `lag_09__T1__smoke`: coefficient `-0.000204` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000192` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000190` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.000186` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000180` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000169` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000168` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000155` (raises CT win probability)
- `lag_08__T_active_infernos`: coefficient `0.000146` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000483` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000434` (raises CT win probability)
- `lag_03__CT3__is_walking`: coefficient `0.000329` (raises CT win probability)
- `lag_14__CT_place_TOPOFMID`: coefficient `0.000317` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.000310` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000306` (lowers CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.000287` (raises CT win probability)
- `lag_12__CT4__duck_amount`: coefficient `0.000286` (raises CT win probability)
- `lag_07__T2__is_walking`: coefficient `0.000277` (raises CT win probability)
- `lag_04__CT3__duck_amount`: coefficient `-0.000272` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15973`, seconds `0.50`, LSTM delta `-0.0311`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001163`
- `lag_00__CT_velocity_mean`: contribution `-0.001043`
- `lag_01__T_place_TSPAWN`: contribution `-0.001034`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000898`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000889`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000604`
- `lag_01__utility_inv_diff`: contribution `-0.000444`
- `lag_01__molly_inv_diff`: contribution `-0.000388`
- `lag_01__T_smoke_inv`: contribution `-0.000303`
- `lag_01__T_molly_inv`: contribution `-0.000237`

### tick `21957`, seconds `94.00`, LSTM delta `-0.0304`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.001163`
- `lag_12__CT4__duck_amount`: contribution `-0.001052`
- `lag_00__CT_place_RUINS`: contribution `-0.000676`
- `lag_07__T2__is_walking`: contribution `-0.000637`
- `lag_07__CT3__is_walking`: contribution `-0.000600`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.000543`

### tick `21829`, seconds `92.00`, LSTM delta `+0.0301`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.001253`
- `lag_00__kill_diff_last_3s`: contribution `+0.001163`
- `lag_14__CT_place_TOPOFMID`: contribution `+0.001150`
- `lag_04__CT3__duck_amount`: contribution `+0.001011`
- `lag_14__CT_place_ARCH`: contribution `+0.000889`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.000810`
- `lag_04__T_B_site_active_infernos`: contribution `+0.000477`
- `lag_09__T1__smoke`: contribution `+0.000440`
- `lag_14__T_active_infernos`: contribution `+0.000375`

### tick `16965`, seconds `16.00`, LSTM delta `-0.0146`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001091`
- `lag_05__CT_place_RUINS`: contribution `-0.000900`
- `lag_05__CT_place_BALCONY`: contribution `-0.000711`
- `lag_00__damage_diff_last_5s`: contribution `-0.000519`
- `lag_10__CT_place_RUINS`: contribution `-0.000473`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.001091`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.000436`

### tick `22021`, seconds `95.00`, LSTM delta `-0.0131`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.001253`
- `lag_00__kill_diff_last_3s`: contribution `-0.001163`
- `lag_03__CT3__is_walking`: contribution `-0.000786`
- `lag_07__T2__is_walking`: contribution `-0.000637`
- `lag_01__T_shots_fired_sum`: contribution `+0.000589`

Top utility-only movements:
- No utility movement among the top local contributors.
