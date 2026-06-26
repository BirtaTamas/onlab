# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m2-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `24687`, seconds `37.50`, LSTM `0.3578`, delta `-0.2601`
- tick `25135`, seconds `44.50`, LSTM `0.0911`, delta `-0.2536`
- tick `24655`, seconds `37.00`, LSTM `0.6179`, delta `+0.2366`
- tick `24399`, seconds `33.00`, LSTM `0.3748`, delta `-0.1364`
- tick `26223`, seconds `61.50`, LSTM `0.1830`, delta `+0.1261`
- tick `24527`, seconds `35.00`, LSTM `0.2762`, delta `+0.0998`
- tick `24463`, seconds `34.00`, LSTM `0.2219`, delta `-0.0953`
- tick `24783`, seconds `39.00`, LSTM `0.3834`, delta `+0.0887`
- tick `25007`, seconds `42.50`, LSTM `0.3084`, delta `-0.0791`
- tick `24559`, seconds `35.50`, LSTM `0.3498`, delta `+0.0737`

## Top 15 local ridge features

- `lag_00__T_place_TRUCK`: coefficient `-0.004188`, |coef| `0.004188`
- `lag_07__T_place_TRUCK`: coefficient `-0.003086`, |coef| `0.003086`
- `lag_00__kill_diff_last_3s`: coefficient `0.002143`, |coef| `0.002143`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001969`, |coef| `0.001969`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001889`, |coef| `0.001889`
- `lag_11__CT_place_JUNGLE`: coefficient `-0.001796`, |coef| `0.001796`
- `lag_06__T_place_TRUCK`: coefficient `0.001678`, |coef| `0.001678`
- `lag_00__CT_kills_last_3s`: coefficient `0.001638`, |coef| `0.001638`
- `lag_01__T_place_TRUCK`: coefficient `-0.001628`, |coef| `0.001628`
- `lag_03__T4__duck_amount`: coefficient `0.001547`, |coef| `0.001547`
- `lag_12__CT_place_JUNGLE`: coefficient `-0.001489`, |coef| `0.001489`
- `lag_03__T_place_TRUCK`: coefficient `-0.001359`, |coef| `0.001359`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001328`, |coef| `0.001328`
- `lag_13__T2__shots_fired`: coefficient `0.001280`, |coef| `0.001280`
- `lag_00__CT4__duck_amount`: coefficient `0.001272`, |coef| `0.001272`

## Top 10 utility ridge features

- `lag_06__utility_damage_diff_last_5s`: coefficient `-0.000669` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.000646` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.000622` (lowers CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000604` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000541` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000530` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000526` (raises CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000508` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000471` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000467` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_TRUCK`: coefficient `-0.004188` (lowers CT win probability)
- `lag_07__T_place_TRUCK`: coefficient `-0.003086` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002143` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001969` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001889` (raises CT win probability)
- `lag_11__CT_place_JUNGLE`: coefficient `-0.001796` (lowers CT win probability)
- `lag_06__T_place_TRUCK`: coefficient `0.001678` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001638` (raises CT win probability)
- `lag_01__T_place_TRUCK`: coefficient `-0.001628` (lowers CT win probability)
- `lag_03__T4__duck_amount`: coefficient `0.001547` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `24687`, seconds `37.50`, LSTM delta `-0.2601`

Top all feature movements:
- `lag_07__T_place_TRUCK`: contribution `-0.053593`
- `lag_04__T_place_TRUCK`: contribution `-0.008820`
- `lag_03__T4__duck_amount`: contribution `-0.005719`
- `lag_00__kill_diff_last_3s`: contribution `-0.005158`
- `lag_08__CT_place_SHOP`: contribution `-0.005123`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.002419`

### tick `25135`, seconds `44.50`, LSTM delta `-0.2536`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `-0.011955`
- `lag_00__CT_duck_amount_mean`: contribution `-0.011311`
- `lag_13__T2__shots_fired`: contribution `-0.009794`
- `lag_04__T_place_TRUCK`: contribution `+0.008820`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006841`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24655`, seconds `37.00`, LSTM delta `+0.2366`

Top all feature movements:
- `lag_06__T_place_TRUCK`: contribution `+0.029147`
- `lag_03__T_place_TRUCK`: contribution `+0.023598`
- `lag_03__T4__duck_amount`: contribution `+0.005719`
- `lag_00__kill_diff_last_3s`: contribution `+0.005158`
- `lag_07__CT_place_SHOP`: contribution `+0.005108`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.002329`

### tick `24399`, seconds `33.00`, LSTM delta `-0.1364`

Top all feature movements:
- `lag_11__CT_place_JUNGLE`: contribution `-0.011520`
- `lag_12__CT_place_STAIRS`: contribution `-0.008960`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005472`
- `lag_00__kill_diff_last_3s`: contribution `-0.005158`
- `lag_02__T_shots_fired_sum`: contribution `-0.004976`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.001900`

### tick `26223`, seconds `61.50`, LSTM delta `+0.1261`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `+0.072733`
- `lag_00__CT_duck_amount_mean`: contribution `+0.008974`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005472`
- `lag_00__kill_diff_last_3s`: contribution `+0.005158`
- `lag_00__CT_kills_last_3s`: contribution `+0.004728`

Top utility-only movements:
- No utility movement among the top local contributors.
