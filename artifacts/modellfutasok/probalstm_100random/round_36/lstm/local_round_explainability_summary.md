# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `26`

## Largest probability jumps

- tick `204864`, seconds `80.50`, LSTM `0.9191`, delta `+0.2827`
- tick `204160`, seconds `69.50`, LSTM `0.8114`, delta `+0.2613`
- tick `202624`, seconds `45.50`, LSTM `0.8406`, delta `+0.1948`
- tick `204800`, seconds `79.50`, LSTM `0.6558`, delta `-0.1640`
- tick `201888`, seconds `34.00`, LSTM `0.6179`, delta `-0.1589`
- tick `203776`, seconds `63.50`, LSTM `0.6139`, delta `-0.1583`
- tick `200544`, seconds `13.00`, LSTM `0.6838`, delta `+0.1191`
- tick `201504`, seconds `28.00`, LSTM `0.6694`, delta `+0.0755`
- tick `201664`, seconds `30.50`, LSTM `0.7291`, delta `-0.0601`
- tick `201536`, seconds `28.50`, LSTM `0.7218`, delta `+0.0524`

## Top 15 local ridge features

- `lag_11__T_duck_amount_mean`: coefficient `-0.005625`, |coef| `0.005625`
- `lag_00__kill_diff_last_3s`: coefficient `0.005274`, |coef| `0.005274`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004821`, |coef| `0.004821`
- `lag_00__damage_diff_last_5s`: coefficient `0.004638`, |coef| `0.004638`
- `lag_00__CT_kills_last_3s`: coefficient `0.003873`, |coef| `0.003873`
- `lag_10__T_duck_amount_mean`: coefficient `-0.003718`, |coef| `0.003718`
- `lag_00__CT_damage_last_5s`: coefficient `0.003099`, |coef| `0.003099`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003009`, |coef| `0.003009`
- `lag_11__T5__duck_amount`: coefficient `-0.002707`, |coef| `0.002707`
- `lag_00__T_kills_last_3s`: coefficient `-0.002693`, |coef| `0.002693`
- `lag_02__T_place_SQUEAKY`: coefficient `-0.002547`, |coef| `0.002547`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002388`, |coef| `0.002388`
- `lag_12__T_duck_amount_mean`: coefficient `0.002322`, |coef| `0.002322`
- `lag_10__T5__duck_amount`: coefficient `-0.002303`, |coef| `0.002303`
- `lag_00__T_place_TROPHY`: coefficient `-0.002148`, |coef| `0.002148`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004821` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.002023` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.001725` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001600` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.001505` (lowers CT win probability)
- `lag_02__CT3__flash`: coefficient `-0.001344` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.001257` (raises CT win probability)
- `lag_15__CT2__flash`: coefficient `-0.001252` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.001210` (lowers CT win probability)
- `lag_10__CT1__smoke`: coefficient `-0.001093` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_duck_amount_mean`: coefficient `-0.005625` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005274` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004638` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003873` (raises CT win probability)
- `lag_10__T_duck_amount_mean`: coefficient `-0.003718` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003099` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003009` (raises CT win probability)
- `lag_11__T5__duck_amount`: coefficient `-0.002707` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002693` (lowers CT win probability)
- `lag_02__T_place_SQUEAKY`: coefficient `-0.002547` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `204864`, seconds `80.50`, LSTM delta `+0.2827`

Top all feature movements:
- `lag_11__T_duck_amount_mean`: contribution `+0.032715`
- `lag_00__T_flash_alpha_mean`: contribution `+0.029252`
- `lag_12__T_duck_amount_mean`: contribution `+0.013505`
- `lag_00__kill_diff_last_3s`: contribution `+0.012695`
- `lag_00__CT_kills_last_3s`: contribution `+0.011181`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.029252`
- `lag_12__T_B_site_active_infernos`: contribution `+0.005718`

### tick `204160`, seconds `69.50`, LSTM delta `+0.2613`

Top all feature movements:
- `lag_05__CT_place_LOCKERROOM`: contribution `+0.024856`
- `lag_11__CT_place_LOCKERROOM`: contribution `+0.022413`
- `lag_00__CT_shots_fired_sum`: contribution `+0.016726`
- `lag_11__T_duck_amount_mean`: contribution `+0.013728`
- `lag_06__T_place_CONTROL`: contribution `+0.013263`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `202624`, seconds `45.50`, LSTM delta `+0.1948`

Top all feature movements:
- `lag_02__T_place_SQUEAKY`: contribution `+0.015858`
- `lag_00__T_place_TROPHY`: contribution `+0.013619`
- `lag_00__kill_diff_last_3s`: contribution `+0.012695`
- `lag_00__CT_kills_last_3s`: contribution `+0.011181`
- `lag_11__T_duck_amount_mean`: contribution `+0.010905`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `204800`, seconds `79.50`, LSTM delta `-0.1640`

Top all feature movements:
- `lag_10__T_duck_amount_mean`: contribution `-0.021626`
- `lag_00__kill_diff_last_3s`: contribution `-0.012695`
- `lag_09__T_duck_amount_mean`: contribution `-0.011656`
- `lag_10__T5__duck_amount`: contribution `-0.008743`
- `lag_00__T_kills_last_3s`: contribution `-0.008532`

Top utility-only movements:
- `lag_10__T_B_site_active_infernos`: contribution `-0.004877`
- `lag_10__T_active_infernos`: contribution `-0.002618`

### tick `201888`, seconds `34.00`, LSTM delta `-0.1589`

Top all feature movements:
- `lag_12__T_place_DECON`: contribution `-0.017604`
- `lag_06__T_place_DECON`: contribution `-0.016128`
- `lag_07__CT_shots_fired_sum`: contribution `-0.015567`
- `lag_00__kill_diff_last_3s`: contribution `-0.012695`
- `lag_07__CT1__shots_fired`: contribution `-0.010890`

Top utility-only movements:
- No utility movement among the top local contributors.
