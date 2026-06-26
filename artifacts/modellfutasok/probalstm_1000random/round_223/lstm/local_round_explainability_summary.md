# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `25`

## Largest probability jumps

- tick `197742`, seconds `110.00`, LSTM `0.7030`, delta `+0.4994`
- tick `195790`, seconds `79.50`, LSTM `0.3018`, delta `-0.3897`
- tick `196110`, seconds `84.50`, LSTM `0.4365`, delta `+0.3684`
- tick `195918`, seconds `81.50`, LSTM `0.2064`, delta `-0.3162`
- tick `196654`, seconds `93.00`, LSTM `0.2904`, delta `-0.2931`
- tick `195854`, seconds `80.50`, LSTM `0.5335`, delta `+0.1973`
- tick `197806`, seconds `111.00`, LSTM `0.8509`, delta `+0.1786`
- tick `197486`, seconds `106.00`, LSTM `0.3360`, delta `+0.1584`
- tick `197550`, seconds `107.00`, LSTM `0.2821`, delta `-0.1450`
- tick `195566`, seconds `76.00`, LSTM `0.6946`, delta `+0.0935`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008251`, |coef| `0.008251`
- `lag_00__T_place_HUT`: coefficient `-0.007160`, |coef| `0.007160`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.007001`, |coef| `0.007001`
- `lag_00__kill_diff_last_3s`: coefficient `0.005649`, |coef| `0.005649`
- `lag_00__damage_diff_last_5s`: coefficient `0.005231`, |coef| `0.005231`
- `lag_08__CT_defusing_count`: coefficient `0.004521`, |coef| `0.004521`
- `lag_00__CT_kills_last_3s`: coefficient `0.004421`, |coef| `0.004421`
- `lag_07__T_place_HUT`: coefficient `-0.004275`, |coef| `0.004275`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.003807`, |coef| `0.003807`
- `lag_02__T_flash_alpha_mean`: coefficient `-0.003736`, |coef| `0.003736`
- `lag_00__T2__flash`: coefficient `-0.003397`, |coef| `0.003397`
- `lag_10__CT_defusing_count`: coefficient `0.003252`, |coef| `0.003252`
- `lag_07__CT_duck_amount_mean`: coefficient `0.003217`, |coef| `0.003217`
- `lag_00__CT_damage_last_5s`: coefficient `0.003193`, |coef| `0.003193`
- `lag_01__CT_kills_last_3s`: coefficient `0.002949`, |coef| `0.002949`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.007001` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.003807` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.003736` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.003397` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.002323` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002260` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.001911` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001882` (lowers CT win probability)
- `lag_01__T2__flash`: coefficient `-0.001719` (lowers CT win probability)
- `lag_02__T2__flash`: coefficient `-0.001711` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008251` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.007160` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005649` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005231` (raises CT win probability)
- `lag_08__CT_defusing_count`: coefficient `0.004521` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004421` (raises CT win probability)
- `lag_07__T_place_HUT`: coefficient `-0.004275` (lowers CT win probability)
- `lag_10__CT_defusing_count`: coefficient `0.003252` (raises CT win probability)
- `lag_07__CT_duck_amount_mean`: coefficient `0.003217` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003193` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `197742`, seconds `110.00`, LSTM delta `+0.4994`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.066745`
- `lag_08__CT_defusing_count`: contribution `+0.043825`
- `lag_00__T_flash_alpha_mean`: contribution `+0.042478`
- `lag_07__T_place_HUT`: contribution `+0.039849`
- `lag_07__CT_duck_amount_mean`: contribution `+0.019263`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.042478`
- `lag_00__T2__flash`: contribution `+0.010001`

### tick `195790`, seconds `79.50`, LSTM delta `-0.3897`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.027193`
- `lag_00__damage_diff_last_5s`: contribution `-0.020417`
- `lag_00__CT3__flash_duration`: contribution `-0.017952`
- `lag_00__T_shots_fired_sum`: contribution `-0.016374`
- `lag_00__T_kills_last_3s`: contribution `-0.016371`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.017952`
- `lag_03__CT3__flash_duration`: contribution `-0.014771`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004567`

### tick `196110`, seconds `84.50`, LSTM delta `+0.3684`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.032219`
- `lag_08__T_place_MINI`: contribution `+0.028652`
- `lag_03__T_place_MINI`: contribution `+0.028069`
- `lag_00__kill_diff_last_3s`: contribution `+0.027193`
- `lag_02__T_place_HUT`: contribution `+0.026084`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.007677`
- `lag_13__CT3__flash_duration`: contribution `+0.007189`
- `lag_06__CT2__flash_duration`: contribution `+0.006400`

### tick `195918`, seconds `81.50`, LSTM delta `-0.3162`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `-0.066745`
- `lag_09__T_place_HUT`: contribution `-0.024587`
- `lag_02__T_place_MINI`: contribution `-0.014866`
- `lag_00__kill_diff_last_3s`: contribution `-0.013596`
- `lag_00__damage_diff_last_5s`: contribution `-0.011802`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `-0.005629`
- `lag_04__CT3__flash_duration`: contribution `-0.004970`
- `lag_06__CT2__flash_duration`: contribution `-0.004576`
- `lag_00__CT2__flash_duration`: contribution `-0.003965`

### tick `196654`, seconds `93.00`, LSTM delta `-0.2931`

Top all feature movements:
- `lag_05__CT_place_VENDING`: contribution `-0.038205`
- `lag_12__CT_place_VENDING`: contribution `-0.034151`
- `lag_12__CT_place_TROPHY`: contribution `-0.028431`
- `lag_02__T_place_HUT`: contribution `+0.026084`
- `lag_09__T_place_HUT`: contribution `-0.024587`

Top utility-only movements:
- No utility movement among the top local contributors.
