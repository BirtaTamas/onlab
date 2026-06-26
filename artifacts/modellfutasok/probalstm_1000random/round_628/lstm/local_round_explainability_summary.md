# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `122655`, seconds `97.00`, LSTM `0.1537`, delta `-0.3562`
- tick `121695`, seconds `82.00`, LSTM `0.5205`, delta `-0.1781`
- tick `121791`, seconds `83.50`, LSTM `0.6532`, delta `+0.1410`
- tick `121663`, seconds `81.50`, LSTM `0.6986`, delta `+0.1402`
- tick `121855`, seconds `84.50`, LSTM `0.5368`, delta `-0.1349`
- tick `123903`, seconds `116.50`, LSTM `0.1155`, delta `+0.0829`
- tick `122271`, seconds `91.00`, LSTM `0.5486`, delta `-0.0708`
- tick `122687`, seconds `97.50`, LSTM `0.0950`, delta `-0.0587`
- tick `122399`, seconds `93.00`, LSTM `0.4930`, delta `-0.0450`
- tick `122079`, seconds `88.00`, LSTM `0.5897`, delta `+0.0437`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.005296`, |coef| `0.005296`
- `lag_00__kill_diff_last_3s`: coefficient `0.003356`, |coef| `0.003356`
- `lag_12__CT_place_BACKOFB`: coefficient `-0.003239`, |coef| `0.003239`
- `lag_00__CT_place_BACKOFB`: coefficient `0.003059`, |coef| `0.003059`
- `lag_00__T_kills_last_3s`: coefficient `-0.002735`, |coef| `0.002735`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `0.002488`, |coef| `0.002488`
- `lag_00__CT_velocity_mean`: coefficient `-0.002302`, |coef| `0.002302`
- `lag_03__T_A_site_active_infernos`: coefficient `0.002118`, |coef| `0.002118`
- `lag_03__CT3__duck_amount`: coefficient `0.002058`, |coef| `0.002058`
- `lag_03__T_B_site_active_infernos`: coefficient `0.002048`, |coef| `0.002048`
- `lag_00__CT3__alive`: coefficient `0.002037`, |coef| `0.002037`
- `lag_00__CT3__hp`: coefficient `0.002009`, |coef| `0.002009`
- `lag_00__CT3__armor`: coefficient `0.001926`, |coef| `0.001926`
- `lag_00__damage_diff_last_5s`: coefficient `0.001913`, |coef| `0.001913`
- `lag_01__T_utility_damage_last_5s`: coefficient `0.001908`, |coef| `0.001908`

## Top 10 utility ridge features

- `lag_03__T_A_site_active_infernos`: coefficient `0.002118` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.002048` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `0.001908` (raises CT win probability)
- `lag_13__T3__smoke`: coefficient `0.001654` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.001493` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.001474` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `0.001431` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001308` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.001206` (raises CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `0.001116` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.005296` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003356` (raises CT win probability)
- `lag_12__CT_place_BACKOFB`: coefficient `-0.003239` (lowers CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.003059` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002735` (lowers CT win probability)
- `lag_13__CT_place_TSIDEUPPER`: coefficient `0.002488` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002302` (lowers CT win probability)
- `lag_03__CT3__duck_amount`: coefficient `0.002058` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.002037` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.002009` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122655`, seconds `97.00`, LSTM delta `-0.3562`

Top all feature movements:
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.039808`
- `lag_12__CT_place_BACKOFB`: contribution `-0.018492`
- `lag_00__CT_place_BACKOFB`: contribution `-0.017465`
- `lag_00__T_kills_last_3s`: contribution `-0.008665`
- `lag_13__CT_place_ENTRANCE`: contribution `-0.008599`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.006303`
- `lag_03__T_B_site_active_infernos`: contribution `-0.005791`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.004359`

### tick `121695`, seconds `82.00`, LSTM delta `-0.1781`

Top all feature movements:
- `lag_11__CT_place_TSTAIRS`: contribution `-0.029055`
- `lag_04__CT_place_TSTAIRS`: contribution `-0.025551`
- `lag_00__CT_shots_fired_sum`: contribution `-0.012616`
- `lag_00__T_kills_last_3s`: contribution `-0.008665`
- `lag_14__CT_place_TSPAWN`: contribution `-0.008333`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.006303`
- `lag_03__T_B_site_active_infernos`: contribution `-0.005791`
- `lag_03__T_active_infernos`: contribution `-0.003069`

### tick `121791`, seconds `83.50`, LSTM delta `+0.1410`

Top all feature movements:
- `lag_07__CT_place_TSTAIRS`: contribution `+0.031277`
- `lag_14__CT_place_TSTAIRS`: contribution `+0.021018`
- `lag_14__CT_place_TSPAWN`: contribution `+0.008333`
- `lag_00__kill_diff_last_3s`: contribution `+0.008078`
- `lag_03__CT_shots_fired_sum`: contribution `+0.007654`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.004359`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002627`

### tick `121663`, seconds `81.50`, LSTM delta `+0.1402`

Top all feature movements:
- `lag_10__CT_place_TSTAIRS`: contribution `+0.029773`
- `lag_03__CT_place_TSTAIRS`: contribution `+0.029584`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.009017`
- `lag_00__kill_diff_last_3s`: contribution `+0.008078`
- `lag_13__CT_place_TMAIN`: contribution `+0.005565`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121855`, seconds `84.50`, LSTM delta `-0.1349`

Top all feature movements:
- `lag_09__CT_place_TSTAIRS`: contribution `-0.038141`
- `lag_00__kill_diff_last_3s`: contribution `-0.016156`
- `lag_00__T_kills_last_3s`: contribution `-0.008665`
- `lag_00__CT_kills_last_3s`: contribution `-0.004426`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.002997`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.002997`
