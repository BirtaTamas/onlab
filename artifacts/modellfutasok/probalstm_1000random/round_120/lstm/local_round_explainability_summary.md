# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `68204`, seconds `15.00`, LSTM `0.2164`, delta `-0.2454`
- tick `71628`, seconds `68.50`, LSTM `0.1016`, delta `-0.2229`
- tick `72940`, seconds `89.00`, LSTM `0.2469`, delta `+0.1620`
- tick `73132`, seconds `92.00`, LSTM `0.2224`, delta `-0.0499`
- tick `69324`, seconds `32.50`, LSTM `0.2402`, delta `-0.0477`
- tick `73292`, seconds `94.50`, LSTM `0.1748`, delta `-0.0469`
- tick `68524`, seconds `20.00`, LSTM `0.3715`, delta `+0.0429`
- tick `73228`, seconds `93.50`, LSTM `0.2615`, delta `+0.0415`
- tick `73260`, seconds `94.00`, LSTM `0.2218`, delta `-0.0398`
- tick `68556`, seconds `20.50`, LSTM `0.3325`, delta `-0.0390`

## Top 15 local ridge features

- `lag_12__T_place_UPSTAIRS`: coefficient `0.003634`, |coef| `0.003634`
- `lag_00__kill_diff_last_3s`: coefficient `0.003489`, |coef| `0.003489`
- `lag_00__T_kills_last_3s`: coefficient `-0.003180`, |coef| `0.003180`
- `lag_00__CT1__duck_amount`: coefficient `-0.002658`, |coef| `0.002658`
- `lag_07__T2__flash_duration`: coefficient `0.002381`, |coef| `0.002381`
- `lag_00__damage_diff_last_5s`: coefficient `0.002337`, |coef| `0.002337`
- `lag_13__CT2__duck_amount`: coefficient `0.002328`, |coef| `0.002328`
- `lag_00__CT1__molly`: coefficient `0.002020`, |coef| `0.002020`
- `lag_00__CT1__alive`: coefficient `0.001983`, |coef| `0.001983`
- `lag_00__T_damage_last_5s`: coefficient `-0.001938`, |coef| `0.001938`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_05__T_place_TRAMP`: coefficient `0.001800`, |coef| `0.001800`
- `lag_00__T3__is_walking`: coefficient `-0.001780`, |coef| `0.001780`
- `lag_00__CT1__utility_total`: coefficient `0.001773`, |coef| `0.001773`
- `lag_00__CT1__hp`: coefficient `0.001772`, |coef| `0.001772`

## Top 10 utility ridge features

- `lag_07__T2__flash_duration`: coefficient `0.002381` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.002020` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001773` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.001739` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001660` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.001523` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `-0.001464` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.001357` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001333` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.001279` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_UPSTAIRS`: coefficient `0.003634` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003489` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003180` (lowers CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `-0.002658` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002337` (raises CT win probability)
- `lag_13__CT2__duck_amount`: coefficient `0.002328` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001983` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001938` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001936` (lowers CT win probability)
- `lag_05__T_place_TRAMP`: coefficient `0.001800` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `68204`, seconds `15.00`, LSTM delta `-0.2454`

Top all feature movements:
- `lag_12__T_place_UPSTAIRS`: contribution `-0.061292`
- `lag_00__T_kills_last_3s`: contribution `-0.010074`
- `lag_00__kill_diff_last_3s`: contribution `-0.008398`
- `lag_13__CT_place_LIBRARY`: contribution `-0.007959`
- `lag_05__CT3__flash_duration`: contribution `-0.007894`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `-0.007894`
- `lag_09__T_utility_damage_last_5s`: contribution `-0.004171`

### tick `71628`, seconds `68.50`, LSTM delta `-0.2229`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.010074`
- `lag_13__CT2__duck_amount`: contribution `-0.008867`
- `lag_00__kill_diff_last_3s`: contribution `-0.008398`
- `lag_00__CT1__duck_amount`: contribution `-0.006507`
- `lag_14__CT2__duck_amount`: contribution `-0.006437`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.005028`

### tick `72940`, seconds `89.00`, LSTM delta `+0.1620`

Top all feature movements:
- `lag_07__T2__flash_duration`: contribution `+0.015734`
- `lag_00__T2__flash_duration`: contribution `+0.008805`
- `lag_00__kill_diff_last_3s`: contribution `+0.008398`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.006138`
- `lag_00__damage_diff_last_5s`: contribution `+0.005219`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.015734`
- `lag_00__T2__flash_duration`: contribution `+0.008805`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.006138`
- `lag_01__CT_A_site_active_infernos`: contribution `+0.005168`
- `lag_10__T_A_site_active_infernos`: contribution `+0.003806`

### tick `73132`, seconds `92.00`, LSTM delta `-0.0499`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008398`
- `lag_06__T2__flash_duration`: contribution `-0.006693`
- `lag_00__CT4__duck_amount`: contribution `-0.003993`
- `lag_07__T3__duck_amount`: contribution `-0.003749`
- `lag_00__CT_kills_last_3s`: contribution `-0.003717`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.006693`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.001775`

### tick `69324`, seconds `32.50`, LSTM delta `-0.0477`

Top all feature movements:
- `lag_00__T3__is_walking`: contribution `-0.004134`
- `lag_09__CT1__is_walking`: contribution `-0.003571`
- `lag_02__T1__is_walking`: contribution `-0.003214`
- `lag_11__T5__is_walking`: contribution `-0.002852`
- `lag_06__CT3__is_walking`: contribution `+0.002583`

Top utility-only movements:
- `lag_00__CT2__smoke`: contribution `-0.001265`
- `lag_03__CT2__smoke`: contribution `-0.001209`
