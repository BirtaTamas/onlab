# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-liquid-bo3-73g5XINyWmLhIm1c4ZyOM7/gamerlegion-vs-liquid-m1-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `76107`, seconds `74.00`, LSTM `0.8033`, delta `+0.2368`
- tick `74635`, seconds `51.00`, LSTM `0.6992`, delta `-0.1639`
- tick `73835`, seconds `38.50`, LSTM `0.6705`, delta `+0.1356`
- tick `74347`, seconds `46.50`, LSTM `0.8554`, delta `+0.1222`
- tick `74091`, seconds `42.50`, LSTM `0.7859`, delta `+0.0920`
- tick `77035`, seconds `88.50`, LSTM `0.9427`, delta `+0.0916`
- tick `74219`, seconds `44.50`, LSTM `0.7001`, delta `-0.0747`
- tick `74507`, seconds `49.00`, LSTM `0.8855`, delta `+0.0655`
- tick `74475`, seconds `48.50`, LSTM `0.8199`, delta `-0.0637`
- tick `75243`, seconds `60.50`, LSTM `0.6782`, delta `-0.0575`

## Top 15 local ridge features

- `lag_11__T_place_ARAMP`: coefficient `0.004129`, |coef| `0.004129`
- `lag_08__T_place_ARAMP`: coefficient `-0.003397`, |coef| `0.003397`
- `lag_00__CT_kills_last_3s`: coefficient `0.003201`, |coef| `0.003201`
- `lag_00__kill_diff_last_3s`: coefficient `0.002990`, |coef| `0.002990`
- `lag_08__T_bomb_zone_count`: coefficient `0.002900`, |coef| `0.002900`
- `lag_11__T_place_LONGA`: coefficient `-0.002843`, |coef| `0.002843`
- `lag_01__T_bomb_zone_count`: coefficient `-0.002276`, |coef| `0.002276`
- `lag_05__T1__duck_amount`: coefficient `0.001761`, |coef| `0.001761`
- `lag_13__CT_place_MIDDLE`: coefficient `-0.001752`, |coef| `0.001752`
- `lag_13__CT_place_MIDDOORS`: coefficient `0.001738`, |coef| `0.001738`
- `lag_06__CT_place_MIDDOORS`: coefficient `-0.001707`, |coef| `0.001707`
- `lag_00__T1__has_bomb`: coefficient `-0.001630`, |coef| `0.001630`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001597`, |coef| `0.001597`
- `lag_00__T1__alive`: coefficient `-0.001556`, |coef| `0.001556`
- `lag_00__CT1__is_walking`: coefficient `0.001528`, |coef| `0.001528`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001597` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `-0.001292` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001154` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.001014` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.001008` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.000955` (raises CT win probability)
- `lag_08__CT_active_infernos`: coefficient `0.000828` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000817` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000814` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000809` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_ARAMP`: coefficient `0.004129` (raises CT win probability)
- `lag_08__T_place_ARAMP`: coefficient `-0.003397` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003201` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002990` (raises CT win probability)
- `lag_08__T_bomb_zone_count`: coefficient `0.002900` (raises CT win probability)
- `lag_11__T_place_LONGA`: coefficient `-0.002843` (lowers CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.002276` (lowers CT win probability)
- `lag_05__T1__duck_amount`: coefficient `0.001761` (raises CT win probability)
- `lag_13__CT_place_MIDDLE`: coefficient `-0.001752` (lowers CT win probability)
- `lag_13__CT_place_MIDDOORS`: coefficient `0.001738` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76107`, seconds `74.00`, LSTM delta `+0.2368`

Top all feature movements:
- `lag_11__T_place_ARAMP`: contribution `+0.037360`
- `lag_08__T_place_ARAMP`: contribution `+0.030740`
- `lag_08__T_bomb_zone_count`: contribution `+0.016882`
- `lag_01__T_bomb_zone_count`: contribution `+0.013247`
- `lag_11__T_place_LONGA`: contribution `+0.012115`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `74635`, seconds `51.00`, LSTM delta `-0.1639`

Top all feature movements:
- `lag_13__CT_place_HOLE`: contribution `-0.014460`
- `lag_00__kill_diff_last_3s`: contribution `-0.007198`
- `lag_13__CT_place_ARAMP`: contribution `-0.006320`
- `lag_00__T_shots_fired_sum`: contribution `-0.004020`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.003644`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73835`, seconds `38.50`, LSTM delta `+0.1356`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009241`
- `lag_00__kill_diff_last_3s`: contribution `+0.007198`
- `lag_01__T_place_SHORTSTAIRS`: contribution `+0.005981`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005232`
- `lag_00__T4__utility_total`: contribution `+0.003589`

Top utility-only movements:
- `lag_00__T4__utility_total`: contribution `+0.003589`
- `lag_00__T4__flash`: contribution `+0.002740`
- `lag_11__CT5__molly`: contribution `+0.001955`

### tick `74347`, seconds `46.50`, LSTM delta `+0.1222`

Top all feature movements:
- `lag_04__CT_place_HOLE`: contribution `+0.014767`
- `lag_08__CT_place_HOLE`: contribution `+0.012026`
- `lag_00__CT_kills_last_3s`: contribution `+0.009241`
- `lag_00__kill_diff_last_3s`: contribution `+0.007198`
- `lag_08__CT5__flash_duration`: contribution `+0.007054`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.007054`
- `lag_03__CT5__flash_duration`: contribution `+0.001484`

### tick `74091`, seconds `42.50`, LSTM delta `+0.0920`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.014592`
- `lag_07__CT1__is_scoped`: contribution `+0.005151`
- `lag_00__CT1__is_walking`: contribution `-0.003567`
- `lag_00__CT5__flash_duration`: contribution `+0.002418`
- `lag_06__T3__is_walking`: contribution `+0.002301`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.002418`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.002191`
