# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `8`

## Largest probability jumps

- tick `50714`, seconds `30.00`, LSTM `0.8712`, delta `+0.2968`
- tick `50426`, seconds `25.50`, LSTM `0.6145`, delta `-0.2161`
- tick `50394`, seconds `25.00`, LSTM `0.8306`, delta `+0.1662`
- tick `50266`, seconds `23.00`, LSTM `0.6439`, delta `+0.1218`
- tick `51450`, seconds `41.50`, LSTM `0.8060`, delta `-0.1211`
- tick `51578`, seconds `43.50`, LSTM `0.9063`, delta `+0.1196`
- tick `50650`, seconds `29.00`, LSTM `0.6408`, delta `+0.0750`
- tick `50682`, seconds `29.50`, LSTM `0.5744`, delta `-0.0664`
- tick `49754`, seconds `15.00`, LSTM `0.5973`, delta `-0.0614`
- tick `51962`, seconds `49.50`, LSTM `0.9653`, delta `+0.0551`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002376`, |coef| `0.002376`
- `lag_09__CT_place_MAINHALL`: coefficient `-0.001896`, |coef| `0.001896`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001800`, |coef| `0.001800`
- `lag_07__CT_shots_fired_sum`: coefficient `-0.001725`, |coef| `0.001725`
- `lag_06__T5__shots_fired`: coefficient `0.001723`, |coef| `0.001723`
- `lag_00__damage_diff_last_5s`: coefficient `0.001703`, |coef| `0.001703`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001693`, |coef| `0.001693`
- `lag_06__T4__duck_amount`: coefficient `-0.001693`, |coef| `0.001693`
- `lag_14__CT_place_TSIDEUPPER`: coefficient `0.001617`, |coef| `0.001617`
- `lag_15__T_shots_fired_sum`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_00__T_kills_last_3s`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_00__CT_kills_last_3s`: coefficient `0.001474`, |coef| `0.001474`
- `lag_15__T5__shots_fired`: coefficient `-0.001435`, |coef| `0.001435`
- `lag_05__T_shots_fired_sum`: coefficient `-0.001425`, |coef| `0.001425`
- `lag_13__CT_place_MAINHALL`: coefficient `0.001328`, |coef| `0.001328`

## Top 10 utility ridge features

- `lag_09__CT2__flash`: coefficient `-0.000762` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.000729` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000683` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.000608` (raises CT win probability)
- `lag_09__CT2__utility_total`: coefficient `-0.000601` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `-0.000590` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.000571` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000561` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `-0.000558` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000524` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002376` (raises CT win probability)
- `lag_09__CT_place_MAINHALL`: coefficient `-0.001896` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001800` (raises CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.001725` (lowers CT win probability)
- `lag_06__T5__shots_fired`: coefficient `0.001723` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001703` (raises CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.001693` (lowers CT win probability)
- `lag_06__T4__duck_amount`: coefficient `-0.001693` (lowers CT win probability)
- `lag_14__CT_place_TSIDEUPPER`: coefficient `0.001617` (raises CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `-0.001529` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `50714`, seconds `30.00`, LSTM delta `+0.2968`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.021583`
- `lag_07__CT_shots_fired_sum`: contribution `+0.020377`
- `lag_15__T_shots_fired_sum`: contribution `+0.017190`
- `lag_15__T5__shots_fired`: contribution `+0.013236`
- `lag_08__T2__shots_fired`: contribution `+0.010350`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50426`, seconds `25.50`, LSTM delta `-0.2161`

Top all feature movements:
- `lag_06__T5__shots_fired`: contribution `-0.015890`
- `lag_00__CT_shots_fired_sum`: contribution `-0.012505`
- `lag_14__CT_place_TSIDEUPPER`: contribution `-0.012152`
- `lag_06__T_shots_fired_sum`: contribution `-0.011696`
- `lag_12__T_bomb_zone_count`: contribution `-0.007559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50394`, seconds `25.00`, LSTM delta `+0.1662`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `+0.016028`
- `lag_08__T_shots_fired_sum`: contribution `-0.013966`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012505`
- `lag_05__T5__shots_fired`: contribution `+0.009071`
- `lag_12__T_bomb_zone_count`: contribution `+0.007559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50266`, seconds `23.00`, LSTM delta `+0.1218`

Top all feature movements:
- `lag_14__CT_place_TSIDEUPPER`: contribution `+0.012152`
- `lag_09__T_bomb_zone_count`: contribution `-0.006918`
- `lag_01__T_shots_fired_sum`: contribution `+0.006409`
- `lag_00__kill_diff_last_3s`: contribution `+0.005719`
- `lag_09__CT_place_TSIDEUPPER`: contribution `+0.005316`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `51450`, seconds `41.50`, LSTM delta `-0.1211`

Top all feature movements:
- `lag_09__CT_place_MAINHALL`: contribution `-0.015695`
- `lag_00__kill_diff_last_3s`: contribution `-0.005719`
- `lag_02__T4__duck_amount`: contribution `-0.004872`
- `lag_00__T_kills_last_3s`: contribution `-0.004783`
- `lag_05__T4__duck_amount`: contribution `-0.003905`

Top utility-only movements:
- No utility movement among the top local contributors.
