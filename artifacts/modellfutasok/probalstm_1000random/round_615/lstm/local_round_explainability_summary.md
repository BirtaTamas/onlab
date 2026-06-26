# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `90857`, seconds `62.50`, LSTM `0.6638`, delta `-0.2543`
- tick `92009`, seconds `80.50`, LSTM `0.7526`, delta `+0.2503`
- tick `90537`, seconds `57.50`, LSTM `0.7278`, delta `+0.2138`
- tick `90825`, seconds `62.00`, LSTM `0.9181`, delta `+0.1940`
- tick `90729`, seconds `60.50`, LSTM `0.6510`, delta `-0.1774`
- tick `88809`, seconds `30.50`, LSTM `0.5537`, delta `+0.1434`
- tick `92809`, seconds `93.00`, LSTM `0.9297`, delta `+0.0927`
- tick `90761`, seconds `61.00`, LSTM `0.7373`, delta `+0.0863`
- tick `90409`, seconds `55.50`, LSTM `0.5155`, delta `+0.0745`
- tick `90633`, seconds `59.00`, LSTM `0.7954`, delta `+0.0608`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007118`, |coef| `0.007118`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004205`, |coef| `0.004205`
- `lag_00__CT_velocity_mean`: coefficient `-0.003749`, |coef| `0.003749`
- `lag_12__T_bomb_zone_count`: coefficient `-0.003579`, |coef| `0.003579`
- `lag_12__T_duck_amount_mean`: coefficient `-0.003552`, |coef| `0.003552`
- `lag_00__kill_diff_last_3s`: coefficient `0.003348`, |coef| `0.003348`
- `lag_00__CT_kills_last_3s`: coefficient `0.003251`, |coef| `0.003251`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002944`, |coef| `0.002944`
- `lag_12__T3__duck_amount`: coefficient `-0.002503`, |coef| `0.002503`
- `lag_14__CT_kills_last_3s`: coefficient `-0.002332`, |coef| `0.002332`
- `lag_02__CT_duck_amount_mean`: coefficient `-0.002328`, |coef| `0.002328`
- `lag_04__T_B_site_active_infernos`: coefficient `-0.002285`, |coef| `0.002285`
- `lag_12__T3__has_bomb`: coefficient `-0.002267`, |coef| `0.002267`
- `lag_03__CT_duck_amount_mean`: coefficient `0.002188`, |coef| `0.002188`
- `lag_00__T3__is_scoped`: coefficient `0.002185`, |coef| `0.002185`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004205` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.002285` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.001701` (lowers CT win probability)
- `lag_15__T_flash_alpha_mean`: coefficient `-0.001418` (lowers CT win probability)
- `lag_14__T_flash_alpha_mean`: coefficient `-0.001389` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001356` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.001219` (lowers CT win probability)
- `lag_04__active_infernos_total`: coefficient `-0.001181` (lowers CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.001021` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.000906` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007118` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.003749` (lowers CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `-0.003579` (lowers CT win probability)
- `lag_12__T_duck_amount_mean`: coefficient `-0.003552` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003348` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003251` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002944` (raises CT win probability)
- `lag_12__T3__duck_amount`: coefficient `-0.002503` (lowers CT win probability)
- `lag_14__CT_kills_last_3s`: coefficient `-0.002332` (lowers CT win probability)
- `lag_02__CT_duck_amount_mean`: coefficient `-0.002328` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `90857`, seconds `62.50`, LSTM delta `-0.2543`

Top all feature movements:
- `lag_12__T_shots_fired_sum`: contribution `-0.023923`
- `lag_12__T5__shots_fired`: contribution `-0.018441`
- `lag_00__CT_shots_fired_sum`: contribution `-0.014315`
- `lag_00__T3__is_scoped`: contribution `-0.014018`
- `lag_03__CT_place_TSIDEUPPER`: contribution `-0.010416`

Top utility-only movements:
- `lag_14__T4__flash_duration`: contribution `-0.005478`

### tick `92009`, seconds `80.50`, LSTM delta `+0.2503`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025513`
- `lag_12__T_bomb_zone_count`: contribution `+0.020837`
- `lag_12__T_duck_amount_mean`: contribution `+0.020656`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010225`
- `lag_02__CT_duck_amount_mean`: contribution `+0.010140`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025513`
- `lag_04__T_B_site_active_infernos`: contribution `+0.006462`
- `lag_04__T_active_infernos`: contribution `+0.003542`

### tick `90537`, seconds `57.50`, LSTM delta `+0.2138`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `+0.017655`
- `lag_02__T_shots_fired_sum`: contribution `+0.015442`
- `lag_02__T5__shots_fired`: contribution `+0.015236`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010225`
- `lag_12__T_shots_fired_sum`: contribution `+0.010073`

Top utility-only movements:
- `lag_09__T4__flash_duration`: contribution `+0.005181`
- `lag_04__T4__flash_duration`: contribution `+0.004868`
- `lag_13__CT5__flash_duration`: contribution `+0.003894`

### tick `90825`, seconds `62.00`, LSTM delta `+0.1940`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `+0.014018`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010225`
- `lag_00__CT_kills_last_3s`: contribution `+0.009385`
- `lag_11__T5__shots_fired`: contribution `+0.009050`
- `lag_04__T3__is_scoped`: contribution `+0.008295`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90729`, seconds `60.50`, LSTM delta `-0.1774`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.022430`
- `lag_08__T5__shots_fired`: contribution `-0.017962`
- `lag_00__kill_diff_last_3s`: contribution `-0.016115`
- `lag_00__T3__is_scoped`: contribution `-0.014018`
- `lag_00__CT_kills_last_3s`: contribution `-0.009385`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.004781`
- `lag_15__T4__flash_duration`: contribution `-0.003517`
