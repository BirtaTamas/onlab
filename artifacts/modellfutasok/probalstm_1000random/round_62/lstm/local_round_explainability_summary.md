# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `14`

## Largest probability jumps

- tick `118765`, seconds `105.50`, LSTM `0.7021`, delta `+0.2990`
- tick `117101`, seconds `79.50`, LSTM `0.8554`, delta `+0.2440`
- tick `118413`, seconds `100.00`, LSTM `0.5778`, delta `-0.2042`
- tick `119245`, seconds `113.00`, LSTM `0.9058`, delta `+0.2034`
- tick `117229`, seconds `81.50`, LSTM `0.6657`, delta `-0.1820`
- tick `116301`, seconds `67.00`, LSTM `0.7497`, delta `-0.1544`
- tick `116237`, seconds `66.00`, LSTM `0.8071`, delta `+0.1523`
- tick `118029`, seconds `94.00`, LSTM `0.7493`, delta `+0.1502`
- tick `116173`, seconds `65.00`, LSTM `0.6762`, delta `+0.1286`
- tick `116269`, seconds `66.50`, LSTM `0.9042`, delta `+0.0970`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008647`, |coef| `0.008647`
- `lag_00__kill_diff_last_3s`: coefficient `0.006153`, |coef| `0.006153`
- `lag_15__T_flash_alpha_mean`: coefficient `-0.005399`, |coef| `0.005399`
- `lag_00__CT_kills_last_3s`: coefficient `0.004903`, |coef| `0.004903`
- `lag_00__damage_diff_last_5s`: coefficient `0.004520`, |coef| `0.004520`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003441`, |coef| `0.003441`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003202`, |coef| `0.003202`
- `lag_11__CT_place_TSIDELOWER`: coefficient `0.003068`, |coef| `0.003068`
- `lag_03__CT_duck_amount_mean`: coefficient `-0.003046`, |coef| `0.003046`
- `lag_15__T_place_SIDEENTRANCE`: coefficient `0.002735`, |coef| `0.002735`
- `lag_00__T_kills_last_3s`: coefficient `-0.002719`, |coef| `0.002719`
- `lag_15__T_velocity_mean`: coefficient `-0.002718`, |coef| `0.002718`
- `lag_11__CT_place_RAMP`: coefficient `-0.002713`, |coef| `0.002713`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `0.002585`, |coef| `0.002585`
- `lag_15__T3__flash`: coefficient `-0.002460`, |coef| `0.002460`

## Top 10 utility ridge features

- `lag_15__T_flash_alpha_mean`: coefficient `-0.005399` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003202` (lowers CT win probability)
- `lag_15__T3__flash`: coefficient `-0.002460` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.001630` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001620` (lowers CT win probability)
- `lag_04__CT4__flash`: coefficient `0.001562` (raises CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.001524` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.001432` (lowers CT win probability)
- `lag_15__T3__utility_total`: coefficient `-0.001236` (lowers CT win probability)
- `lag_11__active_infernos_total`: coefficient `-0.001197` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008647` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.006153` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004903` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004520` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003441` (raises CT win probability)
- `lag_11__CT_place_TSIDELOWER`: coefficient `0.003068` (raises CT win probability)
- `lag_03__CT_duck_amount_mean`: coefficient `-0.003046` (lowers CT win probability)
- `lag_15__T_place_SIDEENTRANCE`: coefficient `0.002735` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002719` (lowers CT win probability)
- `lag_15__T_velocity_mean`: coefficient `-0.002718` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `118765`, seconds `105.50`, LSTM delta `+0.2990`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019427`
- `lag_03__CT_duck_amount_mean`: contribution `+0.018241`
- `lag_00__kill_diff_last_3s`: contribution `+0.014810`
- `lag_00__CT_kills_last_3s`: contribution `+0.014156`
- `lag_05__CT_duck_amount_mean`: contribution `+0.010264`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019427`
- `lag_00__T3__flash`: contribution `+0.004804`

### tick `117101`, seconds `79.50`, LSTM delta `+0.2440`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.014810`
- `lag_00__CT_kills_last_3s`: contribution `+0.014156`
- `lag_15__T_place_SIDEENTRANCE`: contribution `+0.013346`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.011785`
- `lag_00__damage_diff_last_5s`: contribution `+0.011523`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `+0.004049`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.003829`
- `lag_11__active_infernos_total`: contribution `+0.003439`

### tick `118413`, seconds `100.00`, LSTM delta `-0.2042`

Top all feature movements:
- `lag_11__CT_place_TSIDELOWER`: contribution `-0.041677`
- `lag_00__kill_diff_last_3s`: contribution `-0.014810`
- `lag_00__T_duck_amount_mean`: contribution `-0.011291`
- `lag_00__damage_diff_last_5s`: contribution `-0.010198`
- `lag_00__T_kills_last_3s`: contribution `-0.008613`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119245`, seconds `113.00`, LSTM delta `+0.2034`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.083825`
- `lag_15__T_flash_alpha_mean`: contribution `+0.032760`
- `lag_15__T_velocity_mean`: contribution `+0.008618`
- `lag_15__T3__flash`: contribution `+0.007251`
- `lag_00__CT_velocity_mean`: contribution `+0.005891`

Top utility-only movements:
- `lag_15__T_flash_alpha_mean`: contribution `+0.032760`
- `lag_15__T3__flash`: contribution `+0.007251`
- `lag_04__CT4__flash`: contribution `+0.002708`
- `lag_15__T3__utility_total`: contribution `+0.002013`

### tick `117229`, seconds `81.50`, LSTM delta `-0.1820`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.016736`
- `lag_00__kill_diff_last_3s`: contribution `-0.014810`
- `lag_04__T_place_SIDEENTRANCE`: contribution `-0.009105`
- `lag_01__CT_shots_fired_sum`: contribution `+0.008951`
- `lag_00__T_kills_last_3s`: contribution `-0.008613`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `-0.003706`
