# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `124087`, seconds `93.50`, LSTM `0.4750`, delta `+0.2774`
- tick `120727`, seconds `41.00`, LSTM `0.5200`, delta `-0.2464`
- tick `123607`, seconds `86.00`, LSTM `0.1919`, delta `-0.2116`
- tick `123895`, seconds `90.50`, LSTM `0.2264`, delta `+0.1650`
- tick `123863`, seconds `90.00`, LSTM `0.0615`, delta `-0.1584`
- tick `123831`, seconds `89.50`, LSTM `0.2199`, delta `+0.1212`
- tick `121495`, seconds `53.00`, LSTM `0.5307`, delta `+0.1014`
- tick `124311`, seconds `97.00`, LSTM `0.5665`, delta `+0.0914`
- tick `119415`, seconds `20.50`, LSTM `0.8531`, delta `+0.0829`
- tick `121463`, seconds `52.50`, LSTM `0.4293`, delta `+0.0747`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004483`, |coef| `0.004483`
- `lag_00__T_kills_last_3s`: coefficient `-0.004275`, |coef| `0.004275`
- `lag_00__damage_diff_last_5s`: coefficient `0.003971`, |coef| `0.003971`
- `lag_00__T_damage_last_5s`: coefficient `-0.003365`, |coef| `0.003365`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003232`, |coef| `0.003232`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003188`, |coef| `0.003188`
- `lag_00__CT2__duck_amount`: coefficient `0.003072`, |coef| `0.003072`
- `lag_00__T3__duck_amount`: coefficient `0.002778`, |coef| `0.002778`
- `lag_00__T_velocity_mean`: coefficient `-0.002683`, |coef| `0.002683`
- `lag_04__CT_velocity_mean`: coefficient `0.002511`, |coef| `0.002511`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002474`, |coef| `0.002474`
- `lag_11__CT5__duck_amount`: coefficient `0.002462`, |coef| `0.002462`
- `lag_10__CT5__duck_amount`: coefficient `0.002324`, |coef| `0.002324`
- `lag_07__T4__flash_duration`: coefficient `0.002319`, |coef| `0.002319`
- `lag_01__T_kills_last_3s`: coefficient `-0.002210`, |coef| `0.002210`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003188` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.002319` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.002190` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.002141` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.001884` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001754` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.001653` (lowers CT win probability)
- `lag_07__CT2__molly`: coefficient `0.001555` (raises CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001511` (raises CT win probability)
- `lag_09__CT1__smoke`: coefficient `0.001455` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004483` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004275` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003971` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003365` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003232` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.003072` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `0.002778` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.002683` (lowers CT win probability)
- `lag_04__CT_velocity_mean`: coefficient `0.002511` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002474` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `124087`, seconds `93.50`, LSTM delta `+0.2774`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019340`
- `lag_00__CT_duck_amount_mean`: contribution `+0.009579`
- `lag_00__T3__duck_amount`: contribution `+0.009385`
- `lag_00__centroid_distance_xy`: contribution `+0.007463`
- `lag_01__T_kills_last_3s`: contribution `+0.007001`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019340`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.006025`

### tick `120727`, seconds `41.00`, LSTM delta `-0.2464`

Top all feature movements:
- `lag_07__T4__flash_duration`: contribution `-0.016880`
- `lag_00__T_kills_last_3s`: contribution `-0.013544`
- `lag_10__T1__flash_duration`: contribution `-0.010932`
- `lag_00__kill_diff_last_3s`: contribution `-0.010791`
- `lag_00__damage_diff_last_5s`: contribution `-0.008959`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `-0.016880`
- `lag_10__T1__flash_duration`: contribution `-0.010932`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.005834`
- `lag_12__T2__flash_duration`: contribution `-0.005788`
- `lag_08__T_B_site_active_infernos`: contribution `-0.003511`

### tick `123607`, seconds `86.00`, LSTM delta `-0.2116`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013544`
- `lag_00__CT2__duck_amount`: contribution `-0.011703`
- `lag_00__CT_duck_amount_mean`: contribution `-0.011607`
- `lag_00__kill_diff_last_3s`: contribution `-0.010791`
- `lag_00__T3__duck_amount`: contribution `-0.010476`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.007355`
- `lag_07__CT2__molly`: contribution `-0.003835`

### tick `123895`, seconds `90.50`, LSTM delta `+0.1650`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.014603`
- `lag_00__kill_diff_last_3s`: contribution `+0.010791`
- `lag_00__T_damage_last_5s`: contribution `+0.007503`
- `lag_01__T_kills_last_3s`: contribution `-0.007001`
- `lag_12__CT1__duck_amount`: contribution `+0.006764`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003739`

### tick `123863`, seconds `90.00`, LSTM delta `-0.1584`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013544`
- `lag_00__kill_diff_last_3s`: contribution `-0.010791`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008595`
- `lag_00__damage_diff_last_5s`: contribution `-0.006271`
- `lag_11__CT1__duck_amount`: contribution `-0.005803`

Top utility-only movements:
- No utility movement among the top local contributors.
