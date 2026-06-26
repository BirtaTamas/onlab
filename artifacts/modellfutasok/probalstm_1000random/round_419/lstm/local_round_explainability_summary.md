# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `102948`, seconds `112.50`, LSTM `0.5834`, delta `-0.2522`
- tick `103140`, seconds `115.50`, LSTM `0.7641`, delta `+0.2357`
- tick `100836`, seconds `79.50`, LSTM `0.8806`, delta `+0.1520`
- tick `103204`, seconds `116.50`, LSTM `0.9093`, delta `+0.1480`
- tick `98820`, seconds `48.00`, LSTM `0.6597`, delta `+0.1145`
- tick `102628`, seconds `107.50`, LSTM `0.8740`, delta `-0.0783`
- tick `100900`, seconds `80.50`, LSTM `0.9401`, delta `+0.0699`
- tick `96996`, seconds `19.50`, LSTM `0.5507`, delta `-0.0471`
- tick `96612`, seconds `13.50`, LSTM `0.6064`, delta `-0.0297`
- tick `100740`, seconds `78.00`, LSTM `0.7297`, delta `-0.0293`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005030`, |coef| `0.005030`
- `lag_00__T_kills_last_3s`: coefficient `-0.004233`, |coef| `0.004233`
- `lag_00__CT_defusing_count`: coefficient `0.003917`, |coef| `0.003917`
- `lag_10__T_kills_last_3s`: coefficient `-0.003290`, |coef| `0.003290`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002966`, |coef| `0.002966`
- `lag_14__CT1__flash_duration`: coefficient `0.002940`, |coef| `0.002940`
- `lag_08__CT1__is_scoped`: coefficient `0.002807`, |coef| `0.002807`
- `lag_08__T_velocity_mean`: coefficient `-0.002604`, |coef| `0.002604`
- `lag_10__kill_diff_last_3s`: coefficient `0.002499`, |coef| `0.002499`
- `lag_02__T_flash_alpha_mean`: coefficient `-0.002432`, |coef| `0.002432`
- `lag_00__CT_place_RUINS`: coefficient `0.002404`, |coef| `0.002404`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002395`, |coef| `0.002395`
- `lag_00__CT_kills_last_3s`: coefficient `0.002176`, |coef| `0.002176`
- `lag_10__CT4__alive`: coefficient `0.002115`, |coef| `0.002115`
- `lag_10__CT4__hp`: coefficient `0.002072`, |coef| `0.002072`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002966` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `0.002940` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.002432` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001517` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001368` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001156` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001145` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000921` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `0.000881` (raises CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.000800` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005030` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004233` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003917` (raises CT win probability)
- `lag_10__T_kills_last_3s`: coefficient `-0.003290` (lowers CT win probability)
- `lag_08__CT1__is_scoped`: coefficient `0.002807` (raises CT win probability)
- `lag_08__T_velocity_mean`: coefficient `-0.002604` (lowers CT win probability)
- `lag_10__kill_diff_last_3s`: coefficient `0.002499` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.002404` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002395` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002176` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `102948`, seconds `112.50`, LSTM delta `-0.2522`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.013409`
- `lag_14__CT1__flash_duration`: contribution `-0.013101`
- `lag_00__kill_diff_last_3s`: contribution `-0.012106`
- `lag_08__CT1__is_scoped`: contribution `-0.012022`
- `lag_10__T_kills_last_3s`: contribution `-0.010424`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.013101`

### tick `103140`, seconds `115.50`, LSTM delta `+0.2357`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.024212`
- `lag_00__T_flash_alpha_mean`: contribution `+0.017994`
- `lag_00__T_kills_last_3s`: contribution `+0.013409`
- `lag_10__T_kills_last_3s`: contribution `+0.010424`
- `lag_04__T_shots_fired_sum`: contribution `+0.008870`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.017994`

### tick `100836`, seconds `79.50`, LSTM delta `+0.1520`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012106`
- `lag_03__CT_place_BALCONY`: contribution `+0.009454`
- `lag_03__T_flashed_players`: contribution `+0.008582`
- `lag_00__CT_kills_last_3s`: contribution `+0.006282`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006146`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.002199`

### tick `103204`, seconds `116.50`, LSTM delta `+0.1480`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.037970`
- `lag_02__T_flash_alpha_mean`: contribution `+0.014758`
- `lag_02__kill_diff_last_3s`: contribution `+0.008751`
- `lag_08__T_velocity_mean`: contribution `+0.007243`
- `lag_08__CT_place_RUINS`: contribution `+0.005413`

Top utility-only movements:
- `lag_02__T_flash_alpha_mean`: contribution `+0.014758`

### tick `98820`, seconds `48.00`, LSTM delta `+0.1145`

Top all feature movements:
- `lag_03__T_place_BALCONY`: contribution `+0.012173`
- `lag_00__kill_diff_last_3s`: contribution `+0.012106`
- `lag_00__T_place_BALCONY`: contribution `+0.008553`
- `lag_00__CT_kills_last_3s`: contribution `+0.006282`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005121`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.003350`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001454`
