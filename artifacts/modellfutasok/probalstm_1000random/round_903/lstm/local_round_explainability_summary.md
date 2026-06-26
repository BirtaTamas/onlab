# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `12`

## Largest probability jumps

- tick `89961`, seconds `82.00`, LSTM `0.3323`, delta `-0.3175`
- tick `92585`, seconds `123.00`, LSTM `0.6778`, delta `+0.3011`
- tick `88425`, seconds `58.00`, LSTM `0.3053`, delta `-0.2500`
- tick `89385`, seconds `73.00`, LSTM `0.8574`, delta `+0.2386`
- tick `89033`, seconds `67.50`, LSTM `0.5141`, delta `+0.2104`
- tick `89673`, seconds `77.50`, LSTM `0.6684`, delta `-0.1993`
- tick `91817`, seconds `111.00`, LSTM `0.2107`, delta `+0.1803`
- tick `88713`, seconds `62.50`, LSTM `0.1847`, delta `+0.1546`
- tick `88617`, seconds `61.00`, LSTM `0.0537`, delta `-0.1447`
- tick `92425`, seconds `120.50`, LSTM `0.5225`, delta `+0.1390`

## Top 15 local ridge features

- `lag_00__T2__is_scoped`: coefficient `0.007676`, |coef| `0.007676`
- `lag_00__CT_defusing_count`: coefficient `0.007098`, |coef| `0.007098`
- `lag_00__kill_diff_last_3s`: coefficient `0.006036`, |coef| `0.006036`
- `lag_05__CT_defusing_count`: coefficient `0.005727`, |coef| `0.005727`
- `lag_00__CT_kills_last_3s`: coefficient `0.005009`, |coef| `0.005009`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004707`, |coef| `0.004707`
- `lag_09__T_place_JUNGLE`: coefficient `-0.004474`, |coef| `0.004474`
- `lag_00__damage_diff_last_5s`: coefficient `0.003716`, |coef| `0.003716`
- `lag_02__T_place_STAIRS`: coefficient `-0.003208`, |coef| `0.003208`
- `lag_05__T1__duck_amount`: coefficient `-0.003107`, |coef| `0.003107`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003006`, |coef| `0.003006`
- `lag_00__CT_velocity_mean`: coefficient `-0.002927`, |coef| `0.002927`
- `lag_15__T_place_JUNGLE`: coefficient `-0.002516`, |coef| `0.002516`
- `lag_00__T_kills_last_3s`: coefficient `-0.002449`, |coef| `0.002449`
- `lag_00__alive_diff`: coefficient `0.002438`, |coef| `0.002438`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004707` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001307` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.001238` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.001176` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001128` (lowers CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.001015` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `0.000982` (raises CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000977` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000956` (raises CT win probability)
- `lag_13__CT3__molly`: coefficient `0.000938` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T2__is_scoped`: coefficient `0.007676` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.007098` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.006036` (raises CT win probability)
- `lag_05__CT_defusing_count`: coefficient `0.005727` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005009` (raises CT win probability)
- `lag_09__T_place_JUNGLE`: coefficient `-0.004474` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003716` (raises CT win probability)
- `lag_02__T_place_STAIRS`: coefficient `-0.003208` (lowers CT win probability)
- `lag_05__T1__duck_amount`: coefficient `-0.003107` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003006` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `89961`, seconds `82.00`, LSTM delta `-0.3175`

Top all feature movements:
- `lag_00__T2__is_scoped`: contribution `-0.067665`
- `lag_02__T_place_STAIRS`: contribution `-0.061414`
- `lag_00__CT_place_TRAMP`: contribution `-0.027983`
- `lag_04__CT_place_TRAMP`: contribution `-0.020685`
- `lag_00__kill_diff_last_3s`: contribution `-0.014529`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `92585`, seconds `123.00`, LSTM delta `+0.3011`

Top all feature movements:
- `lag_00__T2__is_scoped`: contribution `+0.067665`
- `lag_05__CT_defusing_count`: contribution `+0.055515`
- `lag_00__T_flash_alpha_mean`: contribution `+0.028557`
- `lag_04__CT_defusing_count`: contribution `+0.018170`
- `lag_00__kill_diff_last_3s`: contribution `+0.014529`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.028557`

### tick `88425`, seconds `58.00`, LSTM delta `-0.2500`

Top all feature movements:
- `lag_09__T_place_JUNGLE`: contribution `-0.057953`
- `lag_00__kill_diff_last_3s`: contribution `-0.014529`
- `lag_02__T_place_CONNECTOR`: contribution `-0.009286`
- `lag_07__T_place_CONNECTOR`: contribution `-0.009176`
- `lag_00__damage_diff_last_5s`: contribution `-0.008382`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89385`, seconds `73.00`, LSTM delta `+0.2386`

Top all feature movements:
- `lag_02__T_place_STAIRS`: contribution `+0.061414`
- `lag_00__kill_diff_last_3s`: contribution `+0.014529`
- `lag_00__CT_kills_last_3s`: contribution `+0.014461`
- `lag_13__T_place_STAIRS`: contribution `+0.014133`
- `lag_06__T_place_STAIRS`: contribution `+0.012376`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89033`, seconds `67.50`, LSTM delta `+0.2104`

Top all feature movements:
- `lag_02__T_place_STAIRS`: contribution `+0.061414`
- `lag_14__T_place_JUNGLE`: contribution `+0.017391`
- `lag_00__kill_diff_last_3s`: contribution `+0.014529`
- `lag_00__CT_kills_last_3s`: contribution `+0.014461`
- `lag_09__T_place_STAIRS`: contribution `-0.013797`

Top utility-only movements:
- No utility movement among the top local contributors.
