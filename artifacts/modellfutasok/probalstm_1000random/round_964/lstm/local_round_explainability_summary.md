# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `9`

## Largest probability jumps

- tick `76627`, seconds `122.00`, LSTM `0.4874`, delta `+0.3256`
- tick `77619`, seconds `137.50`, LSTM `0.8129`, delta `+0.3005`
- tick `77203`, seconds `131.00`, LSTM `0.4430`, delta `+0.2543`
- tick `75987`, seconds `112.00`, LSTM `0.4888`, delta `+0.2267`
- tick `75827`, seconds `109.50`, LSTM `0.2987`, delta `-0.2162`
- tick `70515`, seconds `26.50`, LSTM `0.2936`, delta `-0.2144`
- tick `70771`, seconds `30.50`, LSTM `0.4552`, delta `+0.1731`
- tick `76211`, seconds `115.50`, LSTM `0.3475`, delta `-0.1624`
- tick `76819`, seconds `125.00`, LSTM `0.3055`, delta `-0.1553`
- tick `70963`, seconds `33.50`, LSTM `0.6536`, delta `+0.1314`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008440`, |coef| `0.008440`
- `lag_00__kill_diff_last_3s`: coefficient `0.005995`, |coef| `0.005995`
- `lag_13__T_flash_alpha_mean`: coefficient `-0.005280`, |coef| `0.005280`
- `lag_00__CT_velocity_mean`: coefficient `-0.004931`, |coef| `0.004931`
- `lag_00__CT_kills_last_3s`: coefficient `0.004540`, |coef| `0.004540`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003559`, |coef| `0.003559`
- `lag_00__damage_diff_last_5s`: coefficient `0.003190`, |coef| `0.003190`
- `lag_00__T_kills_last_3s`: coefficient `-0.002909`, |coef| `0.002909`
- `lag_04__T_bomb_zone_count`: coefficient `0.002853`, |coef| `0.002853`
- `lag_12__T_flash_alpha_mean`: coefficient `-0.002747`, |coef| `0.002747`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002697`, |coef| `0.002697`
- `lag_01__CT_defusing_count`: coefficient `0.002687`, |coef| `0.002687`
- `lag_00__T_macro_B`: coefficient `-0.002682`, |coef| `0.002682`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002682`, |coef| `0.002682`
- `lag_01__damage_diff_last_5s`: coefficient `0.002637`, |coef| `0.002637`

## Top 10 utility ridge features

- `lag_13__T_flash_alpha_mean`: coefficient `-0.005280` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003559` (lowers CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.002747` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.002457` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.002329` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001837` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001728` (lowers CT win probability)
- `lag_14__T_flash_alpha_mean`: coefficient `-0.001677` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.001606` (raises CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `-0.001441` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.008440` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005995` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.004931` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004540` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003190` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002909` (lowers CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `0.002853` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002697` (lowers CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.002687` (raises CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002682` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `76627`, seconds `122.00`, LSTM delta `+0.3256`

Top all feature movements:
- `lag_11__CT_place_ENTRANCE`: contribution `+0.022982`
- `lag_00__kill_diff_last_3s`: contribution `+0.014429`
- `lag_00__CT_kills_last_3s`: contribution `+0.013108`
- `lag_14__T2__duck_amount`: contribution `+0.007540`
- `lag_00__T_duck_amount_mean`: contribution `+0.007200`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77619`, seconds `137.50`, LSTM delta `+0.3005`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.081817`
- `lag_13__T_flash_alpha_mean`: contribution `+0.032032`
- `lag_00__CT_velocity_mean`: contribution `+0.025186`
- `lag_13__T_velocity_mean`: contribution `+0.005453`
- `lag_13__T2__alive`: contribution `+0.004990`

Top utility-only movements:
- `lag_13__T_flash_alpha_mean`: contribution `+0.032032`

### tick `77203`, seconds `131.00`, LSTM delta `+0.2543`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.021593`
- `lag_00__kill_diff_last_3s`: contribution `+0.014429`
- `lag_00__CT_kills_last_3s`: contribution `+0.013108`
- `lag_12__kill_diff_last_3s`: contribution `+0.007354`
- `lag_06__T_kills_last_3s`: contribution `+0.006686`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.021593`

### tick `75987`, seconds `112.00`, LSTM delta `+0.2267`

Top all feature movements:
- `lag_04__T_bomb_zone_count`: contribution `+0.016611`
- `lag_00__kill_diff_last_3s`: contribution `+0.014429`
- `lag_00__CT_kills_last_3s`: contribution `+0.013108`
- `lag_00__CT3__shots_fired`: contribution `+0.007246`
- `lag_04__T1__duck_amount`: contribution `+0.007103`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.004190`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.004008`
- `lag_07__T2__flash_duration`: contribution `+0.003918`
- `lag_07__T_B_site_active_infernos`: contribution `+0.003536`
- `lag_07__T_A_site_active_infernos`: contribution `+0.003514`

### tick `75827`, seconds `109.50`, LSTM delta `-0.2162`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.014429`
- `lag_00__T_shots_fired_sum`: contribution `-0.010109`
- `lag_00__T_kills_last_3s`: contribution `-0.009215`
- `lag_01__T_shots_fired_sum`: contribution `-0.006200`
- `lag_12__CT_place_CONNECTOR`: contribution `-0.005059`

Top utility-only movements:
- `lag_09__utility_damage_diff_last_5s`: contribution `-0.004974`
- `lag_02__T2__flash_duration`: contribution `-0.004393`
- `lag_02__T_A_site_active_infernos`: contribution `-0.002817`
- `lag_09__CT_utility_damage_last_5s`: contribution `-0.002804`
