# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `46644`, seconds `27.50`, LSTM `0.0919`, delta `-0.1840`
- tick `44916`, seconds `0.50`, LSTM `0.0596`, delta `-0.0624`
- tick `45460`, seconds `9.00`, LSTM `0.1360`, delta `+0.0501`
- tick `45588`, seconds `11.00`, LSTM `0.2106`, delta `+0.0350`
- tick `46132`, seconds `19.50`, LSTM `0.2132`, delta `+0.0295`
- tick `46324`, seconds `22.50`, LSTM `0.2491`, delta `+0.0295`
- tick `45844`, seconds `15.00`, LSTM `0.1945`, delta `-0.0288`
- tick `46260`, seconds `21.50`, LSTM `0.2197`, delta `-0.0281`
- tick `46548`, seconds `26.00`, LSTM `0.2596`, delta `+0.0228`
- tick `46612`, seconds `27.00`, LSTM `0.2759`, delta `+0.0218`

## Top 15 local ridge features

- `lag_10__CT5__duck_amount`: coefficient `0.001892`, |coef| `0.001892`
- `lag_11__T4__duck_amount`: coefficient `-0.001641`, |coef| `0.001641`
- `lag_04__T_place_MAINHALL`: coefficient `-0.001575`, |coef| `0.001575`
- `lag_01__CT_place_RAMP`: coefficient `-0.001557`, |coef| `0.001557`
- `lag_02__T3__duck_amount`: coefficient `0.001548`, |coef| `0.001548`
- `lag_00__T_kills_last_3s`: coefficient `-0.001540`, |coef| `0.001540`
- `lag_03__CT_place_RAMP`: coefficient `-0.001479`, |coef| `0.001479`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001477`, |coef| `0.001477`
- `lag_00__CT4__alive`: coefficient `0.001451`, |coef| `0.001451`
- `lag_08__CT4__is_walking`: coefficient `0.001427`, |coef| `0.001427`
- `lag_00__T5__shots_fired`: coefficient `-0.001422`, |coef| `0.001422`
- `lag_03__T5__is_walking`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__CT4__hp`: coefficient `0.001418`, |coef| `0.001418`
- `lag_00__CT4__armor`: coefficient `0.001340`, |coef| `0.001340`
- `lag_00__T_damage_last_5s`: coefficient `-0.001322`, |coef| `0.001322`

## Top 10 utility ridge features

- `lag_13__T5__smoke`: coefficient `0.001268` (raises CT win probability)
- `lag_09__T5__flash`: coefficient `0.000821` (raises CT win probability)
- `lag_05__T_B_site_active_smokes`: coefficient `-0.000654` (lowers CT win probability)
- `lag_09__T5__utility_total`: coefficient `0.000614` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000588` (lowers CT win probability)
- `lag_13__T5__utility_total`: coefficient `0.000518` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.000485` (lowers CT win probability)
- `lag_05__T_active_smokes`: coefficient `-0.000470` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000465` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.000455` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT5__duck_amount`: coefficient `0.001892` (raises CT win probability)
- `lag_11__T4__duck_amount`: coefficient `-0.001641` (lowers CT win probability)
- `lag_04__T_place_MAINHALL`: coefficient `-0.001575` (lowers CT win probability)
- `lag_01__CT_place_RAMP`: coefficient `-0.001557` (lowers CT win probability)
- `lag_02__T3__duck_amount`: coefficient `0.001548` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001540` (lowers CT win probability)
- `lag_03__CT_place_RAMP`: coefficient `-0.001479` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001477` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001451` (raises CT win probability)
- `lag_08__CT4__is_walking`: coefficient `0.001427` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46644`, seconds `27.50`, LSTM delta `-0.1840`

Top all feature movements:
- `lag_10__CT5__duck_amount`: contribution `-0.006574`
- `lag_11__T4__duck_amount`: contribution `-0.006069`
- `lag_02__T3__duck_amount`: contribution `-0.005836`
- `lag_04__T_place_MAINHALL`: contribution `-0.005687`
- `lag_00__T_shots_fired_sum`: contribution `-0.005538`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44916`, seconds `0.50`, LSTM delta `-0.0624`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.044804`
- `lag_00__T_velocity_mean`: contribution `-0.001622`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001162`
- `lag_01__T_place_TSPAWN`: contribution `-0.001118`
- `lag_01__CT_macro_OTHER`: contribution `-0.000995`

Top utility-only movements:
- `lag_01__T5__molly`: contribution `-0.000304`
- `lag_01__CT5__smoke`: contribution `-0.000294`

### tick `45460`, seconds `9.00`, LSTM delta `+0.0501`

Top all feature movements:
- `lag_11__CT_place_HOUSE`: contribution `+0.005105`
- `lag_09__T_place_TUNNEL`: contribution `+0.003511`
- `lag_12__CT_place_UNKNOWN`: contribution `+0.003033`
- `lag_01__T_place_WATER`: contribution `+0.002979`
- `lag_03__T_place_TUNNEL`: contribution `+0.002491`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.001816`

### tick `45588`, seconds `11.00`, LSTM delta `+0.0350`

Top all feature movements:
- `lag_15__CT_place_HOUSE`: contribution `+0.003848`
- `lag_11__CT_place_HOUSE`: contribution `+0.002552`
- `lag_07__T_place_TUNNEL`: contribution `+0.002245`
- `lag_08__CT5__flash_duration`: contribution `+0.002158`
- `lag_07__T_place_WATER`: contribution `+0.002069`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.002158`
- `lag_00__CT5__flash_duration`: contribution `+0.001063`

### tick `46132`, seconds `19.50`, LSTM delta `+0.0295`

Top all feature movements:
- `lag_05__CT4__is_walking`: contribution `+0.003075`
- `lag_09__CT5__duck_amount`: contribution `+0.002564`
- `lag_02__T3__is_walking`: contribution `+0.002505`
- `lag_13__T3__duck_amount`: contribution `+0.002425`
- `lag_15__T3__duck_amount`: contribution `-0.002281`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.001234`
