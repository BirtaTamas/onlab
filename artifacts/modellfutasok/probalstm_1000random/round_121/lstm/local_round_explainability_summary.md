# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `1`

## Largest probability jumps

- tick `9193`, seconds `87.00`, LSTM `0.0566`, delta `-0.1517`
- tick `7113`, seconds `54.50`, LSTM `0.1681`, delta `-0.1471`
- tick `8137`, seconds `70.50`, LSTM `0.1185`, delta `-0.1191`
- tick `7305`, seconds `57.50`, LSTM `0.2346`, delta `+0.1095`
- tick `8937`, seconds `83.00`, LSTM `0.1314`, delta `+0.1062`
- tick `8585`, seconds `77.50`, LSTM `0.0486`, delta `-0.0701`
- tick `3657`, seconds `0.50`, LSTM `0.1551`, delta `-0.0682`
- tick `6569`, seconds `46.00`, LSTM `0.3118`, delta `+0.0534`
- tick `5289`, seconds `26.00`, LSTM `0.2595`, delta `-0.0454`
- tick `5737`, seconds `33.00`, LSTM `0.3126`, delta `+0.0454`

## Top 15 local ridge features

- `lag_00__CT_place_CRANE`: coefficient `-0.004562`, |coef| `0.004562`
- `lag_00__CT_place_RAFTERS`: coefficient `0.001575`, |coef| `0.001575`
- `lag_08__T_place_MINI`: coefficient `0.001553`, |coef| `0.001553`
- `lag_00__CT_place_VENTS`: coefficient `0.001501`, |coef| `0.001501`
- `lag_01__CT_place_MINI`: coefficient `-0.001444`, |coef| `0.001444`
- `lag_15__CT_place_SECRET`: coefficient `0.001354`, |coef| `0.001354`
- `lag_01__CT_place_CRANE`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001129`, |coef| `0.001129`
- `lag_08__CT_place_SECRET`: coefficient `-0.001082`, |coef| `0.001082`
- `lag_00__CT2__is_walking`: coefficient `-0.001065`, |coef| `0.001065`
- `lag_10__T_place_HUT`: coefficient `-0.000960`, |coef| `0.000960`
- `lag_11__T1__duck_amount`: coefficient `-0.000952`, |coef| `0.000952`
- `lag_02__CT_place_RAFTERS`: coefficient `0.000950`, |coef| `0.000950`
- `lag_01__CT_place_RAFTERS`: coefficient `0.000936`, |coef| `0.000936`
- `lag_00__T_velocity_mean`: coefficient `-0.000880`, |coef| `0.000880`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000634` (lowers CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000620` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000518` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000492` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000430` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000407` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `0.000403` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000402` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000401` (raises CT win probability)
- `lag_08__T1__molly`: coefficient `0.000366` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CRANE`: coefficient `-0.004562` (lowers CT win probability)
- `lag_00__CT_place_RAFTERS`: coefficient `0.001575` (raises CT win probability)
- `lag_08__T_place_MINI`: coefficient `0.001553` (raises CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `0.001501` (raises CT win probability)
- `lag_01__CT_place_MINI`: coefficient `-0.001444` (lowers CT win probability)
- `lag_15__CT_place_SECRET`: coefficient `0.001354` (raises CT win probability)
- `lag_01__CT_place_CRANE`: coefficient `-0.001240` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001129` (raises CT win probability)
- `lag_08__CT_place_SECRET`: coefficient `-0.001082` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001065` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9193`, seconds `87.00`, LSTM delta `-0.1517`

Top all feature movements:
- `lag_08__T_place_MINI`: contribution `-0.021608`
- `lag_01__CT_place_MINI`: contribution `-0.017712`
- `lag_15__CT_place_SECRET`: contribution `-0.013939`
- `lag_10__T_place_HUT`: contribution `-0.008952`
- `lag_03__T_place_HUT`: contribution `-0.004481`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `-0.001947`
- `lag_06__T_A_site_active_infernos`: contribution `-0.001280`

### tick `7113`, seconds `54.50`, LSTM delta `-0.1471`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.074837`
- `lag_00__CT_place_RAFTERS`: contribution `-0.008417`
- `lag_11__T1__duck_amount`: contribution `-0.003727`
- `lag_05__T1__duck_amount`: contribution `-0.001792`
- `lag_08__CT1__is_walking`: contribution `-0.001578`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8137`, seconds `70.50`, LSTM delta `-0.1191`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.074837`
- `lag_00__CT_place_RAFTERS`: contribution `-0.008417`
- `lag_14__T_place_VENDING`: contribution `+0.003048`
- `lag_14__T_place_TROPHY`: contribution `-0.002717`
- `lag_10__CT_place_VENTS`: contribution `-0.002444`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7305`, seconds `57.50`, LSTM delta `+0.1095`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `+0.074837`
- `lag_00__CT_place_RAFTERS`: contribution `+0.008417`
- `lag_11__T1__duck_amount`: contribution `+0.003475`
- `lag_05__T5__duck_amount`: contribution `+0.001886`
- `lag_06__CT_place_CRANE`: contribution `+0.001837`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8937`, seconds `83.00`, LSTM delta `+0.1062`

Top all feature movements:
- `lag_15__CT_place_SECRET`: contribution `+0.013939`
- `lag_00__T_place_MINI`: contribution `+0.011570`
- `lag_08__CT_place_SECRET`: contribution `+0.011139`
- `lag_10__CT_place_SECRET`: contribution `+0.005970`
- `lag_03__T_place_HUT`: contribution `-0.004481`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `+0.001293`
