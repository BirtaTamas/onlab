# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `92361`, seconds `61.50`, LSTM `0.4927`, delta `-0.1792`
- tick `93641`, seconds `81.50`, LSTM `0.3555`, delta `+0.1484`
- tick `94569`, seconds `96.00`, LSTM `0.8549`, delta `+0.1374`
- tick `94441`, seconds `94.00`, LSTM `0.6147`, delta `+0.0917`
- tick `93609`, seconds `81.00`, LSTM `0.2072`, delta `+0.0825`
- tick `92489`, seconds `63.50`, LSTM `0.3129`, delta `-0.0727`
- tick `92425`, seconds `62.50`, LSTM `0.4046`, delta `-0.0638`
- tick `93193`, seconds `74.50`, LSTM `0.2725`, delta `+0.0637`
- tick `93289`, seconds `76.00`, LSTM `0.1924`, delta `-0.0636`
- tick `94601`, seconds `96.50`, LSTM `0.9133`, delta `+0.0584`

## Top 15 local ridge features

- `lag_01__T_place_UNDERA`: coefficient `-0.002413`, |coef| `0.002413`
- `lag_00__T_place_UNDERA`: coefficient `-0.001846`, |coef| `0.001846`
- `lag_00__kill_diff_last_3s`: coefficient `0.001730`, |coef| `0.001730`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001639`, |coef| `0.001639`
- `lag_00__CT_place_ARAMP`: coefficient `0.001616`, |coef| `0.001616`
- `lag_00__CT_kills_last_3s`: coefficient `0.001574`, |coef| `0.001574`
- `lag_00__CT_place_SIDE`: coefficient `-0.001478`, |coef| `0.001478`
- `lag_04__CT_place_ARAMP`: coefficient `0.001468`, |coef| `0.001468`
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001354`, |coef| `0.001354`
- `lag_09__T4__flash_duration`: coefficient `-0.001268`, |coef| `0.001268`
- `lag_02__CT_place_ARAMP`: coefficient `0.001233`, |coef| `0.001233`
- `lag_00__damage_diff_last_5s`: coefficient `0.001226`, |coef| `0.001226`
- `lag_03__CT_place_ARAMP`: coefficient `0.001220`, |coef| `0.001220`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001149`, |coef| `0.001149`
- `lag_06__CT_place_ARAMP`: coefficient `0.001140`, |coef| `0.001140`

## Top 10 utility ridge features

- `lag_09__T4__flash_duration`: coefficient `-0.001268` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.001140` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001054` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000940` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.000929` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000921` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.000797` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000716` (lowers CT win probability)
- `lag_12__CT3__flash`: coefficient `-0.000700` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000689` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_UNDERA`: coefficient `-0.002413` (lowers CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.001846` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001730` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001639` (lowers CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `0.001616` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001574` (raises CT win probability)
- `lag_00__CT_place_SIDE`: coefficient `-0.001478` (lowers CT win probability)
- `lag_04__CT_place_ARAMP`: coefficient `0.001468` (raises CT win probability)
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001354` (lowers CT win probability)
- `lag_02__CT_place_ARAMP`: coefficient `0.001233` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `92361`, seconds `61.50`, LSTM delta `-0.1792`

Top all feature movements:
- `lag_01__T_place_UNDERA`: contribution `-0.037703`
- `lag_00__CT_place_ARAMP`: contribution `-0.010065`
- `lag_04__T_place_EXTENDEDA`: contribution `-0.006713`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005590`
- `lag_00__kill_diff_last_3s`: contribution `-0.004163`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `-0.003316`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.002678`
- `lag_09__CT5__flash_duration`: contribution `-0.002209`

### tick `93641`, seconds `81.50`, LSTM delta `+0.1484`

Top all feature movements:
- `lag_02__CT_place_SIDE`: contribution `+0.033726`
- `lag_00__T_place_UNDERA`: contribution `+0.028848`
- `lag_11__CT_place_SIDE`: contribution `+0.015665`
- `lag_00__CT_kills_last_3s`: contribution `+0.004545`
- `lag_11__T1__is_scoped`: contribution `+0.004314`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.001961`

### tick `94569`, seconds `96.00`, LSTM delta `+0.1374`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.008125`
- `lag_06__CT_place_ARAMP`: contribution `+0.007104`
- `lag_09__T2__flash_duration`: contribution `+0.006801`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.006713`
- `lag_00__CT_kills_last_3s`: contribution `+0.004545`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.006801`
- `lag_02__T1__flash_duration`: contribution `+0.002718`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.002432`
- `lag_09__T1__flash_duration`: contribution `+0.002224`

### tick `94441`, seconds `94.00`, LSTM delta `+0.0917`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.008125`
- `lag_02__CT_place_ARAMP`: contribution `+0.007682`
- `lag_00__CT_kills_last_3s`: contribution `+0.004545`
- `lag_15__CT_place_LONGDOORS`: contribution `+0.004343`
- `lag_00__kill_diff_last_3s`: contribution `+0.004163`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.003969`
- `lag_05__T1__flash_duration`: contribution `+0.002441`
- `lag_05__T_flash_duration_sum`: contribution `+0.002309`

### tick `93609`, seconds `81.00`, LSTM delta `+0.0825`

Top all feature movements:
- `lag_01__CT_place_SIDE`: contribution `+0.024287`
- `lag_10__CT_place_SIDE`: contribution `+0.018247`
- `lag_10__T1__is_scoped`: contribution `+0.004466`
- `lag_13__T_place_EXTENDEDA`: contribution `+0.004152`
- `lag_06__T_place_EXTENDEDA`: contribution `+0.003289`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.002088`
