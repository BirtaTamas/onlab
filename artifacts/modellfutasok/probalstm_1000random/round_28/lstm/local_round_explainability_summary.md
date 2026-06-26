# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `103672`, seconds `33.50`, LSTM `0.5400`, delta `+0.3329`
- tick `103864`, seconds `36.50`, LSTM `0.2986`, delta `-0.2693`
- tick `103352`, seconds `28.50`, LSTM `0.1756`, delta `-0.2641`
- tick `106424`, seconds `76.50`, LSTM `0.1987`, delta `-0.1762`
- tick `102392`, seconds `13.50`, LSTM `0.4190`, delta `+0.1398`
- tick `103608`, seconds `32.50`, LSTM `0.1652`, delta `+0.1208`
- tick `105368`, seconds `60.00`, LSTM `0.4058`, delta `+0.0797`
- tick `104888`, seconds `52.50`, LSTM `0.2763`, delta `+0.0766`
- tick `102104`, seconds `9.00`, LSTM `0.3758`, delta `-0.0741`
- tick `103928`, seconds `37.50`, LSTM `0.2309`, delta `-0.0640`

## Top 15 local ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.003421`, |coef| `0.003421`
- `lag_11__T_place_LONGDOG`: coefficient `0.003254`, |coef| `0.003254`
- `lag_00__kill_diff_last_3s`: coefficient `0.003057`, |coef| `0.003057`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002957`, |coef| `0.002957`
- `lag_00__T1__is_walking`: coefficient `0.002946`, |coef| `0.002946`
- `lag_01__CT4__is_walking`: coefficient `-0.002859`, |coef| `0.002859`
- `lag_00__T_kills_last_3s`: coefficient `-0.002730`, |coef| `0.002730`
- `lag_00__damage_diff_last_5s`: coefficient `0.002718`, |coef| `0.002718`
- `lag_08__T1__is_walking`: coefficient `0.002659`, |coef| `0.002659`
- `lag_08__CT_place_CONNECTOR`: coefficient `0.002448`, |coef| `0.002448`
- `lag_04__CT_place_LONGDOG`: coefficient `0.002395`, |coef| `0.002395`
- `lag_01__CT3__is_walking`: coefficient `0.002275`, |coef| `0.002275`
- `lag_00__T_damage_last_5s`: coefficient `-0.002236`, |coef| `0.002236`
- `lag_09__T_A_site_active_smokes`: coefficient `-0.002178`, |coef| `0.002178`
- `lag_10__T3__is_walking`: coefficient `-0.002168`, |coef| `0.002168`

## Top 10 utility ridge features

- `lag_09__T_A_site_active_smokes`: coefficient `-0.002178` (lowers CT win probability)
- `lag_08__T_A_site_active_smokes`: coefficient `-0.001732` (lowers CT win probability)
- `lag_10__T_A_site_active_smokes`: coefficient `-0.001700` (lowers CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.001681` (lowers CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001637` (raises CT win probability)
- `lag_09__T_active_smokes`: coefficient `-0.001513` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.001493` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001416` (lowers CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `-0.001401` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.001362` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT4__duck_amount`: coefficient `0.003421` (raises CT win probability)
- `lag_11__T_place_LONGDOG`: coefficient `0.003254` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003057` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002957` (raises CT win probability)
- `lag_00__T1__is_walking`: coefficient `0.002946` (raises CT win probability)
- `lag_01__CT4__is_walking`: coefficient `-0.002859` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002730` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002718` (raises CT win probability)
- `lag_08__T1__is_walking`: coefficient `0.002659` (raises CT win probability)
- `lag_08__CT_place_CONNECTOR`: coefficient `0.002448` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `103672`, seconds `33.50`, LSTM delta `+0.3329`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `+0.031523`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `+0.017785`
- `lag_11__T_place_DUMPSTER`: contribution `+0.015379`
- `lag_01__T_shots_fired_sum`: contribution `+0.011159`
- `lag_09__T_shots_fired_sum`: contribution `+0.010708`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.007994`
- `lag_14__T4__flash_duration`: contribution `+0.007733`
- `lag_11__T5__flash_duration`: contribution `+0.007636`
- `lag_02__T4__flash_duration`: contribution `+0.006986`
- `lag_00__T5__flash_duration`: contribution `+0.006886`

### tick `103864`, seconds `36.50`, LSTM delta `-0.2693`

Top all feature movements:
- `lag_09__T_place_ELECTRICALBOX`: contribution `-0.026892`
- `lag_05__CT_shots_fired_sum`: contribution `-0.019861`
- `lag_08__T_place_ELECTRICALBOX`: contribution `-0.018119`
- `lag_00__kill_diff_last_3s`: contribution `-0.014714`
- `lag_15__T_shots_fired_sum`: contribution `-0.012189`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.005311`
- `lag_06__T5__flash_duration`: contribution `-0.004328`

### tick `103352`, seconds `28.50`, LSTM delta `-0.2641`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.017097`
- `lag_04__CT_place_LONGDOG`: contribution `-0.015620`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `-0.014671`
- `lag_01__T_place_DUMPSTER`: contribution `-0.013121`
- `lag_00__CT4__duck_amount`: contribution `-0.012564`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `-0.009382`
- `lag_04__CT2__flash_duration`: contribution `-0.008256`
- `lag_01__T5__flash_duration`: contribution `-0.008101`
- `lag_01__T_flash_duration_sum`: contribution `-0.004566`
- `lag_01__T3__flash_duration`: contribution `-0.003975`

### tick `106424`, seconds `76.50`, LSTM delta `-0.1762`

Top all feature movements:
- `lag_11__T_place_LONGDOG`: contribution `-0.015141`
- `lag_08__CT_place_CONNECTOR`: contribution `-0.008752`
- `lag_00__T_kills_last_3s`: contribution `-0.008649`
- `lag_00__kill_diff_last_3s`: contribution `-0.007357`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.007178`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `-0.003548`

### tick `102392`, seconds `13.50`, LSTM delta `+0.1398`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.045752`
- `lag_04__CT_place_LONGDOG`: contribution `+0.015620`
- `lag_00__CT3__is_scoped`: contribution `+0.007517`
- `lag_02__CT4__duck_amount`: contribution `+0.006928`
- `lag_15__T_place_DUMPSTER`: contribution `+0.006643`

Top utility-only movements:
- `lag_13__CT_flashes_last_5s`: contribution `+0.004196`
- `lag_04__CT1__flash_duration`: contribution `+0.002632`
