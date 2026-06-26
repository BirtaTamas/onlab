# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `120199`, seconds `89.50`, LSTM `0.0652`, delta `-0.1557`
- tick `114535`, seconds `1.00`, LSTM `0.3127`, delta `+0.0814`
- tick `120167`, seconds `89.00`, LSTM `0.2209`, delta `-0.0800`
- tick `114887`, seconds `6.50`, LSTM `0.3345`, delta `-0.0659`
- tick `118343`, seconds `60.50`, LSTM `0.2045`, delta `-0.0483`
- tick `114567`, seconds `1.50`, LSTM `0.3580`, delta `+0.0454`
- tick `114951`, seconds `7.50`, LSTM `0.2470`, delta `-0.0447`
- tick `114919`, seconds `7.00`, LSTM `0.2917`, delta `-0.0427`
- tick `115783`, seconds `20.50`, LSTM `0.2019`, delta `-0.0424`
- tick `116231`, seconds `27.50`, LSTM `0.1825`, delta `+0.0348`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.001852`, |coef| `0.001852`
- `lag_01__CT_smokes_last_5s`: coefficient `0.001728`, |coef| `0.001728`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.001701`, |coef| `0.001701`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.001337`, |coef| `0.001337`
- `lag_00__T_kills_last_3s`: coefficient `-0.001291`, |coef| `0.001291`
- `lag_03__T1__is_walking`: coefficient `-0.001223`, |coef| `0.001223`
- `lag_10__T_flashed_players`: coefficient `0.001176`, |coef| `0.001176`
- `lag_00__CT5__alive`: coefficient `0.001174`, |coef| `0.001174`
- `lag_00__CT2__is_walking`: coefficient `-0.001171`, |coef| `0.001171`
- `lag_12__T_flashed_players`: coefficient `-0.001164`, |coef| `0.001164`
- `lag_09__T_flashed_players`: coefficient `0.001113`, |coef| `0.001113`
- `lag_02__CT_smokes_last_5s`: coefficient `0.001106`, |coef| `0.001106`
- `lag_00__CT1__is_walking`: coefficient `0.001101`, |coef| `0.001101`
- `lag_00__CT5__armor`: coefficient `0.001099`, |coef| `0.001099`
- `lag_00__T_place_BALCONY`: coefficient `-0.001092`, |coef| `0.001092`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.001852` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.001728` (raises CT win probability)
- `lag_02__CT_smokes_last_5s`: coefficient `0.001106` (raises CT win probability)
- `lag_13__CT_B_site_active_smokes`: coefficient `0.000844` (raises CT win probability)
- `lag_03__CT_smokes_last_5s`: coefficient `0.000840` (raises CT win probability)
- `lag_15__CT_smokes_last_5s`: coefficient `0.000713` (raises CT win probability)
- `lag_05__T5__flash`: coefficient `0.000654` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `0.000643` (raises CT win probability)
- `lag_15__T2__flash`: coefficient `0.000640` (raises CT win probability)
- `lag_07__T_A_site_active_smokes`: coefficient `-0.000607` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_APARTMENTS`: coefficient `0.001701` (raises CT win probability)
- `lag_00__closest_enemy_dist_diff`: coefficient `0.001337` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001291` (lowers CT win probability)
- `lag_03__T1__is_walking`: coefficient `-0.001223` (lowers CT win probability)
- `lag_10__T_flashed_players`: coefficient `0.001176` (raises CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001174` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001171` (lowers CT win probability)
- `lag_12__T_flashed_players`: coefficient `-0.001164` (lowers CT win probability)
- `lag_09__T_flashed_players`: coefficient `0.001113` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.001101` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `120199`, seconds `89.50`, LSTM delta `-0.1557`

Top all feature movements:
- `lag_00__CT_place_APARTMENTS`: contribution `-0.006536`
- `lag_10__T_flashed_players`: contribution `-0.004538`
- `lag_12__T_flashed_players`: contribution `-0.004492`
- `lag_00__closest_enemy_dist_diff`: contribution `-0.004195`
- `lag_00__T_kills_last_3s`: contribution `-0.004089`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114535`, seconds `1.00`, LSTM delta `+0.0814`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.032011`
- `lag_02__T_place_TSPAWN`: contribution `+0.002726`
- `lag_02__CT_closest_enemy_dist`: contribution `+0.002027`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.001919`
- `lag_02__centroid_distance_xy`: contribution `+0.001339`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.032011`
- `lag_00__CT4__smoke`: contribution `+0.000722`
- `lag_02__T_smoke_inv`: contribution `+0.000581`
- `lag_02__CT1__smoke`: contribution `+0.000563`
- `lag_02__CT2__smoke`: contribution `+0.000507`

### tick `120167`, seconds `89.00`, LSTM delta `-0.0800`

Top all feature movements:
- `lag_09__T_flashed_players`: contribution `-0.004294`
- `lag_11__CT_place_RUINS`: contribution `-0.003223`
- `lag_00__CT2__is_walking`: contribution `+0.002763`
- `lag_14__T4__duck_amount`: contribution `-0.002714`
- `lag_00__CT1__is_walking`: contribution `-0.002570`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114887`, seconds `6.50`, LSTM delta `-0.0659`

Top all feature movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.029862`
- `lag_11__CT_smokes_last_5s`: contribution `-0.006070`
- `lag_04__T_place_LOWERMID`: contribution `-0.003037`
- `lag_07__CT_place_LIBRARY`: contribution `-0.002947`
- `lag_01__CT_place_RUINS`: contribution `-0.001777`

Top utility-only movements:
- `lag_01__CT_smokes_last_5s`: contribution `-0.029862`
- `lag_11__CT_smokes_last_5s`: contribution `-0.006070`
- `lag_02__CT1__smoke`: contribution `-0.000812`
- `lag_00__CT4__smoke`: contribution `-0.000722`
- `lag_13__CT_B_site_active_smokes`: contribution `-0.000510`

### tick `118343`, seconds `60.50`, LSTM delta `-0.0483`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `-0.005165`
- `lag_01__CT_place_ARCH`: contribution `-0.003923`
- `lag_02__T2__duck_amount`: contribution `-0.002096`
- `lag_02__CT_place_ARCH`: contribution `-0.001947`
- `lag_13__T1__is_walking`: contribution `-0.001915`

Top utility-only movements:
- No utility movement among the top local contributors.
