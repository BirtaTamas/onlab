# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `11`

## Largest probability jumps

- tick `92734`, seconds `103.50`, LSTM `0.5548`, delta `+0.4097`
- tick `93438`, seconds `114.50`, LSTM `0.7290`, delta `+0.4025`
- tick `92798`, seconds `104.50`, LSTM `0.2866`, delta `-0.2632`
- tick `91902`, seconds `90.50`, LSTM `0.7386`, delta `+0.2452`
- tick `90238`, seconds `64.50`, LSTM `0.7536`, delta `+0.2245`
- tick `92126`, seconds `94.00`, LSTM `0.4768`, delta `-0.2100`
- tick `93630`, seconds `117.50`, LSTM `0.8250`, delta `+0.2090`
- tick `91518`, seconds `84.50`, LSTM `0.5234`, delta `-0.1175`
- tick `93470`, seconds `115.00`, LSTM `0.6260`, delta `-0.1030`
- tick `91678`, seconds `87.00`, LSTM `0.3598`, delta `-0.0981`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005388`, |coef| `0.005388`
- `lag_00__CT_defusing_count`: coefficient `0.005378`, |coef| `0.005378`
- `lag_00__CT_shots_fired_sum`: coefficient `0.005193`, |coef| `0.005193`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004555`, |coef| `0.004555`
- `lag_15__T1__flash_duration`: coefficient `-0.004549`, |coef| `0.004549`
- `lag_00__CT_kills_last_3s`: coefficient `0.004362`, |coef| `0.004362`
- `lag_00__CT_velocity_mean`: coefficient `-0.004241`, |coef| `0.004241`
- `lag_07__T_bomb_zone_count`: coefficient `-0.003939`, |coef| `0.003939`
- `lag_07__T_duck_amount_mean`: coefficient `-0.003589`, |coef| `0.003589`
- `lag_06__T_flash_alpha_mean`: coefficient `-0.003366`, |coef| `0.003366`
- `lag_00__T_velocity_mean`: coefficient `-0.003298`, |coef| `0.003298`
- `lag_00__damage_diff_last_5s`: coefficient `0.003280`, |coef| `0.003280`
- `lag_01__T_duck_amount_mean`: coefficient `-0.003109`, |coef| `0.003109`
- `lag_00__CT2__duck_amount`: coefficient `0.002910`, |coef| `0.002910`
- `lag_01__CT_place_CONNECTOR`: coefficient `-0.002796`, |coef| `0.002796`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004555` (lowers CT win probability)
- `lag_15__T1__flash_duration`: coefficient `-0.004549` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.003366` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001935` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001904` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `-0.001865` (lowers CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.001675` (lowers CT win probability)
- `lag_06__T1__flash`: coefficient `-0.001503` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001411` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001400` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005388` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.005378` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.005193` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004362` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.004241` (lowers CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.003939` (lowers CT win probability)
- `lag_07__T_duck_amount_mean`: coefficient `-0.003589` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.003298` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003280` (raises CT win probability)
- `lag_01__T_duck_amount_mean`: coefficient `-0.003109` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `92734`, seconds `103.50`, LSTM delta `+0.4097`

Top all feature movements:
- `lag_15__T1__flash_duration`: contribution `+0.027230`
- `lag_00__CT_shots_fired_sum`: contribution `+0.018040`
- `lag_00__T_velocity_mean`: contribution `+0.017151`
- `lag_00__kill_diff_last_3s`: contribution `+0.012970`
- `lag_00__CT_kills_last_3s`: contribution `+0.012595`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.027230`

### tick `93438`, seconds `114.50`, LSTM delta `+0.4025`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.027638`
- `lag_07__T_bomb_zone_count`: contribution `+0.022929`
- `lag_07__T_duck_amount_mean`: contribution `+0.020875`
- `lag_04__T_duck_amount_mean`: contribution `+0.013122`
- `lag_00__kill_diff_last_3s`: contribution `+0.012970`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.027638`

### tick `92798`, seconds `104.50`, LSTM delta `-0.2632`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.036080`
- `lag_01__T_duck_amount_mean`: contribution `-0.018084`
- `lag_02__T_velocity_mean`: contribution `-0.013541`
- `lag_00__kill_diff_last_3s`: contribution `-0.012970`
- `lag_00__CT2__duck_amount`: contribution `-0.011086`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `91902`, seconds `90.50`, LSTM delta `+0.2452`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014432`
- `lag_04__CT_place_LONGDOG`: contribution `+0.013507`
- `lag_00__kill_diff_last_3s`: contribution `+0.012970`
- `lag_00__CT_kills_last_3s`: contribution `+0.012595`
- `lag_00__T1__flash_duration`: contribution `+0.011396`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.011396`
- `lag_00__CT2__flash_duration`: contribution `+0.011077`
- `lag_00__T_A_site_active_infernos`: contribution `+0.004168`

### tick `90238`, seconds `64.50`, LSTM delta `+0.2245`

Top all feature movements:
- `lag_09__CT_place_IVY`: contribution `+0.022212`
- `lag_00__CT_shots_fired_sum`: contribution `+0.018040`
- `lag_00__kill_diff_last_3s`: contribution `+0.012970`
- `lag_00__CT_kills_last_3s`: contribution `+0.012595`
- `lag_00__T_place_DUMPSTER`: contribution `+0.008428`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `+0.004177`
- `lag_00__T4__flash`: contribution `+0.002890`
