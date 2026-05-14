# Local Round Explainability

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `4`

## Largest probability jumps

- tick `31504`, seconds `57.00`, LSTM `0.8204`, delta `+0.2304`
- tick `31920`, seconds `63.50`, LSTM `0.9126`, delta `+0.1535`
- tick `31408`, seconds `55.50`, LSTM `0.6398`, delta `+0.1493`
- tick `29456`, seconds `25.00`, LSTM `0.4150`, delta `+0.1175`
- tick `29424`, seconds `24.50`, LSTM `0.2974`, delta `+0.1139`
- tick `27888`, seconds `0.50`, LSTM `0.1266`, delta `-0.0710`
- tick `31472`, seconds `56.50`, LSTM `0.5900`, delta `-0.0620`
- tick `29648`, seconds `28.00`, LSTM `0.4027`, delta `-0.0455`
- tick `29520`, seconds `26.00`, LSTM `0.4387`, delta `+0.0431`
- tick `30096`, seconds `35.00`, LSTM `0.5421`, delta `+0.0413`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.003124`, |coef| `0.003124`
- `lag_00__T_place_MAIN`: coefficient `-0.002531`, |coef| `0.002531`
- `lag_00__CT_kills_last_3s`: coefficient `0.002121`, |coef| `0.002121`
- `lag_13__CT_place_BRICKS`: coefficient `0.001864`, |coef| `0.001864`
- `lag_00__kill_diff_last_3s`: coefficient `0.001843`, |coef| `0.001843`
- `lag_12__CT_place_WALKWAY`: coefficient `-0.001805`, |coef| `0.001805`
- `lag_03__T1__is_scoped`: coefficient `-0.001741`, |coef| `0.001741`
- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001686`, |coef| `0.001686`
- `lag_10__CT_place_BRICKS`: coefficient `0.001375`, |coef| `0.001375`
- `lag_02__CT_place_FOUNTAIN`: coefficient `0.001330`, |coef| `0.001330`
- `lag_09__T_place_MAIN`: coefficient `0.001236`, |coef| `0.001236`
- `lag_00__damage_diff_last_5s`: coefficient `0.001202`, |coef| `0.001202`
- `lag_12__CT_place_MAIN`: coefficient `0.001192`, |coef| `0.001192`
- `lag_04__CT5__flash_duration`: coefficient `-0.001188`, |coef| `0.001188`
- `lag_05__CT5__flash_duration`: coefficient `0.001177`, |coef| `0.001177`

## Top 10 utility ridge features

- `lag_04__CT5__flash_duration`: coefficient `-0.001188` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001177` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `-0.001067` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.000998` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.000984` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.000861` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `0.000816` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000807` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000799` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.000753` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.003124` (lowers CT win probability)
- `lag_00__T_place_MAIN`: coefficient `-0.002531` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002121` (raises CT win probability)
- `lag_13__CT_place_BRICKS`: coefficient `0.001864` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001843` (raises CT win probability)
- `lag_12__CT_place_WALKWAY`: coefficient `-0.001805` (lowers CT win probability)
- `lag_03__T1__is_scoped`: coefficient `-0.001741` (lowers CT win probability)
- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001686` (lowers CT win probability)
- `lag_10__CT_place_BRICKS`: coefficient `0.001375` (raises CT win probability)
- `lag_02__CT_place_FOUNTAIN`: coefficient `0.001330` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `31504`, seconds `57.00`, LSTM delta `+0.2304`

Top all feature movements:
- `lag_13__CT_place_BRICKS`: contribution `+0.035801`
- `lag_00__T_place_MAIN`: contribution `+0.016365`
- `lag_03__T1__is_scoped`: contribution `+0.009948`
- `lag_12__CT_place_WALKWAY`: contribution `+0.008861`
- `lag_05__CT5__flash_duration`: contribution `+0.008766`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `+0.008766`
- `lag_08__T1__flash_duration`: contribution `+0.005688`

### tick `31920`, seconds `63.50`, LSTM delta `+0.1535`

Top all feature movements:
- `lag_00__T_place_MAIN`: contribution `+0.016365`
- `lag_03__CT_place_BRICKS`: contribution `+0.012081`
- `lag_06__CT_place_BRICKS`: contribution `+0.009253`
- `lag_04__CT5__flash_duration`: contribution `+0.008851`
- `lag_09__T_place_MAIN`: contribution `+0.007994`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.008851`
- `lag_04__CT_flash_duration_sum`: contribution `+0.001986`

### tick `31408`, seconds `55.50`, LSTM delta `+0.1493`

Top all feature movements:
- `lag_10__CT_place_BRICKS`: contribution `+0.026410`
- `lag_00__T_place_MAIN`: contribution `+0.016365`
- `lag_02__CT5__flash_duration`: contribution `+0.007331`
- `lag_00__CT_kills_last_3s`: contribution `+0.006123`
- `lag_14__T_place_MAIN`: contribution `+0.004468`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.007331`
- `lag_14__T1__flash_duration`: contribution `+0.004349`
- `lag_05__T1__flash_duration`: contribution `+0.004012`

### tick `29456`, seconds `25.00`, LSTM delta `+0.1175`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006123`
- `lag_05__CT_place_WALKWAY`: contribution `+0.005073`
- `lag_00__kill_diff_last_3s`: contribution `+0.004435`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004431`
- `lag_01__T_place_TSTAIRS`: contribution `+0.003904`

Top utility-only movements:
- `lag_12__T_active_infernos`: contribution `+0.003588`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.002437`

### tick `29424`, seconds `24.50`, LSTM delta `+0.1139`

Top all feature movements:
- `lag_03__T1__is_scoped`: contribution `+0.009948`
- `lag_12__CT_place_WALKWAY`: contribution `+0.008861`
- `lag_00__T_place_TSTAIRS`: contribution `+0.005393`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004431`
- `lag_11__T_active_infernos`: contribution `+0.004156`

Top utility-only movements:
- `lag_11__T_active_infernos`: contribution `+0.004156`
- `lag_09__T_utility_damage_last_5s`: contribution `+0.002456`
