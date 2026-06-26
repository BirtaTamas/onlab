# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `31412`, seconds `59.50`, LSTM `0.6713`, delta `+0.1437`
- tick `31924`, seconds `67.50`, LSTM `0.8697`, delta `+0.1056`
- tick `32116`, seconds `70.50`, LSTM `0.9003`, delta `+0.0780`
- tick `28148`, seconds `8.50`, LSTM `0.4239`, delta `-0.0358`
- tick `32212`, seconds `72.00`, LSTM `0.9505`, delta `+0.0309`
- tick `31476`, seconds `60.50`, LSTM `0.7234`, delta `+0.0308`
- tick `31988`, seconds `68.50`, LSTM `0.8447`, delta `-0.0279`
- tick `29172`, seconds `24.50`, LSTM `0.4976`, delta `+0.0279`
- tick `31892`, seconds `67.00`, LSTM `0.7640`, delta `+0.0262`
- tick `31828`, seconds `66.00`, LSTM `0.7193`, delta `+0.0256`

## Top 15 local ridge features

- `lag_13__CT3__flash_duration`: coefficient `0.000979`, |coef| `0.000979`
- `lag_00__CT_kills_last_3s`: coefficient `0.000910`, |coef| `0.000910`
- `lag_06__CT_place_SIDEHALL`: coefficient `0.000901`, |coef| `0.000901`
- `lag_15__CT_place_TSIDELOWER`: coefficient `0.000899`, |coef| `0.000899`
- `lag_13__T5__flash_duration`: coefficient `0.000868`, |coef| `0.000868`
- `lag_14__CT_place_TSIDELOWER`: coefficient `-0.000832`, |coef| `0.000832`
- `lag_00__CT_place_MAINHALL`: coefficient `-0.000825`, |coef| `0.000825`
- `lag_01__CT5__flash_duration`: coefficient `0.000809`, |coef| `0.000809`
- `lag_01__T4__duck_amount`: coefficient `0.000794`, |coef| `0.000794`
- `lag_03__T5__flash_duration`: coefficient `-0.000791`, |coef| `0.000791`
- `lag_08__T_place_MAINHALL`: coefficient `0.000754`, |coef| `0.000754`
- `lag_12__T_place_MAINHALL`: coefficient `0.000733`, |coef| `0.000733`
- `lag_00__kill_diff_last_3s`: coefficient `0.000723`, |coef| `0.000723`
- `lag_11__CT_place_TSIDELOWER`: coefficient `0.000722`, |coef| `0.000722`
- `lag_02__T4__duck_amount`: coefficient `0.000719`, |coef| `0.000719`

## Top 10 utility ridge features

- `lag_13__CT3__flash_duration`: coefficient `0.000979` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.000868` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `0.000809` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000791` (lowers CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.000589` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.000560` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.000555` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000525` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000502` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.000496` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000910` (raises CT win probability)
- `lag_06__CT_place_SIDEHALL`: coefficient `0.000901` (raises CT win probability)
- `lag_15__CT_place_TSIDELOWER`: coefficient `0.000899` (raises CT win probability)
- `lag_14__CT_place_TSIDELOWER`: coefficient `-0.000832` (lowers CT win probability)
- `lag_00__CT_place_MAINHALL`: coefficient `-0.000825` (lowers CT win probability)
- `lag_01__T4__duck_amount`: coefficient `0.000794` (raises CT win probability)
- `lag_08__T_place_MAINHALL`: coefficient `0.000754` (raises CT win probability)
- `lag_12__T_place_MAINHALL`: coefficient `0.000733` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000723` (raises CT win probability)
- `lag_11__CT_place_TSIDELOWER`: coefficient `0.000722` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `31412`, seconds `59.50`, LSTM delta `+0.1437`

Top all feature movements:
- `lag_15__CT_place_TSIDELOWER`: contribution `+0.012214`
- `lag_14__CT_place_TSIDELOWER`: contribution `+0.011296`
- `lag_13__CT3__flash_duration`: contribution `+0.006280`
- `lag_13__T5__flash_duration`: contribution `+0.004908`
- `lag_06__CT_place_SIDEHALL`: contribution `+0.003856`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `+0.006280`
- `lag_13__T5__flash_duration`: contribution `+0.004908`
- `lag_03__T5__flash_duration`: contribution `+0.003831`
- `lag_00__CT3__flash_duration`: contribution `+0.003492`
- `lag_05__CT4__flash_duration`: contribution `+0.001798`

### tick `31924`, seconds `67.50`, LSTM delta `+0.1056`

Top all feature movements:
- `lag_11__CT_place_TSIDELOWER`: contribution `+0.009807`
- `lag_03__CT_place_TSIDELOWER`: contribution `+0.008315`
- `lag_01__CT5__flash_duration`: contribution `+0.006413`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.004071`
- `lag_01__CT_flash_duration_sum`: contribution `+0.003218`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `+0.006413`
- `lag_01__CT_flash_duration_sum`: contribution `+0.003218`
- `lag_06__T_A_site_active_infernos`: contribution `+0.001285`

### tick `32116`, seconds `70.50`, LSTM delta `+0.0780`

Top all feature movements:
- `lag_07__CT5__flash_duration`: contribution `+0.003564`
- `lag_12__T_place_MAINHALL`: contribution `+0.002645`
- `lag_04__CT5__duck_amount`: contribution `+0.002554`
- `lag_06__T_place_MAINHALL`: contribution `-0.002528`
- `lag_07__CT3__flash_duration`: contribution `+0.002446`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.003564`
- `lag_07__CT3__flash_duration`: contribution `+0.002446`
- `lag_04__T1__flash_duration`: contribution `+0.002227`
- `lag_00__T1__flash_duration`: contribution `+0.001907`
- `lag_03__T2__flash_duration`: contribution `+0.001428`

### tick `28148`, seconds `8.50`, LSTM delta `-0.0358`

Top all feature movements:
- `lag_03__CT_place_HOUSE`: contribution `-0.002212`
- `lag_13__CT_place_MAINHALL`: contribution `-0.002151`
- `lag_11__T_place_TUNNEL`: contribution `-0.002077`
- `lag_15__CT_place_BOMBSITEA`: contribution `-0.001643`
- `lag_15__CT_macro_A`: contribution `-0.001643`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001598`
- `lag_00__CT_active_infernos`: contribution `-0.000560`

### tick `32212`, seconds `72.00`, LSTM delta `+0.0309`

Top all feature movements:
- `lag_09__T_place_MAINHALL`: contribution `-0.004732`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004063`
- `lag_00__CT_kills_last_3s`: contribution `+0.002627`
- `lag_05__T_place_MAINHALL`: contribution `+0.002135`
- `lag_00__CT5__flash_duration`: contribution `-0.002131`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.002131`
- `lag_10__CT3__flash_duration`: contribution `+0.001380`
