# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `3`

## Largest probability jumps

- tick `14559`, seconds `35.00`, LSTM `0.2320`, delta `+0.0939`
- tick `12351`, seconds `0.50`, LSTM `0.1334`, delta `-0.0717`
- tick `15135`, seconds `44.00`, LSTM `0.1497`, delta `-0.0695`
- tick `14975`, seconds `41.50`, LSTM `0.2020`, delta `+0.0672`
- tick `14143`, seconds `28.50`, LSTM `0.1891`, delta `+0.0441`
- tick `14911`, seconds `40.50`, LSTM `0.1387`, delta `-0.0413`
- tick `16191`, seconds `60.50`, LSTM `0.1057`, delta `-0.0306`
- tick `12767`, seconds `7.00`, LSTM `0.1242`, delta `-0.0302`
- tick `14495`, seconds `34.00`, LSTM `0.1479`, delta `-0.0290`
- tick `15455`, seconds `49.00`, LSTM `0.1418`, delta `+0.0288`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001114`, |coef| `0.001114`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001114`, |coef| `0.001114`
- `lag_00__T4__shots_fired`: coefficient `-0.001040`, |coef| `0.001040`
- `lag_00__CT1__is_walking`: coefficient `0.000829`, |coef| `0.000829`
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.000761`, |coef| `0.000761`
- `lag_10__T5__is_walking`: coefficient `0.000755`, |coef| `0.000755`
- `lag_00__CT2__is_walking`: coefficient `-0.000723`, |coef| `0.000723`
- `lag_15__T_place_TSIDEUPPER`: coefficient `0.000712`, |coef| `0.000712`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000705`, |coef| `0.000705`
- `lag_00__T_velocity_mean`: coefficient `-0.000688`, |coef| `0.000688`
- `lag_05__T_shots_fired_sum`: coefficient `0.000681`, |coef| `0.000681`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.000662`, |coef| `0.000662`
- `lag_00__T1__duck_amount`: coefficient `-0.000660`, |coef| `0.000660`
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.000660`, |coef| `0.000660`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001114` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000705` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000555` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `0.000524` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `-0.000456` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `0.000438` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.000414` (raises CT win probability)
- `lag_02__CT4__smoke`: coefficient `0.000408` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000391` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000374` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001273` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001114` (lowers CT win probability)
- `lag_00__T4__shots_fired`: coefficient `-0.001040` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000829` (raises CT win probability)
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.000761` (raises CT win probability)
- `lag_10__T5__is_walking`: coefficient `0.000755` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.000723` (lowers CT win probability)
- `lag_15__T_place_TSIDEUPPER`: coefficient `0.000712` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000688` (lowers CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `0.000681` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14559`, seconds `35.00`, LSTM delta `+0.0939`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.018135`
- `lag_00__T4__shots_fired`: contribution `+0.012212`
- `lag_13__T_place_SIDEENTRANCE`: contribution `+0.002460`
- `lag_10__T2__duck_amount`: contribution `+0.002229`
- `lag_10__T5__is_walking`: contribution `+0.001752`

Top utility-only movements:
- `lag_10__T_B_site_active_smokes`: contribution `+0.001382`
- `lag_15__T_B_site_active_infernos`: contribution `+0.001170`
- `lag_14__T_B_site_active_infernos`: contribution `+0.001105`
- `lag_15__T_A_site_active_infernos`: contribution `+0.001056`
- `lag_10__T_A_site_active_smokes`: contribution `+0.000944`

### tick `12351`, seconds `0.50`, LSTM delta `-0.0717`

Top all feature movements:
- `lag_01__CT_place_MAINHALL`: contribution `-0.004551`
- `lag_01__T_place_TSPAWN`: contribution `-0.002703`
- `lag_00__T_velocity_mean`: contribution `-0.002280`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002168`
- `lag_01__CT_place_SIDEHALL`: contribution `-0.002062`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000755`
- `lag_01__T2__flash`: contribution `-0.000724`
- `lag_01__utility_inv_diff`: contribution `-0.000708`
- `lag_01__flash_inv_diff`: contribution `-0.000595`
- `lag_01__T5__flash`: contribution `-0.000555`

### tick `15135`, seconds `44.00`, LSTM delta `-0.0695`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `-0.012763`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.009546`
- `lag_05__T4__shots_fired`: contribution `-0.008415`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.003817`
- `lag_08__T5__is_walking`: contribution `-0.001426`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.009546`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.003817`
- `lag_02__CT4__smoke`: contribution `-0.000890`

### tick `14975`, seconds `41.50`, LSTM delta `+0.0672`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.023862`
- `lag_00__T4__shots_fired`: contribution `+0.016068`
- `lag_13__T_shots_fired_sum`: contribution `+0.003459`
- `lag_00__T1__duck_amount`: contribution `+0.002584`
- `lag_13__T4__shots_fired`: contribution `+0.002569`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.001105`

### tick `14143`, seconds `28.50`, LSTM delta `+0.0441`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.005439`
- `lag_13__T_place_SIDEENTRANCE`: contribution `+0.002460`
- `lag_00__CT1__is_walking`: contribution `+0.001935`
- `lag_00__T_place_TSIDEUPPER`: contribution `+0.001919`
- `lag_10__T5__is_walking`: contribution `+0.001752`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.001045`
