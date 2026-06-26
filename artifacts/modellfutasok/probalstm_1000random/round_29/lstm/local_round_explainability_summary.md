# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-mibr-vs-heroic-bo3-wXQqD_9CDZgrp6ykBiT-3T/mibr-vs-heroic-m2-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `58811`, seconds `50.50`, LSTM `0.1630`, delta `-0.3116`
- tick `60667`, seconds `79.50`, LSTM `0.0285`, delta `-0.2580`
- tick `56315`, seconds `11.50`, LSTM `0.1113`, delta `-0.1834`
- tick `56347`, seconds `12.00`, LSTM `0.2697`, delta `+0.1583`
- tick `57915`, seconds `36.50`, LSTM `0.3089`, delta `-0.1394`
- tick `56411`, seconds `13.00`, LSTM `0.3912`, delta `+0.1364`
- tick `59803`, seconds `66.00`, LSTM `0.1681`, delta `+0.0892`
- tick `58011`, seconds `38.00`, LSTM `0.3808`, delta `+0.0824`
- tick `58843`, seconds `51.00`, LSTM `0.0995`, delta `-0.0635`
- tick `60347`, seconds `74.50`, LSTM `0.2875`, delta `+0.0602`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.006448`, |coef| `0.006448`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.004534`, |coef| `0.004534`
- `lag_00__T_kills_last_3s`: coefficient `-0.004114`, |coef| `0.004114`
- `lag_00__kill_diff_last_3s`: coefficient `0.003795`, |coef| `0.003795`
- `lag_11__CT_place_RAMP`: coefficient `-0.003476`, |coef| `0.003476`
- `lag_06__T_place_MAINHALL`: coefficient `0.003291`, |coef| `0.003291`
- `lag_00__T_damage_last_5s`: coefficient `-0.003117`, |coef| `0.003117`
- `lag_00__damage_diff_last_5s`: coefficient `0.003067`, |coef| `0.003067`
- `lag_02__T_place_MAINHALL`: coefficient `0.002952`, |coef| `0.002952`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002923`, |coef| `0.002923`
- `lag_00__CT3__alive`: coefficient `0.002793`, |coef| `0.002793`
- `lag_00__CT3__molly`: coefficient `0.002787`, |coef| `0.002787`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002539`, |coef| `0.002539`
- `lag_04__CT1__duck_amount`: coefficient `0.002407`, |coef| `0.002407`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002405`, |coef| `0.002405`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002923` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.002787` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002405` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.002269` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.002085` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.001611` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001315` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.001293` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001284` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001246` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.006448` (raises CT win probability)
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.004534` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004114` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003795` (raises CT win probability)
- `lag_11__CT_place_RAMP`: coefficient `-0.003476` (lowers CT win probability)
- `lag_06__T_place_MAINHALL`: coefficient `0.003291` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003117` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003067` (raises CT win probability)
- `lag_02__T_place_MAINHALL`: coefficient `0.002952` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.002793` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `58811`, seconds `50.50`, LSTM delta `-0.3116`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.048470`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `-0.018252`
- `lag_00__T_kills_last_3s`: contribution `-0.013033`
- `lag_06__T_place_MAINHALL`: contribution `-0.011881`
- `lag_02__T_place_MAINHALL`: contribution `-0.010655`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.006882`
- `lag_00__CT3__smoke`: contribution `-0.005019`

### tick `60667`, seconds `79.50`, LSTM delta `-0.2580`

Top all feature movements:
- `lag_07__CT_place_TSIDELOWER`: contribution `-0.026299`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.014802`
- `lag_00__T_kills_last_3s`: contribution `-0.013033`
- `lag_11__CT_place_RAMP`: contribution `-0.010385`
- `lag_00__damage_diff_last_5s`: contribution `-0.010034`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.014802`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.009988`
- `lag_00__CT5__flash_duration`: contribution `-0.009800`
- `lag_05__CT5__flash_duration`: contribution `-0.009510`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.006285`

### tick `56315`, seconds `11.50`, LSTM delta `-0.1834`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `-0.014678`
- `lag_00__T_kills_last_3s`: contribution `-0.013033`
- `lag_00__T_shots_fired_sum`: contribution `-0.009517`
- `lag_00__kill_diff_last_3s`: contribution `-0.009133`
- `lag_06__T2__flash_duration`: contribution `-0.007758`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `-0.014678`
- `lag_06__T2__flash_duration`: contribution `-0.007758`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007723`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.005211`
- `lag_00__CT4__flash_duration`: contribution `-0.003480`

### tick `56347`, seconds `12.00`, LSTM delta `+0.1583`

Top all feature movements:
- `lag_11__T_he_last_5s`: contribution `+0.016193`
- `lag_00__T_shots_fired_sum`: contribution `+0.009517`
- `lag_00__kill_diff_last_3s`: contribution `+0.009133`
- `lag_07__CT_place_TOPOFMID`: contribution `+0.008321`
- `lag_07__T2__flash_duration`: contribution `+0.007446`

Top utility-only movements:
- `lag_11__T_he_last_5s`: contribution `+0.016193`
- `lag_07__T2__flash_duration`: contribution `+0.007446`
- `lag_00__CT5__flash_duration`: contribution `-0.005361`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.005148`
- `lag_09__CT4__flash_duration`: contribution `+0.004238`

### tick `57915`, seconds `36.50`, LSTM delta `-0.1394`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.048470`
- `lag_09__CT1__duck_amount`: contribution `+0.005959`
- `lag_03__CT1__duck_amount`: contribution `-0.005587`
- `lag_03__CT1__is_walking`: contribution `-0.005551`
- `lag_01__T5__is_walking`: contribution `-0.004728`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.004555`
- `lag_01__T_B_site_active_infernos`: contribution `-0.002709`
