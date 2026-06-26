# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `65452`, seconds `20.50`, LSTM `0.7523`, delta `+0.3195`
- tick `65068`, seconds `14.50`, LSTM `0.7387`, delta `+0.1854`
- tick `65196`, seconds `16.50`, LSTM `0.6086`, delta `-0.1434`
- tick `65804`, seconds `26.00`, LSTM `0.4353`, delta `-0.1208`
- tick `65420`, seconds `20.00`, LSTM `0.4328`, delta `-0.1187`
- tick `65484`, seconds `21.00`, LSTM `0.6520`, delta `-0.1003`
- tick `65612`, seconds `23.00`, LSTM `0.6856`, delta `+0.0765`
- tick `65100`, seconds `15.00`, LSTM `0.8066`, delta `+0.0679`
- tick `65772`, seconds `25.50`, LSTM `0.5561`, delta `-0.0601`
- tick `68652`, seconds `70.50`, LSTM `0.1004`, delta `-0.0586`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002176`, |coef| `0.002176`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002070`, |coef| `0.002070`
- `lag_00__kill_diff_last_3s`: coefficient `0.001979`, |coef| `0.001979`
- `lag_00__CT4__is_walking`: coefficient `-0.001730`, |coef| `0.001730`
- `lag_00__CT_place_TSPAWN`: coefficient `-0.001724`, |coef| `0.001724`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001705`, |coef| `0.001705`
- `lag_14__CT_place_SIDEHALL`: coefficient `0.001702`, |coef| `0.001702`
- `lag_13__T_place_TSIDEUPPER`: coefficient `0.001668`, |coef| `0.001668`
- `lag_11__T_place_TSIDEUPPER`: coefficient `0.001668`, |coef| `0.001668`
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001662`, |coef| `0.001662`
- `lag_04__T4__duck_amount`: coefficient `0.001545`, |coef| `0.001545`
- `lag_00__CT1__is_scoped`: coefficient `0.001530`, |coef| `0.001530`
- `lag_12__CT_place_TSIDEUPPER`: coefficient `0.001528`, |coef| `0.001528`
- `lag_12__CT3__duck_amount`: coefficient `0.001527`, |coef| `0.001527`
- `lag_02__T4__is_walking`: coefficient `-0.001476`, |coef| `0.001476`

## Top 10 utility ridge features

- `lag_09__CT_B_site_active_infernos`: coefficient `0.001322` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.001266` (raises CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.001265` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `-0.001188` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.001092` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `-0.001078` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.001055` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.001041` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001025` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.001014` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002176` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002070` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001979` (raises CT win probability)
- `lag_00__CT4__is_walking`: coefficient `-0.001730` (lowers CT win probability)
- `lag_00__CT_place_TSPAWN`: coefficient `-0.001724` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001705` (raises CT win probability)
- `lag_14__CT_place_SIDEHALL`: coefficient `0.001702` (raises CT win probability)
- `lag_13__T_place_TSIDEUPPER`: coefficient `0.001668` (raises CT win probability)
- `lag_11__T_place_TSIDEUPPER`: coefficient `0.001668` (raises CT win probability)
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001662` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65452`, seconds `20.50`, LSTM delta `+0.3195`

Top all feature movements:
- `lag_12__CT_place_TSIDEUPPER`: contribution `+0.011485`
- `lag_09__T5__flash_duration`: contribution `+0.010080`
- `lag_07__T_shots_fired_sum`: contribution `+0.009607`
- `lag_10__CT3__shots_fired`: contribution `+0.007809`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007189`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `+0.010080`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.007125`
- `lag_00__T2__flash_duration`: contribution `+0.004935`
- `lag_09__T2__flash_duration`: contribution `+0.004805`

### tick `65068`, seconds `14.50`, LSTM delta `+0.1854`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.016354`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.009085`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007189`
- `lag_06__CT2__flash_duration`: contribution `+0.006302`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.006078`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.009085`
- `lag_06__CT2__flash_duration`: contribution `+0.006302`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.006078`
- `lag_02__T_utility_damage_last_5s`: contribution `+0.005222`
- `lag_09__CT_active_infernos`: contribution `+0.004256`

### tick `65196`, seconds `16.50`, LSTM delta `-0.1434`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.016354`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.012490`
- `lag_01__T5__flash_duration`: contribution `-0.008409`
- `lag_04__T4__duck_amount`: contribution `+0.005713`
- `lag_04__CT_place_SIDEENTRANCE`: contribution `-0.005533`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `-0.008409`
- `lag_07__CT2__flash_duration`: contribution `-0.003112`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.002950`

### tick `65804`, seconds `26.00`, LSTM delta `-0.1208`

Top all feature movements:
- `lag_08__CT2__flash_duration`: contribution `-0.008570`
- `lag_00__T_shots_fired_sum`: contribution `-0.007043`
- `lag_00__kill_diff_last_3s`: contribution `-0.004764`
- `lag_11__T2__flash_duration`: contribution `-0.004577`
- `lag_00__CT4__is_walking`: contribution `+0.004125`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.008570`
- `lag_11__T2__flash_duration`: contribution `-0.004577`
- `lag_08__CT_flash_duration_sum`: contribution `-0.002013`

### tick `65420`, seconds `20.00`, LSTM delta `-0.1187`

Top all feature movements:
- `lag_08__T5__flash_duration`: contribution `-0.007875`
- `lag_11__CT_place_TSIDEUPPER`: contribution `-0.006026`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.005764`
- `lag_06__T_shots_fired_sum`: contribution `-0.005684`
- `lag_08__T2__duck_amount`: contribution `-0.004784`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `-0.007875`
- `lag_13__T_utility_damage_last_5s`: contribution `-0.005764`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.004542`
