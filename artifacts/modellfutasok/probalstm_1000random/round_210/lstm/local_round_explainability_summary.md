# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `7`

## Largest probability jumps

- tick `55034`, seconds `30.00`, LSTM `0.0866`, delta `-0.1800`
- tick `55066`, seconds `30.50`, LSTM `0.0171`, delta `-0.0695`
- tick `53146`, seconds `0.50`, LSTM `0.0813`, delta `-0.0670`
- tick `54810`, seconds `26.50`, LSTM `0.2171`, delta `+0.0500`
- tick `53818`, seconds `11.00`, LSTM `0.0663`, delta `-0.0464`
- tick `54938`, seconds `28.50`, LSTM `0.2556`, delta `-0.0348`
- tick `54874`, seconds `27.50`, LSTM `0.2649`, delta `+0.0268`
- tick `54906`, seconds `28.00`, LSTM `0.2903`, delta `+0.0255`
- tick `54458`, seconds `21.00`, LSTM `0.1083`, delta `+0.0240`
- tick `54842`, seconds `27.00`, LSTM `0.2381`, delta `+0.0209`

## Top 15 local ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.001562`, |coef| `0.001562`
- `lag_03__CT_place_ELECTRICALBOX`: coefficient `0.001217`, |coef| `0.001217`
- `lag_05__CT_place_LONGDOG`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_14__T_place_LONGDOG`: coefficient `-0.001086`, |coef| `0.001086`
- `lag_12__T_place_LONGDOG`: coefficient `-0.001044`, |coef| `0.001044`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001002`, |coef| `0.001002`
- `lag_03__CT_place_LONGDOG`: coefficient `-0.000980`, |coef| `0.000980`
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `-0.000919`, |coef| `0.000919`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000833`, |coef| `0.000833`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000786`, |coef| `0.000786`
- `lag_00__T_kills_last_3s`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_15__T_place_LONGDOG`: coefficient `-0.000725`, |coef| `0.000725`
- `lag_06__CT4__duck_amount`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_01__CT_place_LONGDOG`: coefficient `0.000678`, |coef| `0.000678`
- `lag_13__T_place_LONGDOG`: coefficient `-0.000677`, |coef| `0.000677`

## Top 10 utility ridge features

- `lag_00__CT4__smoke`: coefficient `0.000485` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.000463` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000417` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000380` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000380` (lowers CT win probability)
- `lag_11__CT4__flash_duration`: coefficient `-0.000346` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000326` (lowers CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000324` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000322` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000318` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LONGDOG`: coefficient `0.001562` (raises CT win probability)
- `lag_03__CT_place_ELECTRICALBOX`: coefficient `0.001217` (raises CT win probability)
- `lag_05__CT_place_LONGDOG`: coefficient `-0.001099` (lowers CT win probability)
- `lag_14__T_place_LONGDOG`: coefficient `-0.001086` (lowers CT win probability)
- `lag_12__T_place_LONGDOG`: coefficient `-0.001044` (lowers CT win probability)
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001002` (raises CT win probability)
- `lag_03__CT_place_LONGDOG`: coefficient `-0.000980` (lowers CT win probability)
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `-0.000919` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000833` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000786` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `55034`, seconds `30.00`, LSTM delta `-0.1800`

Top all feature movements:
- `lag_03__CT_place_ELECTRICALBOX`: contribution `-0.014147`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.010686`
- `lag_00__CT_place_LONGDOG`: contribution `-0.010186`
- `lag_05__CT_place_LONGDOG`: contribution `-0.007167`
- `lag_03__CT_place_LONGDOG`: contribution `-0.006390`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55066`, seconds `30.50`, LSTM delta `-0.0695`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.010186`
- `lag_01__CT_place_LONGDOG`: contribution `-0.004423`
- `lag_06__CT_place_LONGDOG`: contribution `-0.004293`
- `lag_15__T_place_LONGDOG`: contribution `-0.003373`
- `lag_13__T_place_LONGDOG`: contribution `-0.003153`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53146`, seconds `0.50`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `-0.003481`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002969`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002232`
- `lag_01__T_closest_enemy_dist`: contribution `-0.002155`
- `lag_01__centroid_distance_xy`: contribution `-0.001953`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001190`
- `lag_01__molly_inv_diff`: contribution `-0.001061`
- `lag_01__T_utility_inv`: contribution `-0.000775`
- `lag_01__T5__utility_total`: contribution `-0.000720`
- `lag_01__T_smoke_inv`: contribution `-0.000719`

### tick `54810`, seconds `26.50`, LSTM delta `+0.0500`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.011644`
- `lag_07__T_place_LONGDOG`: contribution `+0.002185`
- `lag_11__CT4__flash_duration`: contribution `+0.001979`
- `lag_02__CT3__duck_amount`: contribution `+0.001845`
- `lag_05__T_place_LONGDOG`: contribution `+0.001813`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `+0.001979`
- `lag_12__CT1__flash_duration`: contribution `+0.001549`
- `lag_14__CT3__flash_duration`: contribution `+0.001187`

### tick `53818`, seconds `11.00`, LSTM delta `-0.0464`

Top all feature movements:
- `lag_08__T_place_DUMPSTER`: contribution `-0.003575`
- `lag_12__T_place_TSTAIRS`: contribution `-0.002945`
- `lag_02__T_place_DUMPSTER`: contribution `-0.002637`
- `lag_09__T_place_TSTAIRS`: contribution `-0.002583`
- `lag_15__CT_place_ENTRANCE`: contribution `-0.002519`

Top utility-only movements:
- `lag_02__CT_flash_duration_sum`: contribution `-0.001447`
- `lag_02__T_A_site_active_infernos`: contribution `-0.001130`
- `lag_02__CT1__flash_duration`: contribution `-0.001070`
- `lag_02__CT5__flash_duration`: contribution `-0.000824`
