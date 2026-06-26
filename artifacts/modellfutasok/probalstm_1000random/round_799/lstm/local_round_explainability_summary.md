# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `3`

## Largest probability jumps

- tick `25248`, seconds `27.50`, LSTM `0.0844`, delta `-0.1287`
- tick `25184`, seconds `26.50`, LSTM `0.2777`, delta `+0.0649`
- tick `25216`, seconds `27.00`, LSTM `0.2132`, delta `-0.0645`
- tick `24928`, seconds `22.50`, LSTM `0.2505`, delta `-0.0623`
- tick `24896`, seconds `22.00`, LSTM `0.3128`, delta `-0.0437`
- tick `25024`, seconds `24.00`, LSTM `0.2443`, delta `-0.0301`
- tick `25120`, seconds `25.50`, LSTM `0.1979`, delta `-0.0271`
- tick `24960`, seconds `23.00`, LSTM `0.2770`, delta `+0.0265`
- tick `24384`, seconds `14.00`, LSTM `0.4277`, delta `-0.0262`
- tick `25088`, seconds `25.00`, LSTM `0.2250`, delta `-0.0237`

## Top 15 local ridge features

- `lag_13__T_place_MAINHALL`: coefficient `-0.001004`, |coef| `0.001004`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000872`, |coef| `0.000872`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `0.000869`, |coef| `0.000869`
- `lag_15__CT_place_TSIDEUPPER`: coefficient `-0.000855`, |coef| `0.000855`
- `lag_04__T_flashed_players`: coefficient `0.000811`, |coef| `0.000811`
- `lag_14__CT_shots_fired_sum`: coefficient `0.000740`, |coef| `0.000740`
- `lag_13__T_flashed_players`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_03__CT_B_site_active_infernos`: coefficient `0.000715`, |coef| `0.000715`
- `lag_15__T_place_MAINHALL`: coefficient `-0.000715`, |coef| `0.000715`
- `lag_13__T2__flash_duration`: coefficient `-0.000693`, |coef| `0.000693`
- `lag_12__CT_shots_fired_sum`: coefficient `-0.000661`, |coef| `0.000661`
- `lag_11__T_bomb_zone_count`: coefficient `-0.000652`, |coef| `0.000652`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.000640`, |coef| `0.000640`
- `lag_04__CT4__duck_amount`: coefficient `-0.000626`, |coef| `0.000626`
- `lag_10__T5__flash_duration`: coefficient `0.000625`, |coef| `0.000625`

## Top 10 utility ridge features

- `lag_03__CT_B_site_active_infernos`: coefficient `0.000715` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.000693` (lowers CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.000625` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000612` (raises CT win probability)
- `lag_00__T_molly_inv`: coefficient `0.000567` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `0.000566` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000560` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `0.000546` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `0.000531` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `0.000525` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_MAINHALL`: coefficient `-0.001004` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000872` (raises CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `0.000869` (raises CT win probability)
- `lag_15__CT_place_TSIDEUPPER`: coefficient `-0.000855` (lowers CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.000811` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.000740` (raises CT win probability)
- `lag_13__T_flashed_players`: coefficient `-0.000731` (lowers CT win probability)
- `lag_15__T_place_MAINHALL`: coefficient `-0.000715` (lowers CT win probability)
- `lag_12__CT_shots_fired_sum`: coefficient `-0.000661` (lowers CT win probability)
- `lag_11__T_bomb_zone_count`: coefficient `-0.000652` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `25248`, seconds `27.50`, LSTM delta `-0.1287`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.010801`
- `lag_01__CT_shots_fired_sum`: contribution `-0.009088`
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.006529`
- `lag_10__T5__flash_duration`: contribution `-0.004359`
- `lag_01__CT2__shots_fired`: contribution `-0.004288`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.004359`
- `lag_13__T2__flash_duration`: contribution `-0.003287`
- `lag_04__T2__flash_duration`: contribution `-0.002395`

### tick `25184`, seconds `26.50`, LSTM delta `+0.0649`

Top all feature movements:
- `lag_12__CT_shots_fired_sum`: contribution `+0.009642`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004847`
- `lag_13__CT_shots_fired_sum`: contribution `+0.004288`
- `lag_14__CT_shots_fired_sum`: contribution `+0.004115`
- `lag_11__T_flashed_players`: contribution `+0.003309`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.002749`
- `lag_02__T2__flash_duration`: contribution `+0.001990`

### tick `25216`, seconds `27.00`, LSTM delta `-0.0645`

Top all feature movements:
- `lag_13__CT_shots_fired_sum`: contribution `-0.009004`
- `lag_14__CT_shots_fired_sum`: contribution `+0.005143`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.004809`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003871`
- `lag_15__CT_shots_fired_sum`: contribution `-0.003134`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.002316`
- `lag_12__T2__flash_duration`: contribution `-0.002079`
- `lag_12__T_flash_duration_sum`: contribution `-0.001309`
- `lag_15__T_A_site_active_infernos`: contribution `-0.001296`

### tick `24928`, seconds `22.50`, LSTM delta `-0.0623`

Top all feature movements:
- `lag_15__CT_place_TSIDEUPPER`: contribution `-0.006428`
- `lag_00__T5__flash_duration`: contribution `-0.004272`
- `lag_12__CT_place_TSIDEUPPER`: contribution `-0.003927`
- `lag_05__CT_shots_fired_sum`: contribution `-0.003687`
- `lag_12__T_place_MAINHALL`: contribution `+0.003365`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.004272`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001946`
- `lag_08__CT4__flash_duration`: contribution `-0.001710`
- `lag_13__T2__flash_duration`: contribution `-0.001572`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.001447`

### tick `24896`, seconds `22.00`, LSTM delta `-0.0437`

Top all feature movements:
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.006529`
- `lag_14__CT_place_TSIDEUPPER`: contribution `-0.003408`
- `lag_11__T_place_MAINHALL`: contribution `+0.003212`
- `lag_05__CT_shots_fired_sum`: contribution `-0.002950`
- `lag_03__CT3__shots_fired`: contribution `-0.002794`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `-0.001990`
- `lag_02__CT4__flash_duration`: contribution `-0.001635`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.001447`
- `lag_12__T5__flash_duration`: contribution `-0.001419`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.001355`
