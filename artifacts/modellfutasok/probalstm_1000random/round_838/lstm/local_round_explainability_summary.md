# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `99090`, seconds `84.50`, LSTM `0.3473`, delta `-0.2137`
- tick `99122`, seconds `85.00`, LSTM `0.1376`, delta `-0.2097`
- tick `97618`, seconds `61.50`, LSTM `0.6015`, delta `-0.1950`
- tick `98866`, seconds `81.00`, LSTM `0.5813`, delta `-0.1833`
- tick `97394`, seconds `58.00`, LSTM `0.8614`, delta `+0.1776`
- tick `98802`, seconds `80.00`, LSTM `0.7175`, delta `+0.1596`
- tick `99154`, seconds `85.50`, LSTM `0.0632`, delta `-0.0745`
- tick `94258`, seconds `9.00`, LSTM `0.8032`, delta `+0.0610`
- tick `98834`, seconds `80.50`, LSTM `0.7646`, delta `+0.0470`
- tick `95922`, seconds `35.00`, LSTM `0.8090`, delta `-0.0383`

## Top 15 local ridge features

- `lag_00__CT_place_GRAVEYARD`: coefficient `0.003456`, |coef| `0.003456`
- `lag_01__CT_place_GRAVEYARD`: coefficient `0.003290`, |coef| `0.003290`
- `lag_00__kill_diff_last_3s`: coefficient `0.003042`, |coef| `0.003042`
- `lag_11__T_place_ARCH`: coefficient `0.002886`, |coef| `0.002886`
- `lag_00__damage_diff_last_5s`: coefficient `0.002609`, |coef| `0.002609`
- `lag_11__CT4__is_walking`: coefficient `0.002430`, |coef| `0.002430`
- `lag_00__CT_kills_last_3s`: coefficient `0.002419`, |coef| `0.002419`
- `lag_06__CT1__flash_duration`: coefficient `0.002150`, |coef| `0.002150`
- `lag_14__CT4__duck_amount`: coefficient `0.001979`, |coef| `0.001979`
- `lag_13__CT1__duck_amount`: coefficient `0.001964`, |coef| `0.001964`
- `lag_00__CT_place_RUINS`: coefficient `0.001934`, |coef| `0.001934`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001914`, |coef| `0.001914`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001836`, |coef| `0.001836`
- `lag_02__T1__duck_amount`: coefficient `0.001778`, |coef| `0.001778`
- `lag_01__CT4__shots_fired`: coefficient `0.001689`, |coef| `0.001689`

## Top 10 utility ridge features

- `lag_06__CT1__flash_duration`: coefficient `0.002150` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001162` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000842` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000794` (lowers CT win probability)
- `lag_02__T1__smoke`: coefficient `0.000789` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000788` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000776` (raises CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `-0.000773` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000752` (raises CT win probability)
- `lag_10__T1__smoke`: coefficient `0.000749` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_GRAVEYARD`: coefficient `0.003456` (raises CT win probability)
- `lag_01__CT_place_GRAVEYARD`: coefficient `0.003290` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003042` (raises CT win probability)
- `lag_11__T_place_ARCH`: coefficient `0.002886` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002609` (raises CT win probability)
- `lag_11__CT4__is_walking`: coefficient `0.002430` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002419` (raises CT win probability)
- `lag_14__CT4__duck_amount`: coefficient `0.001979` (raises CT win probability)
- `lag_13__CT1__duck_amount`: coefficient `0.001964` (raises CT win probability)
- `lag_00__CT_place_RUINS`: coefficient `0.001934` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `99090`, seconds `84.50`, LSTM delta `-0.2137`

Top all feature movements:
- `lag_00__CT_place_GRAVEYARD`: contribution `-0.105129`
- `lag_00__kill_diff_last_3s`: contribution `-0.007323`
- `lag_00__T_shots_fired_sum`: contribution `-0.004995`
- `lag_00__damage_diff_last_5s`: contribution `-0.004356`
- `lag_00__T_kills_last_3s`: contribution `-0.004276`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99122`, seconds `85.00`, LSTM delta `-0.2097`

Top all feature movements:
- `lag_01__CT_place_GRAVEYARD`: contribution `-0.100087`
- `lag_00__CT_place_RUINS`: contribution `-0.006758`
- `lag_00__T_shots_fired_sum`: contribution `+0.005994`
- `lag_11__CT4__is_walking`: contribution `-0.005794`
- `lag_12__T4__is_walking`: contribution `-0.003639`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97618`, seconds `61.50`, LSTM delta `-0.1950`

Top all feature movements:
- `lag_05__T_place_ARCH`: contribution `-0.012124`
- `lag_02__T_place_ARCH`: contribution `-0.011724`
- `lag_00__damage_diff_last_5s`: contribution `-0.005886`
- `lag_06__CT3__duck_amount`: contribution `-0.005502`
- `lag_00__T_shots_fired_sum`: contribution `-0.004995`

Top utility-only movements:
- `lag_13__CT1__flash_duration`: contribution `-0.004627`

### tick `98866`, seconds `81.00`, LSTM delta `-0.1833`

Top all feature movements:
- `lag_11__T_place_ARCH`: contribution `-0.026854`
- `lag_01__CT_shots_fired_sum`: contribution `-0.007977`
- `lag_00__kill_diff_last_3s`: contribution `-0.007323`
- `lag_02__T1__duck_amount`: contribution `-0.006962`
- `lag_11__CT4__is_walking`: contribution `-0.005794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `97394`, seconds `58.00`, LSTM delta `+0.1776`

Top all feature movements:
- `lag_06__CT1__flash_duration`: contribution `+0.016990`
- `lag_11__CT_place_LIBRARY`: contribution `+0.007842`
- `lag_00__kill_diff_last_3s`: contribution `+0.007323`
- `lag_00__CT_kills_last_3s`: contribution `+0.006985`
- `lag_05__CT1__is_scoped`: contribution `+0.006477`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.016990`
- `lag_06__CT_flash_duration_sum`: contribution `+0.004125`
