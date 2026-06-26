# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `58275`, seconds `46.00`, LSTM `0.8554`, delta `+0.2626`
- tick `57987`, seconds `41.50`, LSTM `0.6995`, delta `+0.1759`
- tick `58051`, seconds `42.50`, LSTM `0.5519`, delta `-0.1423`
- tick `57955`, seconds `41.00`, LSTM `0.5236`, delta `+0.1099`
- tick `57923`, seconds `40.50`, LSTM `0.4137`, delta `-0.0735`
- tick `58243`, seconds `45.50`, LSTM `0.5928`, delta `+0.0491`
- tick `57827`, seconds `39.00`, LSTM `0.5078`, delta `+0.0375`
- tick `58083`, seconds `43.00`, LSTM `0.5170`, delta `-0.0349`
- tick `58467`, seconds `49.00`, LSTM `0.9500`, delta `+0.0343`
- tick `58755`, seconds `53.50`, LSTM `0.9640`, delta `+0.0340`

## Top 15 local ridge features

- `lag_03__T1__flash_duration`: coefficient `0.001726`, |coef| `0.001726`
- `lag_07__T1__flash_duration`: coefficient `0.001674`, |coef| `0.001674`
- `lag_13__T1__flash_duration`: coefficient `0.001663`, |coef| `0.001663`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001422`, |coef| `0.001422`
- `lag_04__T1__flash_duration`: coefficient `0.001421`, |coef| `0.001421`
- `lag_05__T3__duck_amount`: coefficient `0.001217`, |coef| `0.001217`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_00__T3__is_scoped`: coefficient `0.001149`, |coef| `0.001149`
- `lag_02__T_place_CONNECTOR`: coefficient `0.001138`, |coef| `0.001138`
- `lag_10__T_shots_fired_sum`: coefficient `-0.001094`, |coef| `0.001094`
- `lag_07__CT5__duck_amount`: coefficient `0.001089`, |coef| `0.001089`
- `lag_03__T_shots_fired_sum`: coefficient `0.001089`, |coef| `0.001089`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001079`, |coef| `0.001079`
- `lag_07__CT_place_JUNGLE`: coefficient `-0.001054`, |coef| `0.001054`
- `lag_07__T2__flash_duration`: coefficient `0.001054`, |coef| `0.001054`

## Top 10 utility ridge features

- `lag_03__T1__flash_duration`: coefficient `0.001726` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.001674` (raises CT win probability)
- `lag_13__T1__flash_duration`: coefficient `0.001663` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.001421` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001054` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.000946` (raises CT win probability)
- `lag_12__T1__flash_duration`: coefficient `0.000925` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.000874` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.000793` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000752` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001422` (lowers CT win probability)
- `lag_05__T3__duck_amount`: coefficient `0.001217` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001217` (lowers CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.001149` (raises CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `0.001138` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `-0.001094` (lowers CT win probability)
- `lag_07__CT5__duck_amount`: coefficient `0.001089` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `0.001089` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.001079` (raises CT win probability)
- `lag_07__CT_place_JUNGLE`: coefficient `-0.001054` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `58275`, seconds `46.00`, LSTM delta `+0.2626`

Top all feature movements:
- `lag_13__T1__flash_duration`: contribution `+0.012135`
- `lag_07__T1__flash_duration`: contribution `+0.011633`
- `lag_10__T_shots_fired_sum`: contribution `+0.009025`
- `lag_07__T2__flash_duration`: contribution `+0.007003`
- `lag_07__CT_place_JUNGLE`: contribution `+0.006764`

Top utility-only movements:
- `lag_13__T1__flash_duration`: contribution `+0.012135`
- `lag_07__T1__flash_duration`: contribution `+0.011633`
- `lag_07__T2__flash_duration`: contribution `+0.007003`
- `lag_00__CT3__flash_duration`: contribution `+0.003006`

### tick `57987`, seconds `41.50`, LSTM delta `+0.1759`

Top all feature movements:
- `lag_04__T1__flash_duration`: contribution `+0.010363`
- `lag_01__T_shots_fired_sum`: contribution `+0.010035`
- `lag_04__T3__is_scoped`: contribution `+0.006713`
- `lag_01__T1__shots_fired`: contribution `+0.005486`
- `lag_00__T1__flash_duration`: contribution `+0.005311`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.010363`
- `lag_00__T1__flash_duration`: contribution `+0.005311`

### tick `58051`, seconds `42.50`, LSTM delta `-0.1423`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.008978`
- `lag_00__T3__is_scoped`: contribution `-0.007374`
- `lag_00__CT_place_JUNGLE`: contribution `-0.006920`
- `lag_02__T1__flash_duration`: contribution `-0.006902`
- `lag_07__CT_place_JUNGLE`: contribution `-0.006764`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.006902`
- `lag_00__T1__flash_duration`: contribution `-0.005059`
- `lag_06__T1__flash_duration`: contribution `+0.004947`
- `lag_00__CT3__flash_duration`: contribution `-0.003006`
- `lag_00__T2__flash_duration`: contribution `-0.002899`

### tick `57955`, seconds `41.00`, LSTM delta `+0.1099`

Top all feature movements:
- `lag_03__T1__flash_duration`: contribution `+0.012589`
- `lag_00__T_shots_fired_sum`: contribution `+0.011729`
- `lag_03__T3__is_scoped`: contribution `+0.006380`
- `lag_00__T1__shots_fired`: contribution `+0.005533`
- `lag_04__CT_place_JUNGLE`: contribution `+0.005062`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.012589`

### tick `57923`, seconds `40.50`, LSTM delta `-0.0735`

Top all feature movements:
- `lag_02__T1__flash_duration`: contribution `+0.006902`
- `lag_00__T_shots_fired_sum`: contribution `-0.005331`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.004573`
- `lag_01__T_shots_fired_sum`: contribution `-0.004561`
- `lag_07__CT5__duck_amount`: contribution `-0.004075`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.006902`
