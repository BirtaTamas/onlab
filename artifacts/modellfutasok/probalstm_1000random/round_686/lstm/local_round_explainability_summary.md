# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `37950`, seconds `42.00`, LSTM `0.2082`, delta `-0.2699`
- tick `37726`, seconds `38.50`, LSTM `0.3253`, delta `-0.1926`
- tick `37886`, seconds `41.00`, LSTM `0.4478`, delta `-0.1863`
- tick `37758`, seconds `39.00`, LSTM `0.4984`, delta `+0.1731`
- tick `37854`, seconds `40.50`, LSTM `0.6341`, delta `+0.1255`
- tick `37694`, seconds `38.00`, LSTM `0.5179`, delta `-0.0636`
- tick `38046`, seconds `43.50`, LSTM `0.0933`, delta `-0.0547`
- tick `37982`, seconds `42.50`, LSTM `0.1675`, delta `-0.0406`
- tick `37918`, seconds `41.50`, LSTM `0.4781`, delta `+0.0303`
- tick `36030`, seconds `12.00`, LSTM `0.5127`, delta `+0.0258`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002443`, |coef| `0.002443`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002249`, |coef| `0.002249`
- `lag_04__CT2__duck_amount`: coefficient `-0.001760`, |coef| `0.001760`
- `lag_00__T5__shots_fired`: coefficient `-0.001625`, |coef| `0.001625`
- `lag_15__CT2__is_walking`: coefficient `-0.001601`, |coef| `0.001601`
- `lag_00__CT_kills_last_3s`: coefficient `0.001575`, |coef| `0.001575`
- `lag_09__CT2__duck_amount`: coefficient `-0.001542`, |coef| `0.001542`
- `lag_11__T5__duck_amount`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_00__T_kills_last_3s`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_00__T5__duck_amount`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_01__T4__shots_fired`: coefficient `-0.001284`, |coef| `0.001284`
- `lag_09__CT1__is_walking`: coefficient `0.001283`, |coef| `0.001283`
- `lag_14__CT4__duck_amount`: coefficient `0.001276`, |coef| `0.001276`
- `lag_01__CT1__shots_fired`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_08__T1__is_walking`: coefficient `0.001271`, |coef| `0.001271`

## Top 10 utility ridge features

- `lag_02__CT2__flash`: coefficient `0.000843` (raises CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.000801` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000784` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000706` (raises CT win probability)
- `lag_06__T1__flash`: coefficient `0.000653` (raises CT win probability)
- `lag_10__T3__smoke`: coefficient `-0.000616` (lowers CT win probability)
- `lag_03__T4__molly`: coefficient `0.000604` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000594` (raises CT win probability)
- `lag_06__T1__utility_total`: coefficient `0.000562` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000525` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002443` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002249` (raises CT win probability)
- `lag_04__CT2__duck_amount`: coefficient `-0.001760` (lowers CT win probability)
- `lag_00__T5__shots_fired`: coefficient `-0.001625` (lowers CT win probability)
- `lag_15__CT2__is_walking`: coefficient `-0.001601` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001575` (raises CT win probability)
- `lag_09__CT2__duck_amount`: coefficient `-0.001542` (lowers CT win probability)
- `lag_11__T5__duck_amount`: coefficient `-0.001500` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001486` (lowers CT win probability)
- `lag_00__T5__duck_amount`: coefficient `-0.001406` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `37950`, seconds `42.00`, LSTM delta `-0.2699`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011759`
- `lag_00__CT_shots_fired_sum`: contribution `-0.009376`
- `lag_01__CT_shots_fired_sum`: contribution `-0.005282`
- `lag_00__T_kills_last_3s`: contribution `-0.004709`
- `lag_00__CT_kills_last_3s`: contribution `-0.004549`

Top utility-only movements:
- `lag_00__CT_flash_alpha_mean`: contribution `-0.003074`

### tick `37726`, seconds `38.50`, LSTM delta `-0.1926`

Top all feature movements:
- `lag_09__CT_place_BALCONY`: contribution `-0.007446`
- `lag_04__CT2__duck_amount`: contribution `-0.006704`
- `lag_00__kill_diff_last_3s`: contribution `-0.005880`
- `lag_11__T5__duck_amount`: contribution `-0.005696`
- `lag_09__CT2__duck_amount`: contribution `-0.005232`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37886`, seconds `41.00`, LSTM delta `-0.1863`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012501`
- `lag_01__CT_shots_fired_sum`: contribution `-0.007043`
- `lag_04__CT2__duck_amount`: contribution `-0.006704`
- `lag_09__CT2__duck_amount`: contribution `-0.005876`
- `lag_00__T5__duck_amount`: contribution `-0.005338`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.002834`

### tick `37758`, seconds `39.00`, LSTM delta `+0.1731`

Top all feature movements:
- `lag_00__T5__shots_fired`: contribution `+0.006991`
- `lag_04__CT2__duck_amount`: contribution `+0.006704`
- `lag_10__CT_place_BALCONY`: contribution `+0.006518`
- `lag_00__T_shots_fired_sum`: contribution `+0.006380`
- `lag_00__kill_diff_last_3s`: contribution `+0.005880`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37854`, seconds `40.50`, LSTM delta `+0.1255`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012501`
- `lag_00__kill_diff_last_3s`: contribution `+0.005880`
- `lag_07__CT2__duck_amount`: contribution `+0.004754`
- `lag_14__CT4__duck_amount`: contribution `+0.004685`
- `lag_00__CT_kills_last_3s`: contribution `+0.004549`

Top utility-only movements:
- No utility movement among the top local contributors.
