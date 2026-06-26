# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `6957`, seconds `54.00`, LSTM `0.9176`, delta `+0.0471`
- tick `5677`, seconds `34.00`, LSTM `0.8421`, delta `+0.0412`
- tick `4557`, seconds `16.50`, LSTM `0.8725`, delta `+0.0272`
- tick `5101`, seconds `25.00`, LSTM `0.8196`, delta `-0.0258`
- tick `5965`, seconds `38.50`, LSTM `0.8307`, delta `-0.0249`
- tick `5485`, seconds `31.00`, LSTM `0.8132`, delta `+0.0246`
- tick `6829`, seconds `52.00`, LSTM `0.8797`, delta `+0.0244`
- tick `5357`, seconds `29.00`, LSTM `0.7883`, delta `-0.0242`
- tick `5997`, seconds `39.00`, LSTM `0.8538`, delta `+0.0231`
- tick `5037`, seconds `24.00`, LSTM `0.8598`, delta `+0.0208`

## Top 15 local ridge features

- `lag_00__CT5__is_walking`: coefficient `-0.002173`, |coef| `0.002173`
- `lag_00__CT_kills_last_3s`: coefficient `0.001338`, |coef| `0.001338`
- `lag_08__CT2__flash_duration`: coefficient `-0.001275`, |coef| `0.001275`
- `lag_00__T_place_SILO`: coefficient `-0.001154`, |coef| `0.001154`
- `lag_00__kill_diff_last_3s`: coefficient `0.001116`, |coef| `0.001116`
- `lag_00__CT_walking_count`: coefficient `-0.001113`, |coef| `0.001113`
- `lag_00__T3__alive`: coefficient `-0.001102`, |coef| `0.001102`
- `lag_00__T3__hp`: coefficient `-0.001085`, |coef| `0.001085`
- `lag_00__damage_diff_last_5s`: coefficient `0.001070`, |coef| `0.001070`
- `lag_00__CT_damage_last_5s`: coefficient `0.001052`, |coef| `0.001052`
- `lag_14__T2__is_walking`: coefficient `0.000950`, |coef| `0.000950`
- `lag_00__T4__is_walking`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_14__CT2__duck_amount`: coefficient `0.000922`, |coef| `0.000922`
- `lag_00__CT_place_SECRET`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_14__T1__is_walking`: coefficient `-0.000746`, |coef| `0.000746`

## Top 10 utility ridge features

- `lag_08__CT2__flash_duration`: coefficient `-0.001275` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.000651` (lowers CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `-0.000582` (lowers CT win probability)
- `lag_11__CT2__flash`: coefficient `-0.000447` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000419` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000418` (raises CT win probability)
- `lag_12__CT2__flash`: coefficient `-0.000375` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000357` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000344` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000336` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT5__is_walking`: coefficient `-0.002173` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001338` (raises CT win probability)
- `lag_00__T_place_SILO`: coefficient `-0.001154` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001116` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.001113` (lowers CT win probability)
- `lag_00__T3__alive`: coefficient `-0.001102` (lowers CT win probability)
- `lag_00__T3__hp`: coefficient `-0.001085` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001070` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001052` (raises CT win probability)
- `lag_14__T2__is_walking`: coefficient `0.000950` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `6957`, seconds `54.00`, LSTM delta `+0.0471`

Top all feature movements:
- `lag_08__CT2__flash_duration`: contribution `+0.007783`
- `lag_00__CT_kills_last_3s`: contribution `+0.003864`
- `lag_14__CT2__duck_amount`: contribution `+0.003512`
- `lag_00__kill_diff_last_3s`: contribution `+0.002686`
- `lag_00__T3__alive`: contribution `+0.002665`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.007783`
- `lag_08__CT_flash_duration_sum`: contribution `+0.001614`

### tick `5677`, seconds `34.00`, LSTM delta `+0.0412`

Top all feature movements:
- `lag_00__CT5__is_walking`: contribution `+0.005208`
- `lag_02__T_place_TROPHY`: contribution `+0.004531`
- `lag_02__T_place_VENDING`: contribution `+0.002723`
- `lag_00__T4__is_walking`: contribution `+0.002136`
- `lag_09__T2__is_walking`: contribution `+0.001674`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4557`, seconds `16.50`, LSTM delta `+0.0272`

Top all feature movements:
- `lag_04__CT_place_OBSERVATION`: contribution `+0.008211`
- `lag_00__CT_place_OBSERVATION`: contribution `+0.007872`
- `lag_00__T4__is_walking`: contribution `+0.002136`
- `lag_09__T5__duck_amount`: contribution `+0.001644`
- `lag_12__CT1__duck_amount`: contribution `-0.001582`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5101`, seconds `25.00`, LSTM delta `-0.0258`

Top all feature movements:
- `lag_00__CT_place_SECRET`: contribution `-0.008324`
- `lag_14__T1__is_walking`: contribution `-0.001701`
- `lag_13__CT1__duck_amount`: contribution `-0.001676`
- `lag_00__CT2__is_walking`: contribution `+0.001420`
- `lag_12__CT_place_HELL`: contribution `-0.001305`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5965`, seconds `38.50`, LSTM delta `-0.0249`

Top all feature movements:
- `lag_00__CT5__is_walking`: contribution `-0.005208`
- `lag_14__CT2__duck_amount`: contribution `-0.002628`
- `lag_14__T2__is_walking`: contribution `-0.002182`
- `lag_00__CT_walking_count`: contribution `-0.001998`
- `lag_00__CT2__is_walking`: contribution `-0.001420`

Top utility-only movements:
- No utility movement among the top local contributors.
