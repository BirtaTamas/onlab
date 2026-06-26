# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `119748`, seconds `39.00`, LSTM `0.9309`, delta `+0.0917`
- tick `119428`, seconds `34.00`, LSTM `0.8311`, delta `-0.0311`
- tick `119300`, seconds `32.00`, LSTM `0.8414`, delta `-0.0251`
- tick `119652`, seconds `37.50`, LSTM `0.8541`, delta `+0.0239`
- tick `119684`, seconds `38.00`, LSTM `0.8305`, delta `-0.0236`
- tick `117348`, seconds `1.50`, LSTM `0.8884`, delta `-0.0221`
- tick `119812`, seconds `40.00`, LSTM `0.9602`, delta `+0.0199`
- tick `119140`, seconds `29.50`, LSTM `0.8869`, delta `-0.0194`
- tick `117284`, seconds `0.50`, LSTM `0.9187`, delta `+0.0186`
- tick `117444`, seconds `3.00`, LSTM `0.8543`, delta `-0.0175`

## Top 15 local ridge features

- `lag_00__CT_place_HOLE`: coefficient `0.001065`, |coef| `0.001065`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001028`, |coef| `0.001028`
- `lag_03__CT_place_HOLE`: coefficient `0.000774`, |coef| `0.000774`
- `lag_14__CT_place_EXTENDEDA`: coefficient `-0.000772`, |coef| `0.000772`
- `lag_08__T_place_TUNNELSTAIRS`: coefficient `0.000721`, |coef| `0.000721`
- `lag_00__CT2__duck_amount`: coefficient `0.000544`, |coef| `0.000544`
- `lag_04__T_place_TUNNELSTAIRS`: coefficient `-0.000527`, |coef| `0.000527`
- `lag_00__T4__duck_amount`: coefficient `0.000504`, |coef| `0.000504`
- `lag_00__CT_place_UNDERA`: coefficient `-0.000495`, |coef| `0.000495`
- `lag_14__T3__duck_amount`: coefficient `0.000491`, |coef| `0.000491`
- `lag_15__CT_place_EXTENDEDA`: coefficient `-0.000486`, |coef| `0.000486`
- `lag_01__CT_velocity_mean`: coefficient `-0.000449`, |coef| `0.000449`
- `lag_04__CT1__is_walking`: coefficient `0.000439`, |coef| `0.000439`
- `lag_08__CT_place_ARAMP`: coefficient `0.000438`, |coef| `0.000438`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000422`, |coef| `0.000422`

## Top 10 utility ridge features

- `lag_10__CT5__molly`: coefficient `0.000255` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `-0.000210` (lowers CT win probability)
- `lag_11__CT5__flash`: coefficient `0.000207` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000192` (raises CT win probability)
- `lag_03__utility_inv_diff`: coefficient `-0.000188` (lowers CT win probability)
- `lag_03__smoke_inv_diff`: coefficient `-0.000185` (lowers CT win probability)
- `lag_03__CT4__molly`: coefficient `-0.000176` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `0.000171` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `-0.000170` (lowers CT win probability)
- `lag_10__CT5__utility_total`: coefficient `0.000170` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_HOLE`: coefficient `0.001065` (raises CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001028` (lowers CT win probability)
- `lag_03__CT_place_HOLE`: coefficient `0.000774` (raises CT win probability)
- `lag_14__CT_place_EXTENDEDA`: coefficient `-0.000772` (lowers CT win probability)
- `lag_08__T_place_TUNNELSTAIRS`: coefficient `0.000721` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.000544` (raises CT win probability)
- `lag_04__T_place_TUNNELSTAIRS`: coefficient `-0.000527` (lowers CT win probability)
- `lag_00__T4__duck_amount`: coefficient `0.000504` (raises CT win probability)
- `lag_00__CT_place_UNDERA`: coefficient `-0.000495` (lowers CT win probability)
- `lag_14__T3__duck_amount`: coefficient `0.000491` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `119748`, seconds `39.00`, LSTM delta `+0.0917`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.011887`
- `lag_03__CT_place_HOLE`: contribution `+0.008646`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.005036`
- `lag_00__T_place_MIDDOORS`: contribution `+0.004371`
- `lag_02__CT_place_HOLE`: contribution `+0.004342`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119428`, seconds `34.00`, LSTM delta `-0.0311`

Top all feature movements:
- `lag_00__T_place_MIDDOORS`: contribution `-0.004371`
- `lag_00__CT2__duck_amount`: contribution `-0.002073`
- `lag_15__T_place_LOWERTUNNEL`: contribution `-0.001811`
- `lag_09__T_place_LOWERTUNNEL`: contribution `-0.001713`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.001235`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119300`, seconds `32.00`, LSTM delta `-0.0251`

Top all feature movements:
- `lag_09__T_place_LOWERTUNNEL`: contribution `-0.001713`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.001397`
- `lag_08__T_place_OUTSIDETUNNEL`: contribution `-0.001379`
- `lag_02__T_place_MIDDOORS`: contribution `-0.001292`
- `lag_12__T5__duck_amount`: contribution `-0.001262`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119652`, seconds `37.50`, LSTM delta `+0.0239`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.011887`
- `lag_11__CT_place_EXTENDEDA`: contribution `+0.001498`
- `lag_13__T_place_MIDDOORS`: contribution `+0.001396`
- `lag_03__CT2__duck_amount`: contribution `+0.001157`
- `lag_14__T_place_LOWERTUNNEL`: contribution `+0.000840`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119684`, seconds `38.00`, LSTM delta `-0.0236`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.011887`
- `lag_15__T_place_LOWERTUNNEL`: contribution `-0.001811`
- `lag_12__CT_place_EXTENDEDA`: contribution `+0.001582`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.001575`
- `lag_03__CT2__duck_amount`: contribution `-0.001157`

Top utility-only movements:
- No utility movement among the top local contributors.
