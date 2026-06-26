# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `18`

## Largest probability jumps

- tick `140362`, seconds `17.50`, LSTM `0.1409`, delta `-0.2465`
- tick `140042`, seconds `12.50`, LSTM `0.3801`, delta `+0.0707`
- tick `140394`, seconds `18.00`, LSTM `0.0829`, delta `-0.0580`
- tick `139274`, seconds `0.50`, LSTM `0.2996`, delta `-0.0535`
- tick `139370`, seconds `2.00`, LSTM `0.2366`, delta `-0.0475`
- tick `140138`, seconds `14.00`, LSTM `0.3253`, delta `-0.0473`
- tick `140170`, seconds `14.50`, LSTM `0.3693`, delta `+0.0440`
- tick `139402`, seconds `2.50`, LSTM `0.1929`, delta `-0.0438`
- tick `140074`, seconds `13.00`, LSTM `0.3441`, delta `-0.0360`
- tick `139690`, seconds `7.00`, LSTM `0.2209`, delta `+0.0348`

## Top 15 local ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.002633`, |coef| `0.002633`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002044`, |coef| `0.002044`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001334`, |coef| `0.001334`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001174`, |coef| `0.001174`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.001172`, |coef| `0.001172`
- `lag_11__T1__is_scoped`: coefficient `-0.001049`, |coef| `0.001049`
- `lag_15__CT_place_SIDEENTRANCE`: coefficient `-0.001032`, |coef| `0.001032`
- `lag_15__T_place_TSIDELOWER`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_03__CT_place_HOUSE`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_07__CT2__flash_duration`: coefficient `0.000910`, |coef| `0.000910`
- `lag_05__CT_place_UNKNOWN`: coefficient `-0.000905`, |coef| `0.000905`
- `lag_11__CT_place_SIDEENTRANCE`: coefficient `-0.000892`, |coef| `0.000892`
- `lag_15__T_place_RUINS`: coefficient `0.000818`, |coef| `0.000818`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_00__T_kills_last_3s`: coefficient `-0.000748`, |coef| `0.000748`

## Top 10 utility ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.002633` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.000910` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.000729` (lowers CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.000718` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `0.000677` (raises CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.000640` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000598` (lowers CT win probability)
- `lag_14__CT3__molly`: coefficient `0.000590` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000583` (raises CT win probability)
- `lag_11__CT_flashes_last_5s`: coefficient `0.000545` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.002044` (raises CT win probability)
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001334` (raises CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001174` (raises CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.001172` (lowers CT win probability)
- `lag_11__T1__is_scoped`: coefficient `-0.001049` (lowers CT win probability)
- `lag_15__CT_place_SIDEENTRANCE`: coefficient `-0.001032` (lowers CT win probability)
- `lag_15__T_place_TSIDELOWER`: coefficient `-0.000971` (lowers CT win probability)
- `lag_03__CT_place_HOUSE`: coefficient `-0.000912` (lowers CT win probability)
- `lag_05__CT_place_UNKNOWN`: coefficient `-0.000905` (lowers CT win probability)
- `lag_11__CT_place_SIDEENTRANCE`: coefficient `-0.000892` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `140362`, seconds `17.50`, LSTM delta `-0.2465`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.028946`
- `lag_03__CT_place_TSIDEUPPER`: contribution `-0.010028`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.008807`
- `lag_15__T_place_TSIDELOWER`: contribution `-0.007281`
- `lag_11__T1__is_scoped`: contribution `-0.005995`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.028946`
- `lag_07__CT2__flash_duration`: contribution `-0.004614`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002506`

### tick `140042`, seconds `12.50`, LSTM delta `+0.0707`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `+0.028946`
- `lag_14__T_place_WATER`: contribution `+0.003483`
- `lag_14__T_place_TUNNEL`: contribution `+0.003183`
- `lag_15__T_place_WATER`: contribution `+0.003070`
- `lag_12__T_place_WATER`: contribution `+0.002553`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `+0.028946`
- `lag_06__CT2__flash_duration`: contribution `+0.001865`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.001510`
- `lag_00__CT_active_infernos`: contribution `+0.001145`

### tick `140394`, seconds `18.00`, LSTM delta `-0.0580`

Top all feature movements:
- `lag_08__CT_flashes_last_5s`: contribution `-0.007893`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.004724`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.004156`
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.004071`
- `lag_12__T_place_TUNNEL`: contribution `+0.003488`

Top utility-only movements:
- `lag_08__CT_flashes_last_5s`: contribution `-0.007893`
- `lag_08__CT2__flash_duration`: contribution `-0.003246`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001510`

### tick `139274`, seconds `0.50`, LSTM delta `-0.0535`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.027193`
- `lag_01__T_place_TSPAWN`: contribution `-0.000641`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000565`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000526`
- `lag_01__centroid_distance_xy`: contribution `-0.000470`

Top utility-only movements:
- `lag_01__T3__molly`: contribution `-0.000402`
- `lag_01__T1__flash`: contribution `-0.000388`
- `lag_01__CT5__flash`: contribution `-0.000352`
- `lag_01__T1__utility_total`: contribution `-0.000311`
- `lag_01__CT4__smoke`: contribution `+0.000306`

### tick `139370`, seconds `2.00`, LSTM delta `-0.0475`

Top all feature movements:
- `lag_00__CT_place_UNKNOWN`: contribution `-0.028718`
- `lag_04__CT_place_UNKNOWN`: contribution `-0.010075`
- `lag_02__CT_place_UNKNOWN`: contribution `-0.007076`
- `lag_01__CT_place_UNKNOWN`: contribution `+0.005444`
- `lag_00__T_place_TUNNEL`: contribution `-0.002056`

Top utility-only movements:
- `lag_04__T3__molly`: contribution `-0.000303`
- `lag_04__T1__flash`: contribution `-0.000267`
- `lag_04__CT2__molly`: contribution `-0.000203`
