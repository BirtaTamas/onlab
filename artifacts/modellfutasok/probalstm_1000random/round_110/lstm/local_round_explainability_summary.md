# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m1-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `91806`, seconds `46.00`, LSTM `0.1903`, delta `-0.2810`
- tick `91838`, seconds `46.50`, LSTM `0.0857`, delta `-0.1046`
- tick `91742`, seconds `45.00`, LSTM `0.4443`, delta `+0.0481`
- tick `89854`, seconds `15.50`, LSTM `0.4294`, delta `+0.0440`
- tick `91678`, seconds `44.00`, LSTM `0.3728`, delta `-0.0427`
- tick `91518`, seconds `41.50`, LSTM `0.4307`, delta `-0.0413`
- tick `91870`, seconds `47.00`, LSTM `0.0523`, delta `-0.0334`
- tick `91230`, seconds `37.00`, LSTM `0.4562`, delta `-0.0283`
- tick `91774`, seconds `45.50`, LSTM `0.4713`, delta `+0.0270`
- tick `90046`, seconds `18.50`, LSTM `0.4641`, delta `+0.0269`

## Top 15 local ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.002922`, |coef| `0.002922`
- `lag_08__CT_place_STAIRS`: coefficient `0.002563`, |coef| `0.002563`
- `lag_00__CT_place_SHOP`: coefficient `-0.002008`, |coef| `0.002008`
- `lag_05__CT_flashed_players`: coefficient `-0.001968`, |coef| `0.001968`
- `lag_12__CT3__duck_amount`: coefficient `0.001725`, |coef| `0.001725`
- `lag_05__T2__duck_amount`: coefficient `0.001677`, |coef| `0.001677`
- `lag_15__CT_place_TRUCK`: coefficient `0.001665`, |coef| `0.001665`
- `lag_13__CT_place_TRUCK`: coefficient `-0.001605`, |coef| `0.001605`
- `lag_12__CT_place_STAIRS`: coefficient `0.001550`, |coef| `0.001550`
- `lag_00__CT3__alive`: coefficient `0.001542`, |coef| `0.001542`
- `lag_00__CT3__hp`: coefficient `0.001520`, |coef| `0.001520`
- `lag_00__T_kills_last_3s`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_09__T_place_APARTMENTS`: coefficient `-0.001492`, |coef| `0.001492`
- `lag_09__T_B_site_active_infernos`: coefficient `-0.001442`, |coef| `0.001442`
- `lag_03__CT_place_STAIRS`: coefficient `0.001406`, |coef| `0.001406`

## Top 10 utility ridge features

- `lag_09__T_B_site_active_infernos`: coefficient `-0.001442` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.001309` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.001254` (lowers CT win probability)
- `lag_10__CT3__smoke`: coefficient `0.001188` (raises CT win probability)
- `lag_12__T5__smoke`: coefficient `0.001179` (raises CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.001173` (raises CT win probability)
- `lag_13__T4__molly`: coefficient `0.001169` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.001062` (raises CT win probability)
- `lag_09__T_active_infernos`: coefficient `-0.001039` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000825` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.002922` (raises CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `0.002563` (raises CT win probability)
- `lag_00__CT_place_SHOP`: coefficient `-0.002008` (lowers CT win probability)
- `lag_05__CT_flashed_players`: coefficient `-0.001968` (lowers CT win probability)
- `lag_12__CT3__duck_amount`: coefficient `0.001725` (raises CT win probability)
- `lag_05__T2__duck_amount`: coefficient `0.001677` (raises CT win probability)
- `lag_15__CT_place_TRUCK`: coefficient `0.001665` (raises CT win probability)
- `lag_13__CT_place_TRUCK`: coefficient `-0.001605` (lowers CT win probability)
- `lag_12__CT_place_STAIRS`: coefficient `0.001550` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001542` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `91806`, seconds `46.00`, LSTM delta `-0.2810`

Top all feature movements:
- `lag_08__CT_place_STAIRS`: contribution `-0.019948`
- `lag_00__CT_place_TRUCK`: contribution `-0.018847`
- `lag_15__CT_place_TRUCK`: contribution `-0.010743`
- `lag_13__CT_place_TRUCK`: contribution `-0.010354`
- `lag_00__CT_place_SHOP`: contribution `-0.010070`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `-0.004077`
- `lag_05__CT3__flash_duration`: contribution `-0.003545`
- `lag_05__T2__flash_duration`: contribution `-0.003516`

### tick `91838`, seconds `46.50`, LSTM delta `-0.1046`

Top all feature movements:
- `lag_00__T_place_TRUCK`: contribution `-0.012390`
- `lag_01__CT_place_TRUCK`: contribution `-0.009050`
- `lag_09__CT_place_STAIRS`: contribution `-0.008179`
- `lag_00__CT_place_JUNGLE`: contribution `-0.007571`
- `lag_01__CT_place_SHOP`: contribution `-0.005807`

Top utility-only movements:
- `lag_10__T_B_site_active_infernos`: contribution `-0.002216`

### tick `91742`, seconds `45.00`, LSTM delta `+0.0481`

Top all feature movements:
- `lag_15__CT_place_TRUCK`: contribution `+0.010743`
- `lag_13__CT_place_TRUCK`: contribution `+0.010354`
- `lag_03__CT_flashed_players`: contribution `+0.003285`
- `lag_06__CT_place_STAIRS`: contribution `+0.002769`
- `lag_08__CT2__duck_amount`: contribution `-0.002641`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89854`, seconds `15.50`, LSTM delta `+0.0440`

Top all feature movements:
- `lag_12__CT_place_STAIRS`: contribution `+0.012067`
- `lag_14__T_place_HOUSE`: contribution `+0.004198`
- `lag_15__T_place_HOUSE`: contribution `+0.004055`
- `lag_14__T_place_BACKALLEY`: contribution `-0.003662`
- `lag_04__CT_place_TRUCK`: contribution `+0.003279`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `91678`, seconds `44.00`, LSTM delta `-0.0427`

Top all feature movements:
- `lag_13__CT_place_TRUCK`: contribution `-0.010354`
- `lag_04__CT_place_STAIRS`: contribution `-0.009768`
- `lag_12__CT3__duck_amount`: contribution `-0.006417`
- `lag_09__CT_place_TRUCK`: contribution `-0.006407`
- `lag_13__CT4__duck_amount`: contribution `+0.003733`

Top utility-only movements:
- No utility movement among the top local contributors.
