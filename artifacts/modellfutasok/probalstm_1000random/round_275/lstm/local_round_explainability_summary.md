# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `10`

## Largest probability jumps

- tick `78721`, seconds `15.50`, LSTM `0.7420`, delta `+0.1839`
- tick `81441`, seconds `58.00`, LSTM `0.5285`, delta `-0.1649`
- tick `84673`, seconds `108.50`, LSTM `0.2073`, delta `+0.1379`
- tick `81921`, seconds `65.50`, LSTM `0.0498`, delta `-0.1037`
- tick `81761`, seconds `63.00`, LSTM `0.1626`, delta `-0.0942`
- tick `81505`, seconds `59.00`, LSTM `0.4153`, delta `-0.0881`
- tick `81569`, seconds `60.00`, LSTM `0.2692`, delta `-0.0784`
- tick `81537`, seconds `59.50`, LSTM `0.3477`, delta `-0.0677`
- tick `82497`, seconds `74.50`, LSTM `0.0300`, delta `-0.0597`
- tick `79841`, seconds `33.00`, LSTM `0.7686`, delta `-0.0524`

## Top 15 local ridge features

- `lag_12__T_flashes_last_5s`: coefficient `0.002500`, |coef| `0.002500`
- `lag_03__T_place_PLAYGROUND`: coefficient `0.002100`, |coef| `0.002100`
- `lag_00__CT_place_UPPERPARK`: coefficient `0.001909`, |coef| `0.001909`
- `lag_07__CT_place_LOBBY`: coefficient `-0.001868`, |coef| `0.001868`
- `lag_00__kill_diff_last_3s`: coefficient `0.001792`, |coef| `0.001792`
- `lag_04__CT_place_UPPERPARK`: coefficient `0.001727`, |coef| `0.001727`
- `lag_00__CT_place_PIPE`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_13__T_place_CONNECTOR`: coefficient `-0.001635`, |coef| `0.001635`
- `lag_01__CT_place_UPPERPARK`: coefficient `0.001617`, |coef| `0.001617`
- `lag_03__CT_place_UPPERPARK`: coefficient `0.001572`, |coef| `0.001572`
- `lag_12__T_place_LOWERPARK`: coefficient `-0.001559`, |coef| `0.001559`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001542`, |coef| `0.001542`
- `lag_06__T_flashes_last_5s`: coefficient `0.001510`, |coef| `0.001510`
- `lag_00__CT4__flash_duration`: coefficient `-0.001462`, |coef| `0.001462`
- `lag_02__T_flashes_last_5s`: coefficient `-0.001404`, |coef| `0.001404`

## Top 10 utility ridge features

- `lag_12__T_flashes_last_5s`: coefficient `0.002500` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `0.001510` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.001462` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.001404` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001191` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.001133` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000949` (lowers CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.000945` (raises CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `-0.000898` (lowers CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.000883` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_PLAYGROUND`: coefficient `0.002100` (raises CT win probability)
- `lag_00__CT_place_UPPERPARK`: coefficient `0.001909` (raises CT win probability)
- `lag_07__CT_place_LOBBY`: coefficient `-0.001868` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001792` (raises CT win probability)
- `lag_04__CT_place_UPPERPARK`: coefficient `0.001727` (raises CT win probability)
- `lag_00__CT_place_PIPE`: coefficient `-0.001711` (lowers CT win probability)
- `lag_13__T_place_CONNECTOR`: coefficient `-0.001635` (lowers CT win probability)
- `lag_01__CT_place_UPPERPARK`: coefficient `0.001617` (raises CT win probability)
- `lag_03__CT_place_UPPERPARK`: coefficient `0.001572` (raises CT win probability)
- `lag_12__T_place_LOWERPARK`: coefficient `-0.001559` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78721`, seconds `15.50`, LSTM delta `+0.1839`

Top all feature movements:
- `lag_03__T_place_PLAYGROUND`: contribution `+0.030843`
- `lag_04__CT_place_FOUNTAIN`: contribution `+0.010268`
- `lag_10__CT_place_UPPERPARK`: contribution `+0.007585`
- `lag_01__CT_place_WATER`: contribution `+0.005768`
- `lag_00__T_place_FOUNTAIN`: contribution `+0.005699`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.004984`

### tick `81441`, seconds `58.00`, LSTM delta `-0.1649`

Top all feature movements:
- `lag_00__CT_place_UPPERPARK`: contribution `-0.013591`
- `lag_00__CT4__flash_duration`: contribution `-0.008023`
- `lag_13__T_place_CONNECTOR`: contribution `-0.007916`
- `lag_10__T_place_TSTAIRS`: contribution `-0.006280`
- `lag_06__T1__is_scoped`: contribution `-0.006270`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.008023`

### tick `84673`, seconds `108.50`, LSTM delta `+0.1379`

Top all feature movements:
- `lag_12__T_flashes_last_5s`: contribution `+0.022651`
- `lag_07__CT_place_LOBBY`: contribution `+0.015290`
- `lag_02__T_flashes_last_5s`: contribution `+0.012717`
- `lag_00__T_bomb_zone_count`: contribution `+0.008976`
- `lag_05__T_bomb_zone_count`: contribution `+0.007301`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `+0.022651`
- `lag_02__T_flashes_last_5s`: contribution `+0.012717`

### tick `81921`, seconds `65.50`, LSTM delta `-0.1037`

Top all feature movements:
- `lag_05__CT_place_PIPE`: contribution `-0.052144`
- `lag_00__kill_diff_last_3s`: contribution `-0.004314`
- `lag_14__T2__duck_amount`: contribution `-0.004170`
- `lag_00__CT_place_BRIDGE`: contribution `-0.003778`
- `lag_08__T_place_FOUNTAIN`: contribution `-0.003541`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.002410`

### tick `81761`, seconds `63.00`, LSTM delta `-0.0942`

Top all feature movements:
- `lag_00__CT_place_PIPE`: contribution `-0.078349`
- `lag_10__CT_place_UPPERPARK`: contribution `-0.007585`
- `lag_06__T1__is_scoped`: contribution `-0.006270`
- `lag_04__T_place_LOWERPARK`: contribution `-0.004409`
- `lag_03__T_place_FOUNTAIN`: contribution `+0.004096`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.003560`
- `lag_10__CT4__flash_duration`: contribution `-0.002909`
