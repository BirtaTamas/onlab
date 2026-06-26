# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `86416`, seconds `38.50`, LSTM `0.2895`, delta `-0.2793`
- tick `86864`, seconds `45.50`, LSTM `0.7624`, delta `+0.2392`
- tick `86704`, seconds `43.00`, LSTM `0.2620`, delta `+0.1797`
- tick `85360`, seconds `22.00`, LSTM `0.4287`, delta `-0.1733`
- tick `85072`, seconds `17.50`, LSTM `0.6352`, delta `-0.1654`
- tick `86448`, seconds `39.00`, LSTM `0.1614`, delta `-0.1281`
- tick `84976`, seconds `16.00`, LSTM `0.7168`, delta `+0.0999`
- tick `85936`, seconds `31.00`, LSTM `0.6467`, delta `+0.0902`
- tick `86736`, seconds `43.50`, LSTM `0.3515`, delta `+0.0895`
- tick `85008`, seconds `16.50`, LSTM `0.7969`, delta `+0.0802`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003463`, |coef| `0.003463`
- `lag_00__kill_diff_last_3s`: coefficient `0.003224`, |coef| `0.003224`
- `lag_02__T_flashes_last_5s`: coefficient `0.002711`, |coef| `0.002711`
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `-0.002323`, |coef| `0.002323`
- `lag_00__T_kills_last_3s`: coefficient `-0.002233`, |coef| `0.002233`
- `lag_01__damage_diff_last_5s`: coefficient `0.002037`, |coef| `0.002037`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001999`, |coef| `0.001999`
- `lag_00__T_damage_last_5s`: coefficient `-0.001963`, |coef| `0.001963`
- `lag_10__CT3__is_scoped`: coefficient `-0.001906`, |coef| `0.001906`
- `lag_09__CT3__is_scoped`: coefficient `-0.001904`, |coef| `0.001904`
- `lag_06__CT_place_ALLEY`: coefficient `-0.001852`, |coef| `0.001852`
- `lag_01__CT3__is_scoped`: coefficient `0.001850`, |coef| `0.001850`
- `lag_05__damage_diff_last_5s`: coefficient `0.001847`, |coef| `0.001847`
- `lag_11__CT3__is_scoped`: coefficient `-0.001844`, |coef| `0.001844`
- `lag_00__CT_kills_last_3s`: coefficient `0.001832`, |coef| `0.001832`

## Top 10 utility ridge features

- `lag_02__T_flashes_last_5s`: coefficient `0.002711` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001725` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `0.001420` (raises CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.001416` (lowers CT win probability)
- `lag_15__T3__flash`: coefficient `0.001365` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `0.001354` (raises CT win probability)
- `lag_01__CT3__flash`: coefficient `0.001313` (raises CT win probability)
- `lag_09__CT3__flash`: coefficient `-0.001278` (lowers CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `0.001211` (raises CT win probability)
- `lag_15__T3__utility_total`: coefficient `0.001129` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003463` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003224` (raises CT win probability)
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `-0.002323` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002233` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002037` (raises CT win probability)
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001999` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001963` (lowers CT win probability)
- `lag_10__CT3__is_scoped`: coefficient `-0.001906` (lowers CT win probability)
- `lag_09__CT3__is_scoped`: coefficient `-0.001904` (lowers CT win probability)
- `lag_06__CT_place_ALLEY`: coefficient `-0.001852` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `86416`, seconds `38.50`, LSTM delta `-0.2793`

Top all feature movements:
- `lag_12__CT_place_SIDEENTRANCE`: contribution `-0.009350`
- `lag_01__CT3__is_scoped`: contribution `-0.008412`
- `lag_08__CT3__is_scoped`: contribution `-0.008160`
- `lag_00__kill_diff_last_3s`: contribution `-0.007760`
- `lag_00__T_kills_last_3s`: contribution `-0.007074`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.006366`
- `lag_15__T3__flash`: contribution `-0.004025`

### tick `86864`, seconds `45.50`, LSTM delta `+0.2392`

Top all feature movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.024559`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.015023`
- `lag_00__damage_diff_last_5s`: contribution `+0.007813`
- `lag_00__kill_diff_last_3s`: contribution `+0.007760`
- `lag_00__CT_kills_last_3s`: contribution `+0.005290`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.024559`
- `lag_14__CT3__flash`: contribution `+0.003961`

### tick `86704`, seconds `43.00`, LSTM delta `+0.1797`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.009063`
- `lag_10__CT3__is_scoped`: contribution `+0.008669`
- `lag_00__kill_diff_last_3s`: contribution `+0.007760`
- `lag_00__CT_kills_last_3s`: contribution `+0.005290`
- `lag_09__CT3__flash`: contribution `+0.004719`

Top utility-only movements:
- `lag_09__CT3__flash`: contribution `+0.004719`

### tick `85360`, seconds `22.00`, LSTM delta `-0.1733`

Top all feature movements:
- `lag_11__CT3__is_scoped`: contribution `-0.008389`
- `lag_00__kill_diff_last_3s`: contribution `-0.007760`
- `lag_00__T_kills_last_3s`: contribution `-0.007074`
- `lag_00__damage_diff_last_5s`: contribution `-0.007031`
- `lag_05__T3__flash_duration`: contribution `-0.006748`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.006748`
- `lag_08__CT2__flash_duration`: contribution `-0.006099`

### tick `85072`, seconds `17.50`, LSTM delta `-0.1654`

Top all feature movements:
- `lag_12__CT_place_SIDEENTRANCE`: contribution `-0.009350`
- `lag_00__damage_diff_last_5s`: contribution `-0.009063`
- `lag_00__kill_diff_last_3s`: contribution `-0.007760`
- `lag_00__T_kills_last_3s`: contribution `-0.007074`
- `lag_00__T_place_WATER`: contribution `-0.005683`

Top utility-only movements:
- `lag_12__T3__flash_duration`: contribution `-0.005049`
- `lag_12__CT1__flash_duration`: contribution `-0.003496`
- `lag_10__CT2__flash_duration`: contribution `-0.003293`
- `lag_01__T1__flash_duration`: contribution `-0.002914`
- `lag_02__CT1__flash_duration`: contribution `-0.002794`
