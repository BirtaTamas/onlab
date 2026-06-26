# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `37681`, seconds `108.50`, LSTM `0.1760`, delta `-0.3324`
- tick `37553`, seconds `106.50`, LSTM `0.6562`, delta `+0.1429`
- tick `37585`, seconds `107.00`, LSTM `0.5144`, delta `-0.1418`
- tick `34161`, seconds `53.50`, LSTM `0.4052`, delta `-0.0951`
- tick `37201`, seconds `101.00`, LSTM `0.5205`, delta `-0.0827`
- tick `37137`, seconds `100.00`, LSTM `0.6121`, delta `+0.0748`
- tick `33553`, seconds `44.00`, LSTM `0.5792`, delta `+0.0689`
- tick `37713`, seconds `109.00`, LSTM `0.1158`, delta `-0.0602`
- tick `33329`, seconds `40.50`, LSTM `0.4488`, delta `-0.0597`
- tick `37361`, seconds `103.50`, LSTM `0.4559`, delta `-0.0591`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001805`, |coef| `0.001805`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001776`, |coef| `0.001776`
- `lag_04__T_place_BALCONY`: coefficient `0.001756`, |coef| `0.001756`
- `lag_01__T_bomb_zone_count`: coefficient `-0.001643`, |coef| `0.001643`
- `lag_00__T_kills_last_3s`: coefficient `-0.001519`, |coef| `0.001519`
- `lag_15__CT5__flash_duration`: coefficient `0.001482`, |coef| `0.001482`
- `lag_00__CT1__flash`: coefficient `0.001457`, |coef| `0.001457`
- `lag_00__CT5__is_scoped`: coefficient `0.001453`, |coef| `0.001453`
- `lag_10__CT_place_RUINS`: coefficient `0.001441`, |coef| `0.001441`
- `lag_03__CT_place_PIT`: coefficient `0.001392`, |coef| `0.001392`
- `lag_11__CT3__flash_duration`: coefficient `0.001385`, |coef| `0.001385`
- `lag_04__CT3__shots_fired`: coefficient `-0.001360`, |coef| `0.001360`
- `lag_00__CT_place_LIBRARY`: coefficient `0.001359`, |coef| `0.001359`
- `lag_06__T_place_BALCONY`: coefficient `0.001315`, |coef| `0.001315`
- `lag_15__CT_flashed_players`: coefficient `0.001292`, |coef| `0.001292`

## Top 10 utility ridge features

- `lag_15__CT5__flash_duration`: coefficient `0.001482` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001457` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.001385` (raises CT win probability)
- `lag_15__CT5__flash`: coefficient `0.001286` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.001104` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001091` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.001090` (raises CT win probability)
- `lag_06__T3__flash`: coefficient `-0.001002` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000928` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000889` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001805` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.001776` (raises CT win probability)
- `lag_04__T_place_BALCONY`: coefficient `0.001756` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.001643` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001519` (lowers CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `0.001453` (raises CT win probability)
- `lag_10__CT_place_RUINS`: coefficient `0.001441` (raises CT win probability)
- `lag_03__CT_place_PIT`: coefficient `0.001392` (raises CT win probability)
- `lag_04__CT3__shots_fired`: coefficient `-0.001360` (lowers CT win probability)
- `lag_00__CT_place_LIBRARY`: coefficient `0.001359` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `37681`, seconds `108.50`, LSTM delta `-0.3324`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `-0.024154`
- `lag_10__T_place_BALCONY`: contribution `-0.016552`
- `lag_01__T_bomb_zone_count`: contribution `-0.009565`
- `lag_00__CT_place_LIBRARY`: contribution `-0.008711`
- `lag_15__CT5__flash_duration`: contribution `-0.007776`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.007776`
- `lag_11__CT3__flash_duration`: contribution `-0.005685`
- `lag_00__CT1__flash`: contribution `-0.005213`
- `lag_15__CT5__flash`: contribution `-0.004564`
- `lag_12__T_B_site_active_infernos`: contribution `-0.003121`

### tick `37553`, seconds `106.50`, LSTM delta `+0.1429`

Top all feature movements:
- `lag_06__T_place_BALCONY`: contribution `+0.018084`
- `lag_00__T_place_BALCONY`: contribution `+0.015605`
- `lag_10__CT_place_RUINS`: contribution `+0.005036`
- `lag_00__kill_diff_last_3s`: contribution `+0.004345`
- `lag_02__CT1__duck_amount`: contribution `+0.003866`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.003090`
- `lag_15__CT5__flash_duration`: contribution `-0.002761`
- `lag_15__CT3__flash_duration`: contribution `+0.002672`
- `lag_11__CT5__flash`: contribution `+0.001719`
- `lag_07__CT3__flash_duration`: contribution `+0.001700`

### tick `37585`, seconds `107.00`, LSTM delta `-0.1418`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `-0.013187`
- `lag_07__T_place_BALCONY`: contribution `-0.006704`
- `lag_03__CT_shots_fired_sum`: contribution `-0.004936`
- `lag_00__T_kills_last_3s`: contribution `-0.004812`
- `lag_13__CT_place_LIBRARY`: contribution `-0.004650`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `-0.001705`
- `lag_09__T_B_site_active_infernos`: contribution `-0.001423`
- `lag_12__CT5__flash_duration`: contribution `-0.001406`

### tick `34161`, seconds `53.50`, LSTM delta `-0.0951`

Top all feature movements:
- `lag_00__CT5__is_scoped`: contribution `-0.005195`
- `lag_03__CT_place_TOPOFMID`: contribution `-0.003330`
- `lag_00__CT5__is_walking`: contribution `-0.002801`
- `lag_00__T3__is_walking`: contribution `-0.002776`
- `lag_12__CT3__is_walking`: contribution `-0.002751`

Top utility-only movements:
- `lag_13__CT_A_site_active_infernos`: contribution `-0.002573`

### tick `37201`, seconds `101.00`, LSTM delta `-0.0827`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004812`
- `lag_00__kill_diff_last_3s`: contribution `-0.004345`
- `lag_00__CT5__flash_duration`: contribution `-0.003626`
- `lag_12__CT3__is_walking`: contribution `-0.002751`
- `lag_07__CT5__flash_duration`: contribution `-0.002728`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.003626`
- `lag_07__CT5__flash_duration`: contribution `-0.002728`
- `lag_14__CT4__flash_duration`: contribution `-0.002525`
- `lag_00__CT5__flash`: contribution `-0.002085`
- `lag_07__CT_flash_duration_sum`: contribution `-0.001320`
