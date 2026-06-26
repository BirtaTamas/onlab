# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `114135`, seconds `24.00`, LSTM `0.6652`, delta `-0.1941`
- tick `113943`, seconds `21.00`, LSTM `0.6727`, delta `+0.1634`
- tick `116567`, seconds `62.00`, LSTM `0.8892`, delta `+0.1340`
- tick `114071`, seconds `23.00`, LSTM `0.8484`, delta `+0.1243`
- tick `114679`, seconds `32.50`, LSTM `0.6702`, delta `+0.1156`
- tick `114263`, seconds `26.00`, LSTM `0.5917`, delta `-0.1027`
- tick `114455`, seconds `29.00`, LSTM `0.5368`, delta `-0.0988`
- tick `113911`, seconds `20.50`, LSTM `0.5093`, delta `-0.0927`
- tick `114615`, seconds `31.50`, LSTM `0.5649`, delta `-0.0895`
- tick `116535`, seconds `61.50`, LSTM `0.7552`, delta `+0.0836`

## Top 15 local ridge features

- `lag_11__T_bomb_zone_count`: coefficient `0.003310`, |coef| `0.003310`
- `lag_00__CT_kills_last_3s`: coefficient `0.003160`, |coef| `0.003160`
- `lag_10__T_bomb_zone_count`: coefficient `0.002930`, |coef| `0.002930`
- `lag_00__kill_diff_last_3s`: coefficient `0.002806`, |coef| `0.002806`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002559`, |coef| `0.002559`
- `lag_15__T_place_CTSPAWN`: coefficient `-0.002149`, |coef| `0.002149`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002015`, |coef| `0.002015`
- `lag_00__T1__has_bomb`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001806`, |coef| `0.001806`
- `lag_14__T_place_CTSPAWN`: coefficient `-0.001732`, |coef| `0.001732`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001711`, |coef| `0.001711`
- `lag_12__T_bomb_zone_count`: coefficient `0.001644`, |coef| `0.001644`
- `lag_00__CT_place_STAIRS`: coefficient `0.001634`, |coef| `0.001634`
- `lag_03__T_bomb_zone_count`: coefficient `-0.001618`, |coef| `0.001618`
- `lag_00__T_macro_A`: coefficient `-0.001606`, |coef| `0.001606`

## Top 10 utility ridge features

- `lag_00__T1__smoke`: coefficient `-0.001505` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001375` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.001205` (lowers CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.001194` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.001169` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.000908` (raises CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.000809` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000771` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.000763` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000731` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_bomb_zone_count`: coefficient `0.003310` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003160` (raises CT win probability)
- `lag_10__T_bomb_zone_count`: coefficient `0.002930` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002806` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002559` (lowers CT win probability)
- `lag_15__T_place_CTSPAWN`: coefficient `-0.002149` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002015` (raises CT win probability)
- `lag_00__T1__has_bomb`: coefficient `-0.001816` (lowers CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `-0.001806` (lowers CT win probability)
- `lag_14__T_place_CTSPAWN`: coefficient `-0.001732` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `114135`, seconds `24.00`, LSTM delta `-0.1941`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013507`
- `lag_00__CT_kills_last_3s`: contribution `-0.009124`
- `lag_14__CT5__flash_duration`: contribution `-0.009001`
- `lag_14__CT_flash_duration_sum`: contribution `-0.008599`
- `lag_14__CT_flashed_players`: contribution `-0.004724`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `-0.009001`
- `lag_14__CT_flash_duration_sum`: contribution `-0.008599`
- `lag_00__CT5__flash_duration`: contribution `-0.003873`
- `lag_14__CT2__flash_duration`: contribution `-0.003843`
- `lag_14__T_flash_duration_sum`: contribution `-0.003364`

### tick `113943`, seconds `21.00`, LSTM delta `+0.1634`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009124`
- `lag_04__CT4__flash_duration`: contribution `+0.008561`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008398`
- `lag_00__kill_diff_last_3s`: contribution `+0.006754`
- `lag_00__CT4__duck_amount`: contribution `+0.004909`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.008561`
- `lag_08__CT5__flash_duration`: contribution `+0.004875`
- `lag_08__CT2__flash_duration`: contribution `+0.003763`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002992`
- `lag_08__T_flash_duration_sum`: contribution `+0.002471`

### tick `116567`, seconds `62.00`, LSTM delta `+0.1340`

Top all feature movements:
- `lag_11__T_bomb_zone_count`: contribution `+0.019268`
- `lag_06__T_bomb_zone_count`: contribution `+0.010512`
- `lag_00__CT_kills_last_3s`: contribution `+0.009124`
- `lag_00__kill_diff_last_3s`: contribution `+0.006754`
- `lag_04__CT1__duck_amount`: contribution `+0.005421`

Top utility-only movements:
- `lag_00__T1__smoke`: contribution `+0.003249`

### tick `114071`, seconds `23.00`, LSTM delta `+0.1243`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009124`
- `lag_00__kill_diff_last_3s`: contribution `+0.006754`
- `lag_03__CT_place_SHOP`: contribution `+0.004902`
- `lag_04__T5__duck_amount`: contribution `+0.003297`
- `lag_05__CT_shots_fired_sum`: contribution `-0.003288`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.002525`
- `lag_12__CT2__flash_duration`: contribution `+0.002507`

### tick `114679`, seconds `32.50`, LSTM delta `+0.1156`

Top all feature movements:
- `lag_03__T_bomb_zone_count`: contribution `+0.009419`
- `lag_00__CT_kills_last_3s`: contribution `+0.009124`
- `lag_14__CT4__flash_duration`: contribution `+0.008820`
- `lag_07__T_bomb_zone_count`: contribution `+0.008770`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008398`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.008820`
- `lag_14__CT_flash_duration_sum`: contribution `+0.004502`
