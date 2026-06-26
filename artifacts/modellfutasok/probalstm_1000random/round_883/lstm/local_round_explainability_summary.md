# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `103720`, seconds `57.00`, LSTM `0.2536`, delta `-0.2507`
- tick `103464`, seconds `53.00`, LSTM `0.2592`, delta `-0.2396`
- tick `104744`, seconds `73.00`, LSTM `0.0428`, delta `-0.1819`
- tick `103624`, seconds `55.50`, LSTM `0.4822`, delta `+0.1812`
- tick `103368`, seconds `51.50`, LSTM `0.5261`, delta `-0.1354`
- tick `104264`, seconds `65.50`, LSTM `0.2084`, delta `-0.0913`
- tick `103592`, seconds `55.00`, LSTM `0.3010`, delta `+0.0790`
- tick `103240`, seconds `49.50`, LSTM `0.6699`, delta `+0.0561`
- tick `101576`, seconds `23.50`, LSTM `0.7003`, delta `+0.0548`
- tick `103848`, seconds `59.00`, LSTM `0.1031`, delta `-0.0474`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003021`, |coef| `0.003021`
- `lag_00__kill_diff_last_3s`: coefficient `0.002938`, |coef| `0.002938`
- `lag_08__T_bomb_zone_count`: coefficient `0.001926`, |coef| `0.001926`
- `lag_00__damage_diff_last_5s`: coefficient `0.001854`, |coef| `0.001854`
- `lag_00__T_damage_last_5s`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_08__CT5__is_walking`: coefficient `-0.001511`, |coef| `0.001511`
- `lag_12__T1__duck_amount`: coefficient `0.001462`, |coef| `0.001462`
- `lag_00__T2__flash_duration`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_02__T3__flash_duration`: coefficient `-0.001403`, |coef| `0.001403`
- `lag_06__T1__duck_amount`: coefficient `0.001400`, |coef| `0.001400`
- `lag_01__kill_diff_last_3s`: coefficient `0.001374`, |coef| `0.001374`
- `lag_12__T2__duck_amount`: coefficient `0.001308`, |coef| `0.001308`
- `lag_10__T3__flash_duration`: coefficient `-0.001295`, |coef| `0.001295`
- `lag_02__T2__flash_duration`: coefficient `-0.001292`, |coef| `0.001292`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001278`, |coef| `0.001278`

## Top 10 utility ridge features

- `lag_00__T2__flash_duration`: coefficient `-0.001458` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001403` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001295` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `-0.001292` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001263` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.001149` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001104` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `-0.001035` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.000994` (lowers CT win probability)
- `lag_08__CT5__flash`: coefficient `-0.000981` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003021` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002938` (raises CT win probability)
- `lag_08__T_bomb_zone_count`: coefficient `0.001926` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001854` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001779` (lowers CT win probability)
- `lag_08__CT5__is_walking`: coefficient `-0.001511` (lowers CT win probability)
- `lag_12__T1__duck_amount`: coefficient `0.001462` (raises CT win probability)
- `lag_06__T1__duck_amount`: coefficient `0.001400` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.001374` (raises CT win probability)
- `lag_12__T2__duck_amount`: coefficient `0.001308` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `103720`, seconds `57.00`, LSTM delta `-0.2507`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009570`
- `lag_10__T3__flash_duration`: contribution `-0.008370`
- `lag_10__T2__flash_duration`: contribution `-0.007912`
- `lag_00__kill_diff_last_3s`: contribution `-0.007071`
- `lag_04__CT_place_BALCONY`: contribution `-0.006275`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.008370`
- `lag_10__T2__flash_duration`: contribution `-0.007912`
- `lag_10__T_flash_duration_sum`: contribution `-0.005477`
- `lag_03__T2__flash_duration`: contribution `-0.005094`
- `lag_00__CT1__flash`: contribution `-0.003951`

### tick `103464`, seconds `53.00`, LSTM delta `-0.2396`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009570`
- `lag_02__T3__flash_duration`: contribution `-0.009069`
- `lag_02__T2__flash_duration`: contribution `-0.008897`
- `lag_10__T_flashed_players`: contribution `-0.008684`
- `lag_00__kill_diff_last_3s`: contribution `-0.007071`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.009069`
- `lag_02__T2__flash_duration`: contribution `-0.008897`
- `lag_02__T_flash_duration_sum`: contribution `-0.005703`

### tick `104744`, seconds `73.00`, LSTM delta `-0.1819`

Top all feature movements:
- `lag_08__T_bomb_zone_count`: contribution `-0.011209`
- `lag_00__T_kills_last_3s`: contribution `-0.009570`
- `lag_00__kill_diff_last_3s`: contribution `-0.007071`
- `lag_12__T1__duck_amount`: contribution `-0.005132`
- `lag_06__CT5__duck_amount`: contribution `-0.004461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103624`, seconds `55.50`, LSTM delta `+0.1812`

Top all feature movements:
- `lag_00__T2__flash_duration`: contribution `+0.010037`
- `lag_07__T2__flash_duration`: contribution `+0.008694`
- `lag_15__T_flashed_players`: contribution `+0.008455`
- `lag_00__kill_diff_last_3s`: contribution `+0.007071`
- `lag_01__CT_place_BALCONY`: contribution `+0.006009`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.010037`
- `lag_07__T2__flash_duration`: contribution `+0.008694`
- `lag_07__T3__flash_duration`: contribution `+0.004706`
- `lag_07__T_flash_duration_sum`: contribution `+0.003446`

### tick `103368`, seconds `51.50`, LSTM delta `-0.1354`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009570`
- `lag_00__kill_diff_last_3s`: contribution `-0.007071`
- `lag_07__T_flashed_players`: contribution `-0.006282`
- `lag_06__T_flashed_players`: contribution `-0.004928`
- `lag_00__T_damage_last_5s`: contribution `-0.003327`

Top utility-only movements:
- No utility movement among the top local contributors.
