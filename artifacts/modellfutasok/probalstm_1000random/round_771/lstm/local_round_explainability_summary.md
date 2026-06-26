# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m1-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `32032`, seconds `25.00`, LSTM `0.8054`, delta `+0.5154`
- tick `32352`, seconds `30.00`, LSTM `0.4050`, delta `-0.3365`
- tick `32160`, seconds `27.00`, LSTM `0.8825`, delta `+0.1719`
- tick `31872`, seconds `22.50`, LSTM `0.4307`, delta `-0.1676`
- tick `31904`, seconds `23.00`, LSTM `0.2975`, delta `-0.1332`
- tick `32384`, seconds `30.50`, LSTM `0.2890`, delta `-0.1160`
- tick `32256`, seconds `28.50`, LSTM `0.7487`, delta `-0.1134`
- tick `32512`, seconds `32.50`, LSTM `0.2165`, delta `-0.1061`
- tick `32416`, seconds `31.00`, LSTM `0.3898`, delta `+0.1008`
- tick `32064`, seconds `25.50`, LSTM `0.7128`, delta `-0.0926`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.004134`, |coef| `0.004134`
- `lag_05__CT5__flash_duration`: coefficient `-0.002898`, |coef| `0.002898`
- `lag_00__kill_diff_last_3s`: coefficient `0.002854`, |coef| `0.002854`
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.002703`, |coef| `0.002703`
- `lag_00__CT_kills_last_3s`: coefficient `0.002691`, |coef| `0.002691`
- `lag_00__damage_diff_last_5s`: coefficient `0.002386`, |coef| `0.002386`
- `lag_10__CT5__flash_duration`: coefficient `0.002367`, |coef| `0.002367`
- `lag_14__T5__flash_duration`: coefficient `0.002287`, |coef| `0.002287`
- `lag_04__CT_place_CONNECTOR`: coefficient `0.002271`, |coef| `0.002271`
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.002201`, |coef| `0.002201`
- `lag_14__CT3__flash_duration`: coefficient `0.002140`, |coef| `0.002140`
- `lag_14__CT_flashed_players`: coefficient `0.002120`, |coef| `0.002120`
- `lag_04__CT_place_SHOP`: coefficient `0.001904`, |coef| `0.001904`
- `lag_10__CT_place_TRUCK`: coefficient `-0.001769`, |coef| `0.001769`
- `lag_00__CT_damage_last_5s`: coefficient `0.001700`, |coef| `0.001700`

## Top 10 utility ridge features

- `lag_05__CT5__flash_duration`: coefficient `-0.002898` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.002703` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.002367` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.002287` (raises CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.002201` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.002140` (raises CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.001655` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.001544` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.001534` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.001529` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.004134` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002854` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002691` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002386` (raises CT win probability)
- `lag_04__CT_place_CONNECTOR`: coefficient `0.002271` (raises CT win probability)
- `lag_14__CT_flashed_players`: coefficient `0.002120` (raises CT win probability)
- `lag_04__CT_place_SHOP`: coefficient `0.001904` (raises CT win probability)
- `lag_10__CT_place_TRUCK`: coefficient `-0.001769` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001700` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.001646` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `32032`, seconds `25.00`, LSTM delta `+0.5154`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.048830`
- `lag_05__CT5__flash_duration`: contribution `+0.019323`
- `lag_10__CT5__flash_duration`: contribution `+0.015782`
- `lag_00__CT_kills_last_3s`: contribution `+0.015540`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.014281`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `+0.019323`
- `lag_10__CT5__flash_duration`: contribution `+0.015782`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.014281`
- `lag_14__T5__flash_duration`: contribution `+0.014097`
- `lag_15__T4__flash_duration`: contribution `+0.009836`

### tick `32352`, seconds `30.00`, LSTM delta `-0.3365`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `-0.020795`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.015174`
- `lag_00__kill_diff_last_3s`: contribution `-0.013739`
- `lag_00__damage_diff_last_5s`: contribution `-0.012326`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.010137`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.015174`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.010137`
- `lag_14__CT3__flash_duration`: contribution `-0.008885`
- `lag_15__CT5__flash_duration`: contribution `-0.007507`
- `lag_15__T4__flash_duration`: contribution `+0.005071`

### tick `32160`, seconds `27.00`, LSTM delta `+0.1719`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.022979`
- `lag_04__CT_kills_last_3s`: contribution `+0.008725`
- `lag_03__CT_shots_fired_sum`: contribution `+0.008266`
- `lag_00__CT_kills_last_3s`: contribution `+0.007770`
- `lag_00__kill_diff_last_3s`: contribution `+0.006869`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.004638`
- `lag_14__CT5__flash_duration`: contribution `+0.003820`
- `lag_15__T4__flash_duration`: contribution `+0.003246`
- `lag_12__utility_damage_diff_last_5s`: contribution `+0.003233`
- `lag_14__CT_flash_duration_sum`: contribution `+0.003063`

### tick `31872`, seconds `22.50`, LSTM delta `-0.1676`

Top all feature movements:
- `lag_05__CT5__flash_duration`: contribution `-0.019323`
- `lag_04__CT_place_CONNECTOR`: contribution `-0.008123`
- `lag_00__kill_diff_last_3s`: contribution `-0.006869`
- `lag_14__CT_place_TRUCK`: contribution `-0.006344`
- `lag_00__damage_diff_last_5s`: contribution `-0.005382`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.019323`
- `lag_00__CT5__flash_duration`: contribution `-0.005303`
- `lag_09__T5__flash_duration`: contribution `-0.004102`
- `lag_09__T4__flash_duration`: contribution `-0.003300`
- `lag_05__CT_flash_duration_sum`: contribution `-0.002810`

### tick `31904`, seconds `23.00`, LSTM delta `-0.1332`

Top all feature movements:
- `lag_01__CT5__flash_duration`: contribution `-0.007483`
- `lag_06__CT5__flash_duration`: contribution `-0.004643`
- `lag_06__CT_place_TRUCK`: contribution `-0.004196`
- `lag_15__CT_place_TRUCK`: contribution `-0.004159`
- `lag_10__T5__flash_duration`: contribution `-0.004111`

Top utility-only movements:
- `lag_01__CT5__flash_duration`: contribution `-0.007483`
- `lag_06__CT5__flash_duration`: contribution `-0.004643`
- `lag_10__T5__flash_duration`: contribution `-0.004111`
- `lag_09__CT4__flash_duration`: contribution `-0.003297`
- `lag_01__CT_flash_duration_sum`: contribution `-0.002938`
