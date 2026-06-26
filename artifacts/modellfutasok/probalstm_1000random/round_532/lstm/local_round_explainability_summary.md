# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `29749`, seconds `30.00`, LSTM `0.1777`, delta `-0.2117`
- tick `30389`, seconds `40.00`, LSTM `0.0304`, delta `-0.0684`
- tick `28629`, seconds `12.50`, LSTM `0.3085`, delta `-0.0466`
- tick `29781`, seconds `30.50`, LSTM `0.1331`, delta `-0.0446`
- tick `28341`, seconds `8.00`, LSTM `0.3261`, delta `-0.0385`
- tick `29653`, seconds `28.50`, LSTM `0.3850`, delta `-0.0366`
- tick `30037`, seconds `34.50`, LSTM `0.1332`, delta `-0.0365`
- tick `28597`, seconds `12.00`, LSTM `0.3551`, delta `-0.0355`
- tick `28981`, seconds `18.00`, LSTM `0.3526`, delta `-0.0344`
- tick `29909`, seconds `32.50`, LSTM `0.1637`, delta `+0.0328`

## Top 15 local ridge features

- `lag_00__CT_place_JUNGLE`: coefficient `0.002574`, |coef| `0.002574`
- `lag_06__CT_place_UNDERPASS`: coefficient `-0.001684`, |coef| `0.001684`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_00__T_kills_last_3s`: coefficient `-0.001554`, |coef| `0.001554`
- `lag_10__CT_place_UNDERPASS`: coefficient `0.001435`, |coef| `0.001435`
- `lag_00__CT3__alive`: coefficient `0.001269`, |coef| `0.001269`
- `lag_00__CT3__hp`: coefficient `0.001251`, |coef| `0.001251`
- `lag_00__damage_diff_last_5s`: coefficient `0.001250`, |coef| `0.001250`
- `lag_07__T3__duck_amount`: coefficient `0.001250`, |coef| `0.001250`
- `lag_10__CT_place_CATWALK`: coefficient `-0.001245`, |coef| `0.001245`
- `lag_00__CT3__armor`: coefficient `0.001200`, |coef| `0.001200`
- `lag_14__bomb_events_last_5s`: coefficient `0.001194`, |coef| `0.001194`
- `lag_00__kill_diff_last_3s`: coefficient `0.001180`, |coef| `0.001180`
- `lag_06__CT_place_CATWALK`: coefficient `0.001164`, |coef| `0.001164`
- `lag_15__T2__duck_amount`: coefficient `-0.001097`, |coef| `0.001097`

## Top 10 utility ridge features

- `lag_02__T_B_site_active_infernos`: coefficient `0.000831` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000700` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000696` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000544` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.000511` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `-0.000483` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.000477` (lowers CT win probability)
- `lag_14__CT_B_site_active_smokes`: coefficient `-0.000432` (lowers CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `-0.000400` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000397` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_JUNGLE`: coefficient `0.002574` (raises CT win probability)
- `lag_06__CT_place_UNDERPASS`: coefficient `-0.001684` (lowers CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001656` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001554` (lowers CT win probability)
- `lag_10__CT_place_UNDERPASS`: coefficient `0.001435` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001269` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.001251` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001250` (raises CT win probability)
- `lag_07__T3__duck_amount`: coefficient `0.001250` (raises CT win probability)
- `lag_10__CT_place_CATWALK`: coefficient `-0.001245` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `29749`, seconds `30.00`, LSTM delta `-0.2117`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.016517`
- `lag_06__CT_place_UNDERPASS`: contribution `-0.009767`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.008319`
- `lag_02__T_place_CONNECTOR`: contribution `-0.008020`
- `lag_10__CT_place_CATWALK`: contribution `-0.004958`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30389`, seconds `40.00`, LSTM delta `-0.0684`

Top all feature movements:
- `lag_11__T_place_STAIRS`: contribution `-0.008286`
- `lag_00__T_kills_last_3s`: contribution `-0.004922`
- `lag_03__T5__flash_duration`: contribution `-0.003167`
- `lag_03__T_flash_duration_sum`: contribution `-0.002928`
- `lag_00__kill_diff_last_3s`: contribution `-0.002841`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.003167`
- `lag_03__T_flash_duration_sum`: contribution `-0.002928`
- `lag_03__T1__flash_duration`: contribution `-0.002782`
- `lag_03__T3__flash_duration`: contribution `-0.001442`

### tick `28629`, seconds `12.50`, LSTM delta `-0.0466`

Top all feature movements:
- `lag_14__CT_place_SHOP`: contribution `-0.008270`
- `lag_09__T_flashed_players`: contribution `-0.005259`
- `lag_01__CT_place_UNDERPASS`: contribution `-0.004586`
- `lag_07__T_flashed_players`: contribution `-0.003546`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.003529`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29781`, seconds `30.50`, LSTM delta `-0.0446`

Top all feature movements:
- `lag_01__CT_place_JUNGLE`: contribution `-0.006602`
- `lag_11__CT_place_UNDERPASS`: contribution `-0.004860`
- `lag_03__T_place_CONNECTOR`: contribution `-0.004114`
- `lag_12__T1__duck_amount`: contribution `+0.003749`
- `lag_11__CT_place_CATWALK`: contribution `-0.003219`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.001968`

### tick `28341`, seconds `8.00`, LSTM delta `-0.0385`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `-0.006445`
- `lag_05__CT_place_SHOP`: contribution `-0.005150`
- `lag_01__T_place_SIDEALLEY`: contribution `-0.002593`
- `lag_06__T_place_PALACEINTERIOR`: contribution `-0.002485`
- `lag_01__T_place_TOPOFMID`: contribution `-0.001695`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `-0.001136`
