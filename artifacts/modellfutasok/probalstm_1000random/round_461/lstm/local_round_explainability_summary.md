# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `40132`, seconds `51.50`, LSTM `0.2321`, delta `-0.2263`
- tick `39332`, seconds `39.00`, LSTM `0.8231`, delta `-0.1072`
- tick `40836`, seconds `62.50`, LSTM `0.0418`, delta `-0.1055`
- tick `40164`, seconds `52.00`, LSTM `0.1332`, delta `-0.0988`
- tick `39044`, seconds `34.50`, LSTM `0.7451`, delta `-0.0891`
- tick `39140`, seconds `36.00`, LSTM `0.8134`, delta `+0.0769`
- tick `39492`, seconds `41.50`, LSTM `0.6470`, delta `-0.0694`
- tick `39172`, seconds `36.50`, LSTM `0.8810`, delta `+0.0676`
- tick `40932`, seconds `64.00`, LSTM `0.0790`, delta `+0.0570`
- tick `39716`, seconds `45.00`, LSTM `0.5379`, delta `-0.0517`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003176`, |coef| `0.003176`
- `lag_00__kill_diff_last_3s`: coefficient `0.002909`, |coef| `0.002909`
- `lag_00__T_kills_last_3s`: coefficient `-0.002673`, |coef| `0.002673`
- `lag_06__T_bomb_zone_count`: coefficient `0.002475`, |coef| `0.002475`
- `lag_14__T_bomb_zone_count`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_00__CT2__utility_total`: coefficient `0.002392`, |coef| `0.002392`
- `lag_00__CT2__flash`: coefficient `0.002294`, |coef| `0.002294`
- `lag_00__T_damage_last_5s`: coefficient `-0.002287`, |coef| `0.002287`
- `lag_13__CT_place_LIBRARY`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_15__T_bomb_zone_count`: coefficient `-0.001641`, |coef| `0.001641`
- `lag_06__bomb_planted`: coefficient `-0.001612`, |coef| `0.001612`
- `lag_00__CT2__molly`: coefficient `0.001564`, |coef| `0.001564`
- `lag_02__T4__is_walking`: coefficient `-0.001563`, |coef| `0.001563`
- `lag_00__CT2__alive`: coefficient `0.001536`, |coef| `0.001536`
- `lag_00__CT2__hp`: coefficient `0.001518`, |coef| `0.001518`

## Top 10 utility ridge features

- `lag_00__CT2__utility_total`: coefficient `0.002392` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.002294` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.001564` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.001376` (raises CT win probability)
- `lag_10__T3__molly`: coefficient `-0.001350` (lowers CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.001259` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.001232` (raises CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.001168` (raises CT win probability)
- `lag_00__utility_inv_diff`: coefficient `0.001149` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001102` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.003176` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002909` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002673` (lowers CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `0.002475` (raises CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.002442` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002287` (lowers CT win probability)
- `lag_13__CT_place_LIBRARY`: coefficient `-0.001711` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `-0.001641` (lowers CT win probability)
- `lag_06__bomb_planted`: coefficient `-0.001612` (lowers CT win probability)
- `lag_02__T4__is_walking`: coefficient `-0.001563` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40132`, seconds `51.50`, LSTM delta `-0.2263`

Top all feature movements:
- `lag_06__T_bomb_zone_count`: contribution `-0.014407`
- `lag_14__T_bomb_zone_count`: contribution `-0.014214`
- `lag_13__CT_place_LIBRARY`: contribution `-0.010973`
- `lag_00__CT2__utility_total`: contribution `-0.009017`
- `lag_00__T_kills_last_3s`: contribution `-0.008467`

Top utility-only movements:
- `lag_00__CT2__utility_total`: contribution `-0.009017`
- `lag_00__CT2__flash`: contribution `-0.008297`
- `lag_00__CT2__molly`: contribution `-0.003856`

### tick `39332`, seconds `39.00`, LSTM delta `-0.1072`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008467`
- `lag_12__CT2__is_scoped`: contribution `-0.007085`
- `lag_00__kill_diff_last_3s`: contribution `-0.007001`
- `lag_04__CT_shots_fired_sum`: contribution `-0.005600`
- `lag_00__damage_diff_last_5s`: contribution `-0.005517`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40836`, seconds `62.50`, LSTM delta `-0.1055`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008467`
- `lag_00__damage_diff_last_5s`: contribution `-0.007164`
- `lag_00__kill_diff_last_3s`: contribution `-0.007001`
- `lag_00__T_damage_last_5s`: contribution `-0.005483`
- `lag_02__T4__is_walking`: contribution `-0.003608`

Top utility-only movements:
- `lag_00__CT1__utility_total`: contribution `-0.003022`
- `lag_00__CT1__molly`: contribution `-0.002361`

### tick `40164`, seconds `52.00`, LSTM delta `-0.0988`

Top all feature movements:
- `lag_15__T_bomb_zone_count`: contribution `-0.009554`
- `lag_07__T_bomb_zone_count`: contribution `-0.007919`
- `lag_12__CT2__is_scoped`: contribution `-0.007085`
- `lag_14__CT_place_LIBRARY`: contribution `-0.006234`
- `lag_01__CT_place_LIBRARY`: contribution `-0.005176`

Top utility-only movements:
- `lag_01__CT2__utility_total`: contribution `-0.003590`
- `lag_01__CT2__flash`: contribution `-0.003233`
- `lag_11__T3__molly`: contribution `-0.001673`

### tick `39044`, seconds `34.50`, LSTM delta `-0.0891`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008467`
- `lag_00__damage_diff_last_5s`: contribution `-0.007164`
- `lag_00__kill_diff_last_3s`: contribution `-0.007001`
- `lag_00__T_damage_last_5s`: contribution `-0.005483`
- `lag_03__CT2__is_scoped`: contribution `-0.003446`

Top utility-only movements:
- `lag_00__CT3__utility_total`: contribution `-0.001874`
