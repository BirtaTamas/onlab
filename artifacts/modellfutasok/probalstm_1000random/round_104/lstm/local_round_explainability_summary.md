# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `14841`, seconds `14.50`, LSTM `0.4129`, delta `-0.2090`
- tick `16473`, seconds `40.00`, LSTM `0.3168`, delta `-0.2001`
- tick `16697`, seconds `43.50`, LSTM `0.0723`, delta `-0.1831`
- tick `15385`, seconds `23.00`, LSTM `0.5347`, delta `+0.1432`
- tick `14969`, seconds `16.50`, LSTM `0.3304`, delta `-0.0690`
- tick `15129`, seconds `19.00`, LSTM `0.3809`, delta `+0.0479`
- tick `15001`, seconds `17.00`, LSTM `0.2836`, delta `-0.0468`
- tick `15353`, seconds `22.50`, LSTM `0.3915`, delta `-0.0432`
- tick `15193`, seconds `20.00`, LSTM `0.4460`, delta `+0.0418`
- tick `16505`, seconds `40.50`, LSTM `0.2802`, delta `-0.0366`

## Top 15 local ridge features

- `lag_10__T_place_IVY`: coefficient `0.002156`, |coef| `0.002156`
- `lag_00__T_kills_last_3s`: coefficient `-0.002156`, |coef| `0.002156`
- `lag_00__kill_diff_last_3s`: coefficient `0.001823`, |coef| `0.001823`
- `lag_01__T3__flash_duration`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_13__T5__flash_duration`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_14__T2__has_bomb`: coefficient `-0.001455`, |coef| `0.001455`
- `lag_12__T_place_DUMPSTER`: coefficient `-0.001430`, |coef| `0.001430`
- `lag_11__CT1__is_walking`: coefficient `0.001396`, |coef| `0.001396`
- `lag_00__T_place_TMAIN`: coefficient `0.001329`, |coef| `0.001329`
- `lag_12__T_A_site_active_infernos`: coefficient `-0.001314`, |coef| `0.001314`
- `lag_11__CT3__duck_amount`: coefficient `0.001305`, |coef| `0.001305`
- `lag_12__CT5__is_walking`: coefficient `0.001288`, |coef| `0.001288`
- `lag_10__T_shots_fired_sum`: coefficient `-0.001284`, |coef| `0.001284`
- `lag_12__T2__duck_amount`: coefficient `-0.001245`, |coef| `0.001245`
- `lag_00__CT2__shots_fired`: coefficient `-0.001240`, |coef| `0.001240`

## Top 10 utility ridge features

- `lag_01__T3__flash_duration`: coefficient `-0.001529` (lowers CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.001472` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.001314` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.001171` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.001061` (raises CT win probability)
- `lag_12__active_infernos_total`: coefficient `-0.001002` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `-0.001002` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.001001` (raises CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `-0.000978` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000964` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_IVY`: coefficient `0.002156` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002156` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001823` (raises CT win probability)
- `lag_14__T2__has_bomb`: coefficient `-0.001455` (lowers CT win probability)
- `lag_12__T_place_DUMPSTER`: coefficient `-0.001430` (lowers CT win probability)
- `lag_11__CT1__is_walking`: coefficient `0.001396` (raises CT win probability)
- `lag_00__T_place_TMAIN`: coefficient `0.001329` (raises CT win probability)
- `lag_11__CT3__duck_amount`: coefficient `0.001305` (raises CT win probability)
- `lag_12__CT5__is_walking`: coefficient `0.001288` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `-0.001284` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14841`, seconds `14.50`, LSTM delta `-0.2090`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `-0.014059`
- `lag_14__T_place_DUMPSTER`: contribution `-0.009473`
- `lag_13__T5__flash_duration`: contribution `-0.007803`
- `lag_00__T_kills_last_3s`: contribution `-0.006831`
- `lag_01__CT2__flash_duration`: contribution `-0.005682`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.007803`
- `lag_01__CT2__flash_duration`: contribution `-0.005682`
- `lag_13__CT4__flash_duration`: contribution `-0.005387`
- `lag_10__CT2__flash_duration`: contribution `-0.004773`
- `lag_04__CT4__flash_duration`: contribution `-0.004459`

### tick `16473`, seconds `40.00`, LSTM delta `-0.2001`

Top all feature movements:
- `lag_10__T_place_IVY`: contribution `-0.011523`
- `lag_00__T_kills_last_3s`: contribution `-0.006831`
- `lag_11__CT3__duck_amount`: contribution `-0.004857`
- `lag_12__T2__duck_amount`: contribution `-0.004761`
- `lag_14__T2__has_bomb`: contribution `-0.004542`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.003910`
- `lag_00__CT2__molly`: contribution `-0.002888`
- `lag_08__T_A_site_active_infernos`: contribution `-0.002699`

### tick `16697`, seconds `43.50`, LSTM delta `-0.1831`

Top all feature movements:
- `lag_01__T3__flash_duration`: contribution `-0.009243`
- `lag_00__T_kills_last_3s`: contribution `-0.006831`
- `lag_01__CT1__flash_duration`: contribution `-0.004671`
- `lag_01__T_flashed_players`: contribution `-0.004562`
- `lag_00__CT_flashed_players`: contribution `-0.004472`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.009243`
- `lag_01__CT1__flash_duration`: contribution `-0.004671`
- `lag_01__T_flash_duration_sum`: contribution `-0.003954`
- `lag_00__CT1__flash_duration`: contribution `-0.002523`
- `lag_01__T2__flash_duration`: contribution `-0.002157`

### tick `15385`, seconds `23.00`, LSTM delta `+0.1432`

Top all feature movements:
- `lag_12__T_place_DUMPSTER`: contribution `+0.026014`
- `lag_00__T_place_TMAIN`: contribution `+0.010305`
- `lag_03__CT_shots_fired_sum`: contribution `+0.009893`
- `lag_03__CT5__shots_fired`: contribution `+0.009828`
- `lag_01__T_place_TMAIN`: contribution `+0.006065`

Top utility-only movements:
- `lag_12__CT_A_site_active_infernos`: contribution `+0.003375`
- `lag_08__T1__flash_duration`: contribution `+0.003340`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.002095`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001971`
- `lag_13__CT_flash_duration_sum`: contribution `-0.001881`

### tick `14969`, seconds `16.50`, LSTM delta `-0.0690`

Top all feature movements:
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.005759`
- `lag_03__T_shots_fired_sum`: contribution `+0.005300`
- `lag_00__CT_flashed_players`: contribution `+0.004472`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.004317`
- `lag_14__CT_place_BACKOFB`: contribution `-0.003230`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `-0.003185`
- `lag_00__CT5__flash_duration`: contribution `-0.002802`
- `lag_00__CT1__flash_duration`: contribution `+0.002322`
- `lag_11__T_A_site_active_infernos`: contribution `-0.002297`
- `lag_05__CT2__flash_duration`: contribution `-0.002114`
