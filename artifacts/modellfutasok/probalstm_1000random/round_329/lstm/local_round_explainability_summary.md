# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `31217`, seconds `23.00`, LSTM `0.5319`, delta `+0.3293`
- tick `31857`, seconds `33.00`, LSTM `0.7700`, delta `+0.1640`
- tick `30705`, seconds `15.00`, LSTM `0.3093`, delta `-0.1530`
- tick `32337`, seconds `40.50`, LSTM `0.7796`, delta `-0.1203`
- tick `33809`, seconds `63.50`, LSTM `0.8760`, delta `+0.1089`
- tick `32177`, seconds `38.00`, LSTM `0.8912`, delta `+0.1053`
- tick `30737`, seconds `15.50`, LSTM `0.2284`, delta `-0.0809`
- tick `32369`, seconds `41.00`, LSTM `0.7050`, delta `-0.0747`
- tick `32945`, seconds `50.00`, LSTM `0.6976`, delta `-0.0682`
- tick `32657`, seconds `45.50`, LSTM `0.8454`, delta `+0.0523`

## Top 15 local ridge features

- `lag_13__T_shots_fired_sum`: coefficient `-0.002640`, |coef| `0.002640`
- `lag_07__T4__flash_duration`: coefficient `0.002509`, |coef| `0.002509`
- `lag_00__kill_diff_last_3s`: coefficient `0.002506`, |coef| `0.002506`
- `lag_13__T4__shots_fired`: coefficient `-0.002419`, |coef| `0.002419`
- `lag_00__CT_kills_last_3s`: coefficient `0.002147`, |coef| `0.002147`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_00__damage_diff_last_5s`: coefficient `0.001708`, |coef| `0.001708`
- `lag_00__T4__flash_duration`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_07__CT3__is_scoped`: coefficient `0.001543`, |coef| `0.001543`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001454`, |coef| `0.001454`
- `lag_03__CT_place_UNDERPASS`: coefficient `0.001349`, |coef| `0.001349`
- `lag_01__T_place_TRUCK`: coefficient `-0.001336`, |coef| `0.001336`
- `lag_00__CT_damage_last_5s`: coefficient `0.001322`, |coef| `0.001322`
- `lag_03__CT3__is_scoped`: coefficient `-0.001316`, |coef| `0.001316`
- `lag_15__T_place_SIDEALLEY`: coefficient `-0.001315`, |coef| `0.001315`

## Top 10 utility ridge features

- `lag_07__T4__flash_duration`: coefficient `0.002509` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001674` (lowers CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.001253` (raises CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.001214` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.001182` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001138` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001057` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001033` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000999` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000926` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_shots_fired_sum`: coefficient `-0.002640` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002506` (raises CT win probability)
- `lag_13__T4__shots_fired`: coefficient `-0.002419` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002147` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001899` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001708` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `0.001543` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001454` (raises CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `0.001349` (raises CT win probability)
- `lag_01__T_place_TRUCK`: coefficient `-0.001336` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `31217`, seconds `23.00`, LSTM delta `+0.3293`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `+0.033644`
- `lag_13__T4__shots_fired`: contribution `+0.025404`
- `lag_07__T4__flash_duration`: contribution `+0.019663`
- `lag_00__T4__flash_duration`: contribution `+0.013123`
- `lag_07__CT3__is_scoped`: contribution `+0.007018`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `+0.019663`
- `lag_00__T4__flash_duration`: contribution `+0.013123`
- `lag_00__T3__flash_duration`: contribution `+0.006100`
- `lag_07__T_flash_duration_sum`: contribution `+0.003994`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.003631`

### tick `31857`, seconds `33.00`, LSTM delta `+0.1640`

Top all feature movements:
- `lag_03__CT_place_UNDERPASS`: contribution `+0.007821`
- `lag_00__CT_kills_last_3s`: contribution `+0.006198`
- `lag_00__kill_diff_last_3s`: contribution `+0.006032`
- `lag_11__CT_place_CONNECTOR`: contribution `+0.004395`
- `lag_00__damage_diff_last_5s`: contribution `+0.003852`

Top utility-only movements:
- `lag_00__T1__flash`: contribution `+0.002565`

### tick `30705`, seconds `15.00`, LSTM delta `-0.1530`

Top all feature movements:
- `lag_14__CT1__flash_duration`: contribution `-0.008159`
- `lag_02__CT_place_UNDERPASS`: contribution `-0.007457`
- `lag_04__CT1__flash_duration`: contribution `-0.007180`
- `lag_00__T_shots_fired_sum`: contribution `-0.007118`
- `lag_00__CT_place_TRUCK`: contribution `-0.006613`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.008159`
- `lag_04__CT1__flash_duration`: contribution `-0.007180`
- `lag_07__T_flash_duration_sum`: contribution `-0.005386`
- `lag_07__T3__flash_duration`: contribution `-0.003370`
- `lag_07__T1__flash_duration`: contribution `-0.002347`

### tick `32337`, seconds `40.50`, LSTM delta `-0.1203`

Top all feature movements:
- `lag_01__T_place_TRUCK`: contribution `-0.023196`
- `lag_00__T_shots_fired_sum`: contribution `-0.007118`
- `lag_00__kill_diff_last_3s`: contribution `-0.006032`
- `lag_11__CT_place_CONNECTOR`: contribution `-0.004395`
- `lag_06__CT5__duck_amount`: contribution `-0.003607`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.001638`

### tick `33809`, seconds `63.50`, LSTM delta `+0.1089`

Top all feature movements:
- `lag_05__CT_place_PALACEALLEY`: contribution `+0.014834`
- `lag_10__CT_place_TRAMP`: contribution `+0.012145`
- `lag_05__CT_place_TRAMP`: contribution `+0.007814`
- `lag_00__T_place_SHOP`: contribution `+0.007113`
- `lag_00__CT_kills_last_3s`: contribution `+0.006198`

Top utility-only movements:
- No utility movement among the top local contributors.
