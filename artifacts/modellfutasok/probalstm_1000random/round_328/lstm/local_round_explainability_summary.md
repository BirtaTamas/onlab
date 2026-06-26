# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `18390`, seconds `103.00`, LSTM `0.7671`, delta `+0.2322`
- tick `18486`, seconds `104.50`, LSTM `0.9480`, delta `+0.1168`
- tick `18454`, seconds `104.00`, LSTM `0.8313`, delta `+0.1130`
- tick `14774`, seconds `46.50`, LSTM `0.3136`, delta `-0.0972`
- tick `18230`, seconds `100.50`, LSTM `0.5275`, delta `+0.0861`
- tick `17398`, seconds `87.50`, LSTM `0.2929`, delta `-0.0785`
- tick `13174`, seconds `21.50`, LSTM `0.5251`, delta `-0.0754`
- tick `18166`, seconds `99.50`, LSTM `0.4576`, delta `-0.0745`
- tick `14870`, seconds `48.00`, LSTM `0.3408`, delta `+0.0686`
- tick `17782`, seconds `93.50`, LSTM `0.4347`, delta `+0.0656`

## Top 15 local ridge features

- `lag_00__CT_place_OBSERVATION`: coefficient `-0.002513`, |coef| `0.002513`
- `lag_00__kill_diff_last_3s`: coefficient `0.002249`, |coef| `0.002249`
- `lag_00__CT_place_CRANE`: coefficient `-0.001888`, |coef| `0.001888`
- `lag_00__CT_kills_last_3s`: coefficient `0.001811`, |coef| `0.001811`
- `lag_13__CT_place_OBSERVATION`: coefficient `-0.001725`, |coef| `0.001725`
- `lag_00__damage_diff_last_5s`: coefficient `0.001719`, |coef| `0.001719`
- `lag_07__T_shots_fired_sum`: coefficient `0.001547`, |coef| `0.001547`
- `lag_06__T_place_RAMP`: coefficient `-0.001534`, |coef| `0.001534`
- `lag_07__CT_place_TUNNELS`: coefficient `-0.001510`, |coef| `0.001510`
- `lag_00__CT_damage_last_5s`: coefficient `0.001504`, |coef| `0.001504`
- `lag_00__CT2__duck_amount`: coefficient `0.001315`, |coef| `0.001315`
- `lag_06__T5__duck_amount`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_06__T_duck_amount_mean`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_01__T_kills_last_3s`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_04__T2__duck_amount`: coefficient `-0.001225`, |coef| `0.001225`

## Top 10 utility ridge features

- `lag_14__CT_B_site_active_infernos`: coefficient `-0.001054` (lowers CT win probability)
- `lag_07__CT1__molly`: coefficient `-0.001013` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000918` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000901` (lowers CT win probability)
- `lag_07__CT1__utility_total`: coefficient `-0.000755` (lowers CT win probability)
- `lag_07__CT1__flash`: coefficient `-0.000754` (lowers CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.000718` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000620` (lowers CT win probability)
- `lag_14__CT_active_infernos`: coefficient `-0.000608` (lowers CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.000575` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_OBSERVATION`: coefficient `-0.002513` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002249` (raises CT win probability)
- `lag_00__CT_place_CRANE`: coefficient `-0.001888` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001811` (raises CT win probability)
- `lag_13__CT_place_OBSERVATION`: coefficient `-0.001725` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001719` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `0.001547` (raises CT win probability)
- `lag_06__T_place_RAMP`: coefficient `-0.001534` (lowers CT win probability)
- `lag_07__CT_place_TUNNELS`: coefficient `-0.001510` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001504` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18390`, seconds `103.00`, LSTM delta `+0.2322`

Top all feature movements:
- `lag_13__CT_place_OBSERVATION`: contribution `+0.030036`
- `lag_07__T_shots_fired_sum`: contribution `+0.009276`
- `lag_06__T_shots_fired_sum`: contribution `+0.006412`
- `lag_06__T_place_RAMP`: contribution `+0.005425`
- `lag_00__kill_diff_last_3s`: contribution `+0.005414`

Top utility-only movements:
- `lag_14__CT_B_site_active_infernos`: contribution `+0.003621`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.003239`

### tick `18486`, seconds `104.50`, LSTM delta `+0.1168`

Top all feature movements:
- `lag_00__CT_place_OBSERVATION`: contribution `+0.043768`
- `lag_09__T_shots_fired_sum`: contribution `-0.006779`
- `lag_00__T_flash_alpha_mean`: contribution `+0.005468`
- `lag_00__kill_diff_last_3s`: contribution `+0.005414`
- `lag_00__CT_kills_last_3s`: contribution `+0.005227`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.005468`

### tick `18454`, seconds `104.00`, LSTM delta `+0.1130`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.006779`
- `lag_00__T_shots_fired_sum`: contribution `+0.005118`
- `lag_00__CT2__duck_amount`: contribution `+0.005011`
- `lag_15__CT_place_OBSERVATION`: contribution `+0.003939`
- `lag_08__T5__duck_amount`: contribution `+0.003765`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14774`, seconds `46.50`, LSTM delta `-0.0972`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.030965`
- `lag_15__T_place_CONTROL`: contribution `-0.007254`
- `lag_09__T_place_CONTROL`: contribution `-0.007166`
- `lag_06__T_place_RAMP`: contribution `-0.005425`
- `lag_04__T2__duck_amount`: contribution `-0.004683`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18230`, seconds `100.50`, LSTM delta `+0.0861`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.005414`
- `lag_00__CT_kills_last_3s`: contribution `+0.005227`
- `lag_07__T_shots_fired_sum`: contribution `-0.004638`
- `lag_12__CT3__duck_amount`: contribution `+0.004112`
- `lag_02__T_shots_fired_sum`: contribution `+0.003636`

Top utility-only movements:
- No utility movement among the top local contributors.
