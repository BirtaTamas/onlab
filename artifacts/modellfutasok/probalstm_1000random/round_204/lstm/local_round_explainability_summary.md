# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `62134`, seconds `64.00`, LSTM `0.8322`, delta `+0.3086`
- tick `59958`, seconds `30.00`, LSTM `0.5694`, delta `+0.1743`
- tick `59894`, seconds `29.00`, LSTM `0.3947`, delta `-0.1500`
- tick `59830`, seconds `28.00`, LSTM `0.5718`, delta `-0.1483`
- tick `59766`, seconds `27.00`, LSTM `0.7017`, delta `+0.1426`
- tick `60150`, seconds `33.00`, LSTM `0.5620`, delta `-0.0555`
- tick `63510`, seconds `85.50`, LSTM `0.9074`, delta `-0.0426`
- tick `62166`, seconds `64.50`, LSTM `0.8718`, delta `+0.0395`
- tick `60022`, seconds `31.00`, LSTM `0.5934`, delta `+0.0321`
- tick `59638`, seconds `25.00`, LSTM `0.5749`, delta `+0.0301`

## Top 15 local ridge features

- `lag_00__T_place_RAMP`: coefficient `-0.004169`, |coef| `0.004169`
- `lag_00__kill_diff_last_3s`: coefficient `0.003810`, |coef| `0.003810`
- `lag_00__CT_kills_last_3s`: coefficient `0.003687`, |coef| `0.003687`
- `lag_13__T_place_CONTROL`: coefficient `-0.002912`, |coef| `0.002912`
- `lag_07__CT_place_OBSERVATION`: coefficient `-0.002304`, |coef| `0.002304`
- `lag_00__T_spread_xy`: coefficient `-0.002272`, |coef| `0.002272`
- `lag_00__T2__has_bomb`: coefficient `-0.002240`, |coef| `0.002240`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002101`, |coef| `0.002101`
- `lag_05__CT_place_OBSERVATION`: coefficient `0.001998`, |coef| `0.001998`
- `lag_13__T_place_RAMP`: coefficient `0.001978`, |coef| `0.001978`
- `lag_10__T4__duck_amount`: coefficient `0.001937`, |coef| `0.001937`
- `lag_00__T2__shots_fired`: coefficient `0.001882`, |coef| `0.001882`
- `lag_08__T4__duck_amount`: coefficient `-0.001810`, |coef| `0.001810`
- `lag_00__T4__alive`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_00__T2__alive`: coefficient `-0.001736`, |coef| `0.001736`

## Top 10 utility ridge features

- `lag_00__T2__smoke`: coefficient `-0.001590` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001574` (lowers CT win probability)
- `lag_14__T4__molly`: coefficient `0.001410` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001292` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001099` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.001000` (lowers CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000923` (raises CT win probability)
- `lag_13__T4__molly`: coefficient `0.000706` (raises CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000645` (lowers CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000639` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_RAMP`: coefficient `-0.004169` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003810` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003687` (raises CT win probability)
- `lag_13__T_place_CONTROL`: coefficient `-0.002912` (lowers CT win probability)
- `lag_07__CT_place_OBSERVATION`: coefficient `-0.002304` (lowers CT win probability)
- `lag_00__T_spread_xy`: coefficient `-0.002272` (lowers CT win probability)
- `lag_00__T2__has_bomb`: coefficient `-0.002240` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002101` (raises CT win probability)
- `lag_05__CT_place_OBSERVATION`: coefficient `0.001998` (raises CT win probability)
- `lag_13__T_place_RAMP`: coefficient `0.001978` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `62134`, seconds `64.00`, LSTM delta `+0.3086`

Top all feature movements:
- `lag_00__T_place_RAMP`: contribution `+0.029491`
- `lag_00__CT_kills_last_3s`: contribution `+0.021288`
- `lag_13__T_place_CONTROL`: contribution `+0.020689`
- `lag_00__kill_diff_last_3s`: contribution `+0.018339`
- `lag_00__T_spread_xy`: contribution `+0.007647`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `+0.003492`

### tick `59958`, seconds `30.00`, LSTM delta `+0.1743`

Top all feature movements:
- `lag_07__CT_place_OBSERVATION`: contribution `+0.040132`
- `lag_11__CT_place_OBSERVATION`: contribution `+0.022962`
- `lag_13__T_place_CONTROL`: contribution `-0.020689`
- `lag_00__T_place_CONTROL`: contribution `+0.006933`
- `lag_12__T_place_CONTROL`: contribution `-0.006713`

Top utility-only movements:
- `lag_10__T_flash_duration_sum`: contribution `+0.005517`
- `lag_10__T2__flash_duration`: contribution `+0.002988`

### tick `59894`, seconds `29.00`, LSTM delta `-0.1500`

Top all feature movements:
- `lag_05__CT_place_OBSERVATION`: contribution `-0.034803`
- `lag_09__CT_place_OBSERVATION`: contribution `-0.020042`
- `lag_00__kill_diff_last_3s`: contribution `-0.009170`
- `lag_00__T2__shots_fired`: contribution `-0.008861`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007297`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `-0.005449`
- `lag_08__T2__flash_duration`: contribution `-0.003141`
- `lag_08__T4__flash_duration`: contribution `-0.003023`

### tick `59830`, seconds `28.00`, LSTM delta `-0.1483`

Top all feature movements:
- `lag_07__CT_place_OBSERVATION`: contribution `-0.040132`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.019123`
- `lag_00__kill_diff_last_3s`: contribution `-0.009170`
- `lag_09__T_place_CONTROL`: contribution `-0.006726`
- `lag_01__T_place_RAMP`: contribution `-0.005786`

Top utility-only movements:
- `lag_06__T_flash_duration_sum`: contribution `-0.005191`
- `lag_06__T2__flash_duration`: contribution `-0.003701`

### tick `59766`, seconds `27.00`, LSTM delta `+0.1426`

Top all feature movements:
- `lag_05__CT_place_OBSERVATION`: contribution `+0.034803`
- `lag_00__T_place_RAMP`: contribution `+0.014745`
- `lag_00__CT_kills_last_3s`: contribution `+0.010644`
- `lag_00__kill_diff_last_3s`: contribution `+0.009170`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005837`

Top utility-only movements:
- `lag_04__T_flash_duration_sum`: contribution `+0.004403`
- `lag_04__T4__flash_duration`: contribution `+0.002635`
- `lag_04__T1__flash_duration`: contribution `+0.002136`
