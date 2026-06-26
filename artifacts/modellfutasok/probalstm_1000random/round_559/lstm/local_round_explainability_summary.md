# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `4032`, seconds `43.50`, LSTM `0.7116`, delta `+0.5160`
- tick `3808`, seconds `40.00`, LSTM `0.3454`, delta `-0.2770`
- tick `3744`, seconds `39.00`, LSTM `0.7040`, delta `+0.2655`
- tick `2816`, seconds `24.50`, LSTM `0.4880`, delta `-0.2094`
- tick `4512`, seconds `51.00`, LSTM `0.9146`, delta `+0.1218`
- tick `3008`, seconds `27.50`, LSTM `0.4655`, delta `+0.0857`
- tick `3776`, seconds `39.50`, LSTM `0.6225`, delta `-0.0815`
- tick `4192`, seconds `46.00`, LSTM `0.7932`, delta `+0.0789`
- tick `3968`, seconds `42.50`, LSTM `0.2376`, delta `-0.0704`
- tick `4128`, seconds `45.00`, LSTM `0.7170`, delta `+0.0682`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004748`, |coef| `0.004748`
- `lag_00__T_place_VENTS`: coefficient `-0.004277`, |coef| `0.004277`
- `lag_15__T_bomb_zone_count`: coefficient `-0.004215`, |coef| `0.004215`
- `lag_02__T_place_VENTS`: coefficient `0.003991`, |coef| `0.003991`
- `lag_00__CT_kills_last_3s`: coefficient `0.003953`, |coef| `0.003953`
- `lag_03__CT_place_RAFTERS`: coefficient `0.002725`, |coef| `0.002725`
- `lag_06__T2__flash_duration`: coefficient `-0.002656`, |coef| `0.002656`
- `lag_03__CT_place_HEAVEN`: coefficient `-0.002618`, |coef| `0.002618`
- `lag_00__damage_diff_last_5s`: coefficient `0.002543`, |coef| `0.002543`
- `lag_00__T_place_HUT`: coefficient `-0.002369`, |coef| `0.002369`
- `lag_10__CT1__duck_amount`: coefficient `0.002360`, |coef| `0.002360`
- `lag_13__T2__flash_duration`: coefficient `0.002350`, |coef| `0.002350`
- `lag_00__CT_place_HEAVEN`: coefficient `0.002240`, |coef| `0.002240`
- `lag_14__T1__duck_amount`: coefficient `-0.002031`, |coef| `0.002031`
- `lag_00__T_kills_last_3s`: coefficient `-0.001912`, |coef| `0.001912`

## Top 10 utility ridge features

- `lag_06__T2__flash_duration`: coefficient `-0.002656` (lowers CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.002350` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.001481` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.001266` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.001231` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.001154` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001146` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.001084` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001083` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `-0.001073` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004748` (raises CT win probability)
- `lag_00__T_place_VENTS`: coefficient `-0.004277` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `-0.004215` (lowers CT win probability)
- `lag_02__T_place_VENTS`: coefficient `0.003991` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003953` (raises CT win probability)
- `lag_03__CT_place_RAFTERS`: coefficient `0.002725` (raises CT win probability)
- `lag_03__CT_place_HEAVEN`: coefficient `-0.002618` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002543` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.002369` (lowers CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `0.002360` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `4032`, seconds `43.50`, LSTM delta `+0.5160`

Top all feature movements:
- `lag_00__T_place_VENTS`: contribution `+0.057689`
- `lag_02__T_place_VENTS`: contribution `+0.053827`
- `lag_15__T_bomb_zone_count`: contribution `+0.024535`
- `lag_00__kill_diff_last_3s`: contribution `+0.022856`
- `lag_00__CT_kills_last_3s`: contribution `+0.022824`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.018484`
- `lag_13__T_flash_duration_sum`: contribution `+0.005313`

### tick `3808`, seconds `40.00`, LSTM delta `-0.2770`

Top all feature movements:
- `lag_15__T_bomb_zone_count`: contribution `-0.024535`
- `lag_06__T2__flash_duration`: contribution `-0.020891`
- `lag_03__CT_place_RAFTERS`: contribution `-0.014563`
- `lag_03__CT_place_HEAVEN`: contribution `-0.014134`
- `lag_00__CT_place_HEAVEN`: contribution `-0.012092`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.020891`
- `lag_06__T_flash_duration_sum`: contribution `-0.006214`

### tick `3744`, seconds `39.00`, LSTM delta `+0.2655`

Top all feature movements:
- `lag_03__CT_place_RAFTERS`: contribution `+0.014563`
- `lag_03__CT_place_HEAVEN`: contribution `+0.014134`
- `lag_00__kill_diff_last_3s`: contribution `+0.011428`
- `lag_00__CT_kills_last_3s`: contribution `+0.011412`
- `lag_04__T2__flash_duration`: contribution `+0.009079`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.009079`
- `lag_04__T_flash_duration_sum`: contribution `+0.003403`

### tick `2816`, seconds `24.50`, LSTM delta `-0.2094`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.022856`
- `lag_06__T_place_HUT`: contribution `-0.013233`
- `lag_05__T_place_HUT`: contribution `-0.012596`
- `lag_00__CT_kills_last_3s`: contribution `-0.011412`
- `lag_03__CT4__flash_duration`: contribution `-0.009561`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `-0.009561`
- `lag_03__CT_flash_duration_sum`: contribution `-0.003846`
- `lag_00__CT2__flash_duration`: contribution `-0.002545`

### tick `4512`, seconds `51.00`, LSTM delta `+0.1218`

Top all feature movements:
- `lag_15__T_place_VENTS`: contribution `+0.014223`
- `lag_00__kill_diff_last_3s`: contribution `+0.011428`
- `lag_00__CT_kills_last_3s`: contribution `+0.011412`
- `lag_15__T_place_HUT`: contribution `+0.010111`
- `lag_10__CT1__duck_amount`: contribution `+0.008724`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.003712`
- `lag_14__T2__flash_duration`: contribution `+0.001949`
