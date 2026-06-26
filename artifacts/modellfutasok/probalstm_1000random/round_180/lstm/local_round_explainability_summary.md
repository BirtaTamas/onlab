# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `9706`, seconds `34.00`, LSTM `0.3120`, delta `-0.2056`
- tick `10378`, seconds `44.50`, LSTM `0.8561`, delta `+0.1907`
- tick `9930`, seconds `37.50`, LSTM `0.6684`, delta `+0.1494`
- tick `9770`, seconds `35.00`, LSTM `0.4953`, delta `+0.0985`
- tick `9738`, seconds `34.50`, LSTM `0.3967`, delta `+0.0847`
- tick `10410`, seconds `45.00`, LSTM `0.9331`, delta `+0.0771`
- tick `9994`, seconds `38.50`, LSTM `0.7077`, delta `+0.0466`
- tick `9834`, seconds `36.00`, LSTM `0.4960`, delta `+0.0449`
- tick `9802`, seconds `35.50`, LSTM `0.4511`, delta `-0.0441`
- tick `10186`, seconds `41.50`, LSTM `0.7225`, delta `+0.0338`

## Top 15 local ridge features

- `lag_12__CT_place_LIBRARY`: coefficient `0.001374`, |coef| `0.001374`
- `lag_08__T2__shots_fired`: coefficient `-0.001308`, |coef| `0.001308`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001293`, |coef| `0.001293`
- `lag_08__T_shots_fired_sum`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001192`, |coef| `0.001192`
- `lag_00__kill_diff_last_3s`: coefficient `0.001137`, |coef| `0.001137`
- `lag_13__T_place_ARCH`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_14__CT_shots_fired_sum`: coefficient `0.001095`, |coef| `0.001095`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001064`, |coef| `0.001064`
- `lag_00__T_flashed_players`: coefficient `-0.001047`, |coef| `0.001047`
- `lag_01__CT4__flash_duration`: coefficient `-0.001018`, |coef| `0.001018`
- `lag_02__T4__flash_duration`: coefficient `-0.000978`, |coef| `0.000978`
- `lag_14__CT5__shots_fired`: coefficient `0.000978`, |coef| `0.000978`
- `lag_00__CT_kills_last_3s`: coefficient `0.000976`, |coef| `0.000976`
- `lag_15__CT_shots_fired_sum`: coefficient `-0.000949`, |coef| `0.000949`

## Top 10 utility ridge features

- `lag_01__CT4__flash_duration`: coefficient `-0.001018` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000978` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.000899` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000898` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.000806` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.000754` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.000746` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000639` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.000624` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_LIBRARY`: coefficient `0.001374` (raises CT win probability)
- `lag_08__T2__shots_fired`: coefficient `-0.001308` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001293` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.001240` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001192` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001137` (raises CT win probability)
- `lag_13__T_place_ARCH`: coefficient `-0.001128` (lowers CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.001095` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001064` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001047` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9706`, seconds `34.00`, LSTM delta `-0.2056`

Top all feature movements:
- `lag_14__CT_shots_fired_sum`: contribution `-0.012931`
- `lag_13__T_place_ARCH`: contribution `-0.010492`
- `lag_12__CT_place_LIBRARY`: contribution `-0.008807`
- `lag_14__CT5__shots_fired`: contribution `-0.008789`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008281`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.007731`
- `lag_00__T1__flash_duration`: contribution `-0.005537`
- `lag_04__T4__flash_duration`: contribution `-0.004649`
- `lag_00__CT4__flash_duration`: contribution `-0.003703`
- `lag_04__T_flash_duration_sum`: contribution `-0.003499`

### tick `10378`, seconds `44.50`, LSTM delta `+0.1907`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.016737`
- `lag_08__T2__shots_fired`: contribution `+0.013848`
- `lag_12__CT_place_LIBRARY`: contribution `+0.008807`
- `lag_02__T4__flash_duration`: contribution `+0.007640`
- `lag_07__CT_place_LIBRARY`: contribution `+0.005201`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.007640`
- `lag_04__T1__flash_duration`: contribution `+0.004914`
- `lag_06__T2__flash_duration`: contribution `+0.004358`
- `lag_04__T_flash_duration_sum`: contribution `+0.002463`

### tick `9930`, seconds `37.50`, LSTM delta `+0.1494`

Top all feature movements:
- `lag_05__T_place_ARCH`: contribution `+0.008759`
- `lag_06__T_shots_fired_sum`: contribution `+0.005720`
- `lag_08__T_shots_fired_sum`: contribution `-0.005579`
- `lag_02__T4__flash_duration`: contribution `-0.005364`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004969`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.005364`
- `lag_07__CT4__flash_duration`: contribution `+0.004045`
- `lag_08__CT4__flash_duration`: contribution `+0.003561`
- `lag_07__T1__flash_duration`: contribution `+0.002765`
- `lag_02__T2__flash_duration`: contribution `+0.002494`

### tick `9770`, seconds `35.00`, LSTM delta `+0.0985`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.010372`
- `lag_00__T_place_ARCH`: contribution `+0.007004`
- `lag_01__T3__shots_fired`: contribution `+0.006145`
- `lag_03__CT_shots_fired_sum`: contribution `+0.004766`
- `lag_15__T_place_ARCH`: contribution `+0.004675`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.003377`
- `lag_03__CT4__flash_duration`: contribution `+0.003087`
- `lag_02__T4__flash_duration`: contribution `+0.002781`
- `lag_11__T3__flash_duration`: contribution `+0.001575`
- `lag_02__CT4__flash_duration`: contribution `+0.001538`

### tick `9738`, seconds `34.50`, LSTM delta `+0.0847`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.012607`
- `lag_15__CT_shots_fired_sum`: contribution `+0.011210`
- `lag_01__CT4__flash_duration`: contribution `+0.007731`
- `lag_15__CT5__shots_fired`: contribution `+0.006172`
- `lag_00__T3__shots_fired`: contribution `+0.005721`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.007731`
- `lag_05__T4__flash_duration`: contribution `-0.001871`
- `lag_05__T_flash_duration_sum`: contribution `-0.001667`
- `lag_01__CT_flash_duration_sum`: contribution `+0.001580`
- `lag_02__CT4__flash_duration`: contribution `-0.001538`
