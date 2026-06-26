# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `53507`, seconds `100.50`, LSTM `0.8469`, delta `+0.2323`
- tick `50435`, seconds `52.50`, LSTM `0.7691`, delta `+0.1481`
- tick `52003`, seconds `77.00`, LSTM `0.7392`, delta `-0.1288`
- tick `50883`, seconds `59.50`, LSTM `0.8828`, delta `+0.1263`
- tick `50499`, seconds `53.50`, LSTM `0.6479`, delta `-0.0949`
- tick `53251`, seconds `96.50`, LSTM `0.6139`, delta `-0.0718`
- tick `52035`, seconds `77.50`, LSTM `0.6739`, delta `-0.0653`
- tick `53475`, seconds `100.00`, LSTM `0.6145`, delta `+0.0653`
- tick `52771`, seconds `89.00`, LSTM `0.7242`, delta `+0.0615`
- tick `53027`, seconds `93.00`, LSTM `0.7862`, delta `+0.0608`

## Top 15 local ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `-0.003660`, |coef| `0.003660`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002728`, |coef| `0.002728`
- `lag_00__kill_diff_last_3s`: coefficient `0.002613`, |coef| `0.002613`
- `lag_09__T_place_SIDEENTRANCE`: coefficient `-0.002534`, |coef| `0.002534`
- `lag_00__T3__is_scoped`: coefficient `0.002486`, |coef| `0.002486`
- `lag_11__CT_place_TSIDEUPPER`: coefficient `-0.002044`, |coef| `0.002044`
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `0.002039`, |coef| `0.002039`
- `lag_08__T_place_SIDEENTRANCE`: coefficient `-0.001951`, |coef| `0.001951`
- `lag_07__T_bomb_zone_count`: coefficient `0.001883`, |coef| `0.001883`
- `lag_00__CT_kills_last_3s`: coefficient `0.001788`, |coef| `0.001788`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001640`, |coef| `0.001640`
- `lag_00__T3__has_bomb`: coefficient `-0.001592`, |coef| `0.001592`
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_15__T5__is_walking`: coefficient `0.001515`, |coef| `0.001515`
- `lag_00__CT4__duck_amount`: coefficient `0.001484`, |coef| `0.001484`

## Top 10 utility ridge features

- `lag_09__CT_B_site_active_infernos`: coefficient `-0.001254` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.001118` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.000867` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000777` (lowers CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.000674` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000673` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.000672` (raises CT win probability)
- `lag_10__CT_active_infernos`: coefficient `-0.000655` (lowers CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.000604` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.000599` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TSIDEUPPER`: coefficient `-0.003660` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002728` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002613` (raises CT win probability)
- `lag_09__T_place_SIDEENTRANCE`: coefficient `-0.002534` (lowers CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.002486` (raises CT win probability)
- `lag_11__CT_place_TSIDEUPPER`: coefficient `-0.002044` (lowers CT win probability)
- `lag_12__CT_place_SIDEENTRANCE`: coefficient `0.002039` (raises CT win probability)
- `lag_08__T_place_SIDEENTRANCE`: coefficient `-0.001951` (lowers CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `0.001883` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001788` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `53507`, seconds `100.50`, LSTM delta `+0.2323`

Top all feature movements:
- `lag_12__CT_place_TSIDEUPPER`: contribution `+0.027515`
- `lag_00__T_bomb_zone_count`: contribution `+0.015878`
- `lag_09__T_place_SIDEENTRANCE`: contribution `+0.012368`
- `lag_07__T_bomb_zone_count`: contribution `+0.010960`
- `lag_01__CT_shots_fired_sum`: contribution `+0.009116`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `+0.003841`

### tick `50435`, seconds `52.50`, LSTM delta `+0.1481`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.006289`
- `lag_04__CT_flashed_players`: contribution `+0.006049`
- `lag_02__CT1__shots_fired`: contribution `+0.005628`
- `lag_00__CT_kills_last_3s`: contribution `+0.005162`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.003751`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52003`, seconds `77.00`, LSTM delta `-0.1288`

Top all feature movements:
- `lag_08__T_place_SIDEENTRANCE`: contribution `-0.009520`
- `lag_04__T3__is_scoped`: contribution `-0.009342`
- `lag_00__kill_diff_last_3s`: contribution `-0.006289`
- `lag_00__CT4__duck_amount`: contribution `-0.005451`
- `lag_05__T5__duck_amount`: contribution `-0.004871`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50883`, seconds `59.50`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.006289`
- `lag_00__CT_kills_last_3s`: contribution `+0.005162`
- `lag_00__CT4__duck_amount`: contribution `+0.005148`
- `lag_09__T_place_TSIDELOWER`: contribution `+0.004611`
- `lag_11__T_shots_fired_sum`: contribution `+0.004451`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `+0.004083`
- `lag_11__T2__flash_duration`: contribution `+0.003756`
- `lag_09__T1__flash_duration`: contribution `+0.003581`
- `lag_05__CT5__flash_duration`: contribution `+0.002644`
- `lag_11__CT_flash_duration_sum`: contribution `+0.002320`

### tick `50499`, seconds `53.50`, LSTM delta `-0.0949`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.007046`
- `lag_00__kill_diff_last_3s`: contribution `-0.006289`
- `lag_00__T_kills_last_3s`: contribution `-0.004680`
- `lag_09__T_place_TSIDELOWER`: contribution `-0.004611`
- `lag_04__CT1__shots_fired`: contribution `-0.003881`

Top utility-only movements:
- No utility movement among the top local contributors.
