# Local Round Explainability

- csv_path: `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9\vitality-vs-hotu-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `12869`, seconds `13.50`, LSTM `0.0717`, delta `-0.4433`
- tick `12901`, seconds `14.00`, LSTM `0.0224`, delta `-0.0493`
- tick `13925`, seconds `30.00`, LSTM `0.1344`, delta `+0.0448`
- tick `15269`, seconds `51.00`, LSTM `0.0126`, delta `-0.0430`
- tick `14181`, seconds `34.00`, LSTM `0.0658`, delta `-0.0347`
- tick `13829`, seconds `28.50`, LSTM `0.0942`, delta `+0.0337`
- tick `12517`, seconds `8.00`, LSTM `0.4658`, delta `-0.0265`
- tick `12581`, seconds `9.00`, LSTM `0.4715`, delta `+0.0227`
- tick `12677`, seconds `10.50`, LSTM `0.4950`, delta `+0.0226`
- tick `12773`, seconds `12.00`, LSTM `0.5033`, delta `-0.0189`

## Top 15 local ridge features

- `lag_08__CT_place_HOLE`: coefficient `0.002740`, |coef| `0.002740`
- `lag_09__CT_place_HOLE`: coefficient `-0.002628`, |coef| `0.002628`
- `lag_14__CT5__flash_duration`: coefficient `-0.002194`, |coef| `0.002194`
- `lag_14__CT_flashed_players`: coefficient `-0.002044`, |coef| `0.002044`
- `lag_00__T_kills_last_3s`: coefficient `-0.001998`, |coef| `0.001998`
- `lag_14__CT1__flash_duration`: coefficient `-0.001923`, |coef| `0.001923`
- `lag_03__T2__flash_duration`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_14__T_flashed_players`: coefficient `-0.001824`, |coef| `0.001824`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001823`, |coef| `0.001823`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001710`, |coef| `0.001710`
- `lag_03__T_flashed_players`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_14__T_place_LONGDOORS`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001631`, |coef| `0.001631`
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001539`, |coef| `0.001539`
- `lag_08__T_place_LONGDOORS`: coefficient `-0.001522`, |coef| `0.001522`

## Top 10 utility ridge features

- `lag_14__CT5__flash_duration`: coefficient `-0.002194` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.001923` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001850` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001631` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001241` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001233` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.001182` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.001168` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `-0.001162` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001074` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_HOLE`: coefficient `0.002740` (raises CT win probability)
- `lag_09__CT_place_HOLE`: coefficient `-0.002628` (lowers CT win probability)
- `lag_14__CT_flashed_players`: coefficient `-0.002044` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001998` (lowers CT win probability)
- `lag_14__T_flashed_players`: coefficient `-0.001824` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001823` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001710` (raises CT win probability)
- `lag_03__T_flashed_players`: coefficient `-0.001697` (lowers CT win probability)
- `lag_14__T_place_LONGDOORS`: coefficient `-0.001649` (lowers CT win probability)
- `lag_00__CT_place_LONGDOORS`: coefficient `0.001539` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `12869`, seconds `13.50`, LSTM delta `-0.4433`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `-0.030586`
- `lag_09__CT_place_HOLE`: contribution `-0.029343`
- `lag_00__T_shots_fired_sum`: contribution `-0.013670`
- `lag_14__CT_flashed_players`: contribution `-0.013426`
- `lag_00__T_kills_last_3s`: contribution `-0.012658`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `-0.011654`
- `lag_14__CT1__flash_duration`: contribution `-0.011060`
- `lag_03__T2__flash_duration`: contribution `-0.010241`
- `lag_14__CT_flash_duration_sum`: contribution `-0.008754`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005281`

### tick `12901`, seconds `14.00`, LSTM delta `-0.0493`

Top all feature movements:
- `lag_09__CT_place_HOLE`: contribution `+0.029343`
- `lag_01__T_shots_fired_sum`: contribution `-0.009564`
- `lag_14__T_flashed_players`: contribution `+0.007041`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004697`
- `lag_04__T_flashed_players`: contribution `-0.004502`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.003978`
- `lag_15__CT1__flash_duration`: contribution `-0.003132`
- `lag_04__T2__flash_duration`: contribution `-0.002573`
- `lag_15__CT_flash_duration_sum`: contribution `-0.002430`

### tick `13925`, seconds `30.00`, LSTM delta `+0.0448`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.014767`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `+0.004491`
- `lag_06__T_shots_fired_sum`: contribution `+0.003225`
- `lag_11__CT2__is_scoped`: contribution `+0.002130`
- `lag_04__CT2__duck_amount`: contribution `+0.001953`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `+0.000918`
- `lag_03__CT_active_infernos`: contribution `+0.000881`

### tick `15269`, seconds `51.00`, LSTM delta `-0.0430`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.006835`
- `lag_00__T_kills_last_3s`: contribution `-0.006329`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004751`
- `lag_00__kill_diff_last_3s`: contribution `-0.003532`
- `lag_00__T_damage_last_5s`: contribution `-0.002942`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14181`, seconds `34.00`, LSTM delta `-0.0347`

Top all feature movements:
- `lag_02__CT_place_TUNNELSTAIRS`: contribution `-0.008813`
- `lag_00__T_shots_fired_sum`: contribution `-0.008202`
- `lag_08__CT_place_UPPERTUNNEL`: contribution `-0.003037`
- `lag_06__CT_place_UNDERA`: contribution `+0.002619`
- `lag_14__T_shots_fired_sum`: contribution `+0.002085`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.001386`
