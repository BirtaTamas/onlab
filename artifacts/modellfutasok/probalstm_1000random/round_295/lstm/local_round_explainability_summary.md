# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `69832`, seconds `38.00`, LSTM `0.8943`, delta `+0.1235`
- tick `68936`, seconds `24.00`, LSTM `0.6588`, delta `+0.1002`
- tick `70504`, seconds `48.50`, LSTM `0.9657`, delta `+0.0628`
- tick `68904`, seconds `23.50`, LSTM `0.5585`, delta `+0.0466`
- tick `70120`, seconds `42.50`, LSTM `0.9524`, delta `+0.0430`
- tick `69000`, seconds `25.00`, LSTM `0.6504`, delta `-0.0427`
- tick `69672`, seconds `35.50`, LSTM `0.6916`, delta `+0.0415`
- tick `68968`, seconds `24.50`, LSTM `0.6931`, delta `+0.0344`
- tick `70440`, seconds `47.50`, LSTM `0.9191`, delta `-0.0335`
- tick `68104`, seconds `11.00`, LSTM `0.5375`, delta `+0.0292`

## Top 15 local ridge features

- `lag_02__T4__flash_duration`: coefficient `0.001532`, |coef| `0.001532`
- `lag_05__T2__flash_duration`: coefficient `0.001302`, |coef| `0.001302`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001243`, |coef| `0.001243`
- `lag_00__CT_kills_last_3s`: coefficient `0.001220`, |coef| `0.001220`
- `lag_00__kill_diff_last_3s`: coefficient `0.001053`, |coef| `0.001053`
- `lag_05__T_flashed_players`: coefficient `0.001009`, |coef| `0.001009`
- `lag_05__T_flash_duration_sum`: coefficient `0.000971`, |coef| `0.000971`
- `lag_12__T_place_BALCONY`: coefficient `-0.000873`, |coef| `0.000873`
- `lag_03__T4__flash_duration`: coefficient `0.000851`, |coef| `0.000851`
- `lag_00__CT_damage_last_5s`: coefficient `0.000846`, |coef| `0.000846`
- `lag_05__T1__flash_duration`: coefficient `0.000827`, |coef| `0.000827`
- `lag_00__CT_place_BANANA`: coefficient `0.000794`, |coef| `0.000794`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000786`, |coef| `0.000786`
- `lag_14__CT2__duck_amount`: coefficient `-0.000767`, |coef| `0.000767`
- `lag_05__CT3__is_scoped`: coefficient `0.000751`, |coef| `0.000751`

## Top 10 utility ridge features

- `lag_02__T4__flash_duration`: coefficient `0.001532` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001302` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.000971` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000851` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000827` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000786` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000738` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000709` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.000708` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000633` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001243` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001220` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001053` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `0.001009` (raises CT win probability)
- `lag_12__T_place_BALCONY`: coefficient `-0.000873` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000846` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.000794` (raises CT win probability)
- `lag_14__CT2__duck_amount`: coefficient `-0.000767` (lowers CT win probability)
- `lag_05__CT3__is_scoped`: coefficient `0.000751` (raises CT win probability)
- `lag_12__CT_place_BANANA`: coefficient `0.000749` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `69832`, seconds `38.00`, LSTM delta `+0.1235`

Top all feature movements:
- `lag_05__T2__flash_duration`: contribution `+0.009474`
- `lag_02__T4__flash_duration`: contribution `+0.009472`
- `lag_05__T_flashed_players`: contribution `+0.005841`
- `lag_05__T_flash_duration_sum`: contribution `+0.005579`
- `lag_05__T1__flash_duration`: contribution `+0.004028`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.009474`
- `lag_02__T4__flash_duration`: contribution `+0.009472`
- `lag_05__T_flash_duration_sum`: contribution `+0.005579`
- `lag_05__T1__flash_duration`: contribution `+0.004028`
- `lag_07__CT_B_site_active_infernos`: contribution `+0.001642`

### tick `68936`, seconds `24.00`, LSTM delta `+0.1002`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004318`
- `lag_13__CT5__flash_duration`: contribution `+0.004125`
- `lag_03__T1__is_scoped`: contribution `+0.003722`
- `lag_07__T1__is_scoped`: contribution `+0.003592`
- `lag_00__CT_kills_last_3s`: contribution `+0.003523`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.004125`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001368`
- `lag_00__T3__utility_total`: contribution `+0.001222`

### tick `70504`, seconds `48.50`, LSTM delta `+0.0628`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004318`
- `lag_00__CT_kills_last_3s`: contribution `+0.003523`
- `lag_00__kill_diff_last_3s`: contribution `+0.002535`
- `lag_12__CT_place_BANANA`: contribution `+0.002219`
- `lag_15__T1__is_scoped`: contribution `+0.002202`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.001890`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.001245`
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.001031`

### tick `68904`, seconds `23.50`, LSTM delta `+0.0466`

Top all feature movements:
- `lag_12__CT5__flash_duration`: contribution `+0.004038`
- `lag_09__CT3__is_scoped`: contribution `+0.003091`
- `lag_12__CT3__is_scoped`: contribution `+0.002687`
- `lag_02__T1__is_scoped`: contribution `+0.002485`
- `lag_00__T5__duck_amount`: contribution `+0.002036`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.004038`
- `lag_08__T_utility_damage_last_5s`: contribution `+0.001031`
- `lag_08__T_active_infernos`: contribution `+0.000984`

### tick `70120`, seconds `42.50`, LSTM delta `+0.0430`

Top all feature movements:
- `lag_08__T4__flash_duration`: contribution `+0.004677`
- `lag_11__T4__flash_duration`: contribution `+0.003875`
- `lag_03__T1__is_scoped`: contribution `-0.003722`
- `lag_00__CT_kills_last_3s`: contribution `+0.003523`
- `lag_00__kill_diff_last_3s`: contribution `+0.002535`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.004677`
- `lag_11__T4__flash_duration`: contribution `+0.003875`
- `lag_09__T4__flash_duration`: contribution `-0.002277`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.002194`
- `lag_00__T2__flash_duration`: contribution `-0.002168`
