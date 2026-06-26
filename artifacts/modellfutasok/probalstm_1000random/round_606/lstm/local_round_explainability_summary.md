# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `59018`, seconds `53.50`, LSTM `0.2863`, delta `-0.3455`
- tick `62698`, seconds `111.00`, LSTM `0.6426`, delta `+0.3147`
- tick `62538`, seconds `108.50`, LSTM `0.5919`, delta `+0.2620`
- tick `62794`, seconds `112.50`, LSTM `0.8522`, delta `+0.2229`
- tick `62634`, seconds `110.00`, LSTM `0.3563`, delta `-0.1679`
- tick `59498`, seconds `61.00`, LSTM `0.5816`, delta `+0.1312`
- tick `59050`, seconds `54.00`, LSTM `0.1829`, delta `-0.1034`
- tick `60106`, seconds `70.50`, LSTM `0.5535`, delta `+0.0782`
- tick `59402`, seconds `59.50`, LSTM `0.3780`, delta `+0.0773`
- tick `62730`, seconds `111.50`, LSTM `0.5695`, delta `-0.0731`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003440`, |coef| `0.003440`
- `lag_00__kill_diff_last_3s`: coefficient `0.003358`, |coef| `0.003358`
- `lag_00__CT_defusing_count`: coefficient `0.003058`, |coef| `0.003058`
- `lag_00__damage_diff_last_5s`: coefficient `0.002793`, |coef| `0.002793`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002765`, |coef| `0.002765`
- `lag_00__T_kills_last_3s`: coefficient `-0.002748`, |coef| `0.002748`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.002459`, |coef| `0.002459`
- `lag_01__T1__flash_duration`: coefficient `-0.002402`, |coef| `0.002402`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002361`, |coef| `0.002361`
- `lag_04__CT3__shots_fired`: coefficient `-0.002140`, |coef| `0.002140`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002078`, |coef| `0.002078`
- `lag_01__T2__flash_duration`: coefficient `-0.002046`, |coef| `0.002046`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001917`, |coef| `0.001917`
- `lag_08__T5__flash_duration`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_08__T3__flash_duration`: coefficient `-0.001887`, |coef| `0.001887`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `-0.002402` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002078` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.002046` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001917` (lowers CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.001889` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.001887` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.001790` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.001540` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001345` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.001327` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.003440` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003358` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003058` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002793` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.002765` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002748` (lowers CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `-0.002459` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002361` (raises CT win probability)
- `lag_04__CT3__shots_fired`: coefficient `-0.002140` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001815` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `59018`, seconds `53.50`, LSTM delta `-0.3455`

Top all feature movements:
- `lag_01__T1__flash_duration`: contribution `-0.017589`
- `lag_00__T_shots_fired_sum`: contribution `-0.012894`
- `lag_01__T2__flash_duration`: contribution `-0.012163`
- `lag_08__T3__flash_duration`: contribution `-0.011239`
- `lag_08__T_flash_duration_sum`: contribution `-0.010403`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.017589`
- `lag_01__T2__flash_duration`: contribution `-0.012163`
- `lag_08__T3__flash_duration`: contribution `-0.011239`
- `lag_08__T_flash_duration_sum`: contribution `-0.010403`
- `lag_08__T5__flash_duration`: contribution `-0.009889`

### tick `62698`, seconds `111.00`, LSTM delta `+0.3147`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.027332`
- `lag_00__T_shots_fired_sum`: contribution `+0.018052`
- `lag_04__CT3__shots_fired`: contribution `+0.014306`
- `lag_00__T_flash_alpha_mean`: contribution `+0.012610`
- `lag_00__CT_duck_amount_mean`: contribution `+0.010523`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.012610`

### tick `62538`, seconds `108.50`, LSTM delta `+0.2620`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.014763`
- `lag_00__kill_diff_last_3s`: contribution `+0.008084`
- `lag_02__CT_shots_fired_sum`: contribution `+0.007685`
- `lag_00__CT3__shots_fired`: contribution `+0.005363`
- `lag_06__CT3__duck_amount`: contribution `+0.004680`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62794`, seconds `112.50`, LSTM delta `+0.2229`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.029640`
- `lag_02__CT_shots_fired_sum`: contribution `-0.013448`
- `lag_03__T_flash_alpha_mean`: contribution `+0.011633`
- `lag_03__CT_duck_amount_mean`: contribution `+0.007660`
- `lag_07__CT_shots_fired_sum`: contribution `+0.006839`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.011633`

### tick `62634`, seconds `110.00`, LSTM delta `-0.1679`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.030739`
- `lag_00__T_kills_last_3s`: contribution `-0.008706`
- `lag_02__CT3__shots_fired`: contribution `-0.008305`
- `lag_00__kill_diff_last_3s`: contribution `-0.008084`
- `lag_04__CT3__shots_fired`: contribution `-0.005502`

Top utility-only movements:
- No utility movement among the top local contributors.
