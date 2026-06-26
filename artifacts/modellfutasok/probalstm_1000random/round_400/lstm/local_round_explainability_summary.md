# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-gamerlegion-vs-inner-circle-bo3-TOF4f6Uhtdi7Vqylk0QEY6/gamerlegion-vs-inner-circle-m1-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `83693`, seconds `78.50`, LSTM `0.7438`, delta `+0.4557`
- tick `81325`, seconds `41.50`, LSTM `0.0792`, delta `-0.1866`
- tick `81133`, seconds `38.50`, LSTM `0.4265`, delta `-0.0843`
- tick `83565`, seconds `76.50`, LSTM `0.1941`, delta `+0.0551`
- tick `81293`, seconds `41.00`, LSTM `0.2658`, delta `-0.0509`
- tick `83853`, seconds `81.00`, LSTM `0.8139`, delta `+0.0471`
- tick `81261`, seconds `40.50`, LSTM `0.3167`, delta `-0.0431`
- tick `83597`, seconds `77.00`, LSTM `0.2344`, delta `+0.0403`
- tick `82509`, seconds `60.00`, LSTM `0.0489`, delta `+0.0399`
- tick `82541`, seconds `60.50`, LSTM `0.0861`, delta `+0.0372`

## Top 15 local ridge features

- `lag_04__CT_defusing_count`: coefficient `0.007430`, |coef| `0.007430`
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.003738`, |coef| `0.003738`
- `lag_10__CT_place_SIDEHALL`: coefficient `-0.003377`, |coef| `0.003377`
- `lag_01__T5__shots_fired`: coefficient `0.003052`, |coef| `0.003052`
- `lag_11__T5__flash_duration`: coefficient `-0.002903`, |coef| `0.002903`
- `lag_04__CT5__duck_amount`: coefficient `0.002895`, |coef| `0.002895`
- `lag_00__T5__shots_fired`: coefficient `0.002775`, |coef| `0.002775`
- `lag_06__CT_A_site_active_infernos`: coefficient `0.002707`, |coef| `0.002707`
- `lag_04__T2__is_scoped`: coefficient `-0.002572`, |coef| `0.002572`
- `lag_00__kill_diff_last_3s`: coefficient `0.002429`, |coef| `0.002429`
- `lag_03__CT_defusing_count`: coefficient `0.002270`, |coef| `0.002270`
- `lag_01__CT2__duck_amount`: coefficient `0.002252`, |coef| `0.002252`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002194`, |coef| `0.002194`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002188`, |coef| `0.002188`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002144`, |coef| `0.002144`

## Top 10 utility ridge features

- `lag_11__T5__flash_duration`: coefficient `-0.002903` (lowers CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.002707` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.001967` (raises CT win probability)
- `lag_08__CT4__molly`: coefficient `-0.001880` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001783` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001434` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001251` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001223` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001128` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.001044` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_defusing_count`: coefficient `0.007430` (raises CT win probability)
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.003738` (lowers CT win probability)
- `lag_10__CT_place_SIDEHALL`: coefficient `-0.003377` (lowers CT win probability)
- `lag_01__T5__shots_fired`: coefficient `0.003052` (raises CT win probability)
- `lag_04__CT5__duck_amount`: coefficient `0.002895` (raises CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.002775` (raises CT win probability)
- `lag_04__T2__is_scoped`: coefficient `-0.002572` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002429` (raises CT win probability)
- `lag_03__CT_defusing_count`: coefficient `0.002270` (raises CT win probability)
- `lag_01__CT2__duck_amount`: coefficient `0.002252` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `83693`, seconds `78.50`, LSTM delta `+0.4557`

Top all feature movements:
- `lag_04__CT_defusing_count`: contribution `+0.072025`
- `lag_04__T2__is_scoped`: contribution `+0.022671`
- `lag_00__CT_place_SIDEHALL`: contribution `+0.015990`
- `lag_10__CT_place_SIDEHALL`: contribution `+0.014445`
- `lag_11__T5__flash_duration`: contribution `+0.012086`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.012086`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.009553`
- `lag_08__CT4__molly`: contribution `+0.004632`
- `lag_06__CT_active_infernos`: contribution `+0.004533`

### tick `81325`, seconds `41.50`, LSTM delta `-0.1866`

Top all feature movements:
- `lag_15__T2__is_scoped`: contribution `-0.011210`
- `lag_10__T2__is_scoped`: contribution `-0.007957`
- `lag_05__CT_place_SIDEHALL`: contribution `-0.007837`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005853`
- `lag_00__kill_diff_last_3s`: contribution `-0.005846`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.005853`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.003543`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.003275`
- `lag_01__T_A_site_active_infernos`: contribution `-0.003073`
- `lag_06__T_A_site_active_infernos`: contribution `-0.003035`

### tick `81133`, seconds `38.50`, LSTM delta `-0.0843`

Top all feature movements:
- `lag_04__T2__is_scoped`: contribution `+0.022671`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.009861`
- `lag_09__T2__is_scoped`: contribution `-0.006086`
- `lag_13__T_place_TUNNEL`: contribution `-0.005102`
- `lag_03__T1__duck_amount`: contribution `-0.004509`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.003725`
- `lag_00__T_active_infernos`: contribution `-0.001424`

### tick `83565`, seconds `76.50`, LSTM delta `+0.0551`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.013716`
- `lag_00__T2__is_scoped`: contribution `+0.007181`
- `lag_13__CT4__duck_amount`: contribution `-0.005992`
- `lag_06__CT_place_SIDEHALL`: contribution `+0.004814`
- `lag_07__T5__flash_duration`: contribution `+0.003066`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `+0.003066`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001830`
- `lag_14__T5__flash_duration`: contribution `+0.001232`
- `lag_02__CT_active_infernos`: contribution `+0.000977`
- `lag_04__CT4__molly`: contribution `+0.000918`

### tick `81293`, seconds `41.00`, LSTM delta `-0.0509`

Top all feature movements:
- `lag_14__T2__is_scoped`: contribution `-0.006653`
- `lag_09__T2__is_scoped`: contribution `+0.006086`
- `lag_04__CT_place_SIDEHALL`: contribution `-0.005357`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.004503`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004072`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.004503`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004072`
- `lag_00__T_A_site_active_infernos`: contribution `-0.003725`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.002073`
- `lag_05__T_A_site_active_infernos`: contribution `-0.001950`
