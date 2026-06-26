# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `5`

## Largest probability jumps

- tick `43730`, seconds `86.00`, LSTM `0.0531`, delta `-0.1313`
- tick `40754`, seconds `39.50`, LSTM `0.6991`, delta `+0.1266`
- tick `41810`, seconds `56.00`, LSTM `0.3823`, delta `-0.1227`
- tick `41234`, seconds `47.00`, LSTM `0.5044`, delta `-0.1114`
- tick `43858`, seconds `88.00`, LSTM `0.0235`, delta `-0.0844`
- tick `42194`, seconds `62.00`, LSTM `0.3820`, delta `+0.0751`
- tick `43794`, seconds `87.00`, LSTM `0.1116`, delta `+0.0728`
- tick `38482`, seconds `4.00`, LSTM `0.5775`, delta `-0.0713`
- tick `42354`, seconds `64.50`, LSTM `0.3966`, delta `+0.0653`
- tick `42706`, seconds `70.00`, LSTM `0.2438`, delta `-0.0632`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001789`, |coef| `0.001789`
- `lag_15__T_place_CONSTRUCTION`: coefficient `-0.001696`, |coef| `0.001696`
- `lag_00__T_kills_last_3s`: coefficient `-0.001627`, |coef| `0.001627`
- `lag_00__damage_diff_last_5s`: coefficient `0.001538`, |coef| `0.001538`
- `lag_00__CT_place_BRIDGE`: coefficient `0.001514`, |coef| `0.001514`
- `lag_11__CT_shots_fired_sum`: coefficient `0.001153`, |coef| `0.001153`
- `lag_00__T_damage_last_5s`: coefficient `-0.001100`, |coef| `0.001100`
- `lag_11__CT1__shots_fired`: coefficient `0.001070`, |coef| `0.001070`
- `lag_00__CT4__is_scoped`: coefficient `-0.001050`, |coef| `0.001050`
- `lag_15__CT3__duck_amount`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_01__CT_place_WATER`: coefficient `0.001011`, |coef| `0.001011`
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000978`, |coef| `0.000978`
- `lag_00__CT3__alive`: coefficient `0.000961`, |coef| `0.000961`
- `lag_09__T_flashed_players`: coefficient `0.000957`, |coef| `0.000957`
- `lag_10__T_flashed_players`: coefficient `0.000957`, |coef| `0.000957`

## Top 10 utility ridge features

- `lag_12__T_utility_damage_last_5s`: coefficient `0.000978` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.000862` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.000822` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `0.000779` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.000769` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000739` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.000729` (raises CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.000714` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `0.000702` (raises CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.000700` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001789` (raises CT win probability)
- `lag_15__T_place_CONSTRUCTION`: coefficient `-0.001696` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001627` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001538` (raises CT win probability)
- `lag_00__CT_place_BRIDGE`: coefficient `0.001514` (raises CT win probability)
- `lag_11__CT_shots_fired_sum`: coefficient `0.001153` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001100` (lowers CT win probability)
- `lag_11__CT1__shots_fired`: coefficient `0.001070` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001050` (lowers CT win probability)
- `lag_15__CT3__duck_amount`: coefficient `-0.001033` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `43730`, seconds `86.00`, LSTM delta `-0.1313`

Top all feature movements:
- `lag_15__T_place_CONSTRUCTION`: contribution `-0.021080`
- `lag_01__CT_place_WATER`: contribution `-0.006142`
- `lag_00__T_kills_last_3s`: contribution `-0.005155`
- `lag_15__T_place_WATER`: contribution `-0.004438`
- `lag_00__kill_diff_last_3s`: contribution `-0.004307`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `40754`, seconds `39.50`, LSTM delta `+0.1266`

Top all feature movements:
- `lag_09__T_flashed_players`: contribution `+0.007388`
- `lag_00__CT_place_BACKOFA`: contribution `+0.007211`
- `lag_08__T2__flash_duration`: contribution `+0.005313`
- `lag_09__T1__flash_duration`: contribution `+0.004895`
- `lag_09__T_flash_duration_sum`: contribution `+0.004814`

Top utility-only movements:
- `lag_08__T2__flash_duration`: contribution `+0.005313`
- `lag_09__T1__flash_duration`: contribution `+0.004895`
- `lag_09__T_flash_duration_sum`: contribution `+0.004814`
- `lag_03__T4__flash_duration`: contribution `+0.002222`

### tick `41810`, seconds `56.00`, LSTM delta `-0.1227`

Top all feature movements:
- `lag_11__CT_shots_fired_sum`: contribution `-0.011214`
- `lag_11__CT_place_CONSTRUCTION`: contribution `-0.010734`
- `lag_12__CT_place_CONSTRUCTION`: contribution `-0.010058`
- `lag_11__CT1__shots_fired`: contribution `-0.007918`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.007542`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.007542`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.002380`

### tick `41234`, seconds `47.00`, LSTM delta `-0.1114`

Top all feature movements:
- `lag_08__CT_place_CONSTRUCTION`: contribution `-0.008948`
- `lag_00__CT2__flash_duration`: contribution `-0.005790`
- `lag_00__T_kills_last_3s`: contribution `-0.005155`
- `lag_15__T_place_WATER`: contribution `-0.004438`
- `lag_00__kill_diff_last_3s`: contribution `-0.004307`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.005790`
- `lag_14__CT2__flash_duration`: contribution `-0.002855`
- `lag_12__T5__flash_duration`: contribution `-0.002408`
- `lag_09__T2__flash_duration`: contribution `-0.002051`
- `lag_05__T4__flash_duration`: contribution `-0.001959`

### tick `43858`, seconds `88.00`, LSTM delta `-0.0844`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005155`
- `lag_00__kill_diff_last_3s`: contribution `-0.004307`
- `lag_02__T_place_WATER`: contribution `-0.003779`
- `lag_05__CT_place_WATER`: contribution `+0.003498`
- `lag_12__T_place_ALLEY`: contribution `-0.002589`

Top utility-only movements:
- No utility movement among the top local contributors.
