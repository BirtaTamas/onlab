# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `40149`, seconds `113.00`, LSTM `0.2992`, delta `+0.2057`
- tick `35381`, seconds `38.50`, LSTM `0.3428`, delta `-0.1719`
- tick `35541`, seconds `41.00`, LSTM `0.2568`, delta `-0.0876`
- tick `37109`, seconds `65.50`, LSTM `0.4330`, delta `+0.0649`
- tick `38325`, seconds `84.50`, LSTM `0.1305`, delta `-0.0589`
- tick `40565`, seconds `119.50`, LSTM `0.4912`, delta `+0.0587`
- tick `38005`, seconds `79.50`, LSTM `0.2789`, delta `-0.0526`
- tick `40437`, seconds `117.50`, LSTM `0.4201`, delta `+0.0508`
- tick `36981`, seconds `63.50`, LSTM `0.3400`, delta `+0.0490`
- tick `38037`, seconds `80.00`, LSTM `0.2355`, delta `-0.0434`

## Top 15 local ridge features

- `lag_00__T4__alive`: coefficient `-0.002495`, |coef| `0.002495`
- `lag_00__T4__hp`: coefficient `-0.002447`, |coef| `0.002447`
- `lag_00__T4__armor`: coefficient `-0.002280`, |coef| `0.002280`
- `lag_00__damage_diff_last_5s`: coefficient `0.002139`, |coef| `0.002139`
- `lag_00__T4__has_helmet`: coefficient `-0.002119`, |coef| `0.002119`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002090`, |coef| `0.002090`
- `lag_00__T_macro_B`: coefficient `-0.002090`, |coef| `0.002090`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002065`, |coef| `0.002065`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001914`, |coef| `0.001914`
- `lag_14__CT5__is_walking`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_01__T4__duck_amount`: coefficient `0.001706`, |coef| `0.001706`
- `lag_00__CT_damage_last_5s`: coefficient `0.001697`, |coef| `0.001697`
- `lag_00__kill_diff_last_3s`: coefficient `0.001637`, |coef| `0.001637`
- `lag_00__CT_kills_last_3s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_13__CT5__is_walking`: coefficient `-0.001561`, |coef| `0.001561`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000706` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000627` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000574` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000560` (raises CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.000479` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000451` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.000450` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.000441` (lowers CT win probability)
- `lag_11__CT5__molly`: coefficient `0.000437` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000432` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T4__alive`: coefficient `-0.002495` (lowers CT win probability)
- `lag_00__T4__hp`: coefficient `-0.002447` (lowers CT win probability)
- `lag_00__T4__armor`: coefficient `-0.002280` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002139` (raises CT win probability)
- `lag_00__T4__has_helmet`: coefficient `-0.002119` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002090` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002090` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002065` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001914` (raises CT win probability)
- `lag_14__CT5__is_walking`: coefficient `-0.001708` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40149`, seconds `113.00`, LSTM delta `+0.2057`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007173`
- `lag_01__T4__duck_amount`: contribution `+0.006216`
- `lag_00__T4__alive`: contribution `+0.006131`
- `lag_00__T4__hp`: contribution `+0.005900`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005319`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35381`, seconds `38.50`, LSTM delta `-0.1719`

Top all feature movements:
- `lag_02__CT_place_SECRET`: contribution `-0.013072`
- `lag_00__T2__is_scoped`: contribution `+0.010073`
- `lag_00__CT_place_SECRET`: contribution `-0.008822`
- `lag_08__T_place_SQUEAKY`: contribution `-0.007316`
- `lag_01__T_shots_fired_sum`: contribution `-0.006877`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35541`, seconds `41.00`, LSTM delta `-0.0876`

Top all feature movements:
- `lag_14__CT5__is_walking`: contribution `-0.004093`
- `lag_13__T_place_SQUEAKY`: contribution `-0.003947`
- `lag_05__T_place_SECRET`: contribution `-0.003938`
- `lag_05__T2__is_scoped`: contribution `+0.003240`
- `lag_14__CT_place_RAFTERS`: contribution `-0.003094`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37109`, seconds `65.50`, LSTM delta `+0.0649`

Top all feature movements:
- `lag_01__CT_place_LOBBY`: contribution `+0.005959`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005739`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005319`
- `lag_00__CT_kills_last_3s`: contribution `+0.004612`
- `lag_00__kill_diff_last_3s`: contribution `+0.003941`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38325`, seconds `84.50`, LSTM delta `-0.0589`

Top all feature movements:
- `lag_12__CT_place_CONTROL`: contribution `-0.007559`
- `lag_15__CT_place_CONTROL`: contribution `-0.007538`
- `lag_14__T2__is_scoped`: contribution `-0.004934`
- `lag_05__T2__is_scoped`: contribution `-0.003240`
- `lag_12__CT2__is_walking`: contribution `-0.002532`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.000894`
