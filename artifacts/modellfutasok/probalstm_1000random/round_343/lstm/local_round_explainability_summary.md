# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `48657`, seconds `111.00`, LSTM `0.1447`, delta `-0.3215`
- tick `48017`, seconds `101.00`, LSTM `0.4009`, delta `+0.2777`
- tick `43281`, seconds `27.00`, LSTM `0.1521`, delta `-0.2768`
- tick `44305`, seconds `43.00`, LSTM `0.3740`, delta `+0.2244`
- tick `44337`, seconds `43.50`, LSTM `0.1512`, delta `-0.2228`
- tick `43121`, seconds `24.50`, LSTM `0.2861`, delta `+0.1357`
- tick `47889`, seconds `99.00`, LSTM `0.1856`, delta `-0.1134`
- tick `48881`, seconds `114.50`, LSTM `0.0511`, delta `-0.0918`
- tick `48081`, seconds `102.00`, LSTM `0.5417`, delta `+0.0878`
- tick `43185`, seconds `25.50`, LSTM `0.4193`, delta `+0.0798`

## Top 15 local ridge features

- `lag_15__T_place_STAIRS`: coefficient `0.005514`, |coef| `0.005514`
- `lag_13__T_place_STAIRS`: coefficient `0.005371`, |coef| `0.005371`
- `lag_00__kill_diff_last_3s`: coefficient `0.004762`, |coef| `0.004762`
- `lag_00__damage_diff_last_5s`: coefficient `0.004364`, |coef| `0.004364`
- `lag_12__T_place_JUNGLE`: coefficient `-0.003281`, |coef| `0.003281`
- `lag_00__T_kills_last_3s`: coefficient `-0.003176`, |coef| `0.003176`
- `lag_10__T_place_JUNGLE`: coefficient `0.003032`, |coef| `0.003032`
- `lag_02__T_place_JUNGLE`: coefficient `0.003016`, |coef| `0.003016`
- `lag_09__T_place_STAIRS`: coefficient `-0.002977`, |coef| `0.002977`
- `lag_00__CT_kills_last_3s`: coefficient `0.002818`, |coef| `0.002818`
- `lag_00__CT_damage_last_5s`: coefficient `0.002649`, |coef| `0.002649`
- `lag_02__CT3__is_scoped`: coefficient `0.002600`, |coef| `0.002600`
- `lag_05__T_place_STAIRS`: coefficient `0.002509`, |coef| `0.002509`
- `lag_00__CT1__duck_amount`: coefficient `0.002469`, |coef| `0.002469`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.002426`, |coef| `0.002426`

## Top 10 utility ridge features

- `lag_05__T_A_site_active_infernos`: coefficient `0.002180` (raises CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.001802` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.001668` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.001637` (raises CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.001529` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001441` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.001232` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001225` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `0.001165` (raises CT win probability)
- `lag_05__T1__molly`: coefficient `-0.001153` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_STAIRS`: coefficient `0.005514` (raises CT win probability)
- `lag_13__T_place_STAIRS`: coefficient `0.005371` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004762` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004364` (raises CT win probability)
- `lag_12__T_place_JUNGLE`: coefficient `-0.003281` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003176` (lowers CT win probability)
- `lag_10__T_place_JUNGLE`: coefficient `0.003032` (raises CT win probability)
- `lag_02__T_place_JUNGLE`: coefficient `0.003016` (raises CT win probability)
- `lag_09__T_place_STAIRS`: coefficient `-0.002977` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002818` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `48657`, seconds `111.00`, LSTM delta `-0.3215`

Top all feature movements:
- `lag_15__T_place_STAIRS`: contribution `-0.105568`
- `lag_12__T_place_JUNGLE`: contribution `-0.042497`
- `lag_02__T_place_JUNGLE`: contribution `-0.039071`
- `lag_02__CT3__is_scoped`: contribution `-0.011827`
- `lag_00__kill_diff_last_3s`: contribution `-0.011462`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `-0.003668`

### tick `48017`, seconds `101.00`, LSTM delta `+0.2777`

Top all feature movements:
- `lag_13__T_place_STAIRS`: contribution `+0.102830`
- `lag_00__kill_diff_last_3s`: contribution `+0.011462`
- `lag_00__CT_place_TRUCK`: contribution `+0.009880`
- `lag_00__CT3__is_scoped`: contribution `+0.009373`
- `lag_00__damage_diff_last_5s`: contribution `+0.009057`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.006490`
- `lag_04__T_A_site_active_infernos`: contribution `+0.004964`
- `lag_05__T_active_infernos`: contribution `+0.003185`

### tick `43281`, seconds `27.00`, LSTM delta `-0.2768`

Top all feature movements:
- `lag_04__CT_place_UNDERPASS`: contribution `-0.012304`
- `lag_11__CT_place_UNDERPASS`: contribution `-0.011741`
- `lag_00__kill_diff_last_3s`: contribution `-0.011462`
- `lag_00__T_kills_last_3s`: contribution `-0.010061`
- `lag_01__T_shots_fired_sum`: contribution `-0.009489`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.009363`
- `lag_03__CT4__flash_duration`: contribution `-0.008505`

### tick `44305`, seconds `43.00`, LSTM delta `+0.2244`

Top all feature movements:
- `lag_02__CT_place_STAIRS`: contribution `+0.014235`
- `lag_12__CT_place_STAIRS`: contribution `+0.013778`
- `lag_00__kill_diff_last_3s`: contribution `+0.011462`
- `lag_13__CT_place_TRUCK`: contribution `+0.011319`
- `lag_00__damage_diff_last_5s`: contribution `+0.009845`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44337`, seconds `43.50`, LSTM delta `-0.2228`

Top all feature movements:
- `lag_03__CT_place_STAIRS`: contribution `-0.017360`
- `lag_13__CT_place_STAIRS`: contribution `-0.015204`
- `lag_00__kill_diff_last_3s`: contribution `-0.011462`
- `lag_14__CT_place_TRUCK`: contribution `-0.010485`
- `lag_00__T_kills_last_3s`: contribution `-0.010061`

Top utility-only movements:
- No utility movement among the top local contributors.
