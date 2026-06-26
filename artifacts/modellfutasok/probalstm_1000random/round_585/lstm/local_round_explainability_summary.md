# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `70452`, seconds `77.00`, LSTM `0.6437`, delta `-0.2046`
- tick `70356`, seconds `75.50`, LSTM `0.7675`, delta `+0.2003`
- tick `70900`, seconds `84.00`, LSTM `0.4743`, delta `+0.1716`
- tick `71028`, seconds `86.00`, LSTM `0.7067`, delta `+0.1449`
- tick `70804`, seconds `82.50`, LSTM `0.4501`, delta `-0.1244`
- tick `70836`, seconds `83.00`, LSTM `0.3483`, delta `-0.1018`
- tick `67988`, seconds `38.50`, LSTM `0.5909`, delta `+0.0882`
- tick `70932`, seconds `84.50`, LSTM `0.5420`, delta `+0.0677`
- tick `67956`, seconds `38.00`, LSTM `0.5027`, delta `-0.0635`
- tick `70868`, seconds `83.50`, LSTM `0.3027`, delta `-0.0456`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002861`, |coef| `0.002861`
- `lag_00__CT_kills_last_3s`: coefficient `0.002122`, |coef| `0.002122`
- `lag_00__damage_diff_last_5s`: coefficient `0.002007`, |coef| `0.002007`
- `lag_06__T_place_WATER`: coefficient `0.001661`, |coef| `0.001661`
- `lag_14__CT_place_HOUSE`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_00__T_kills_last_3s`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_08__T1__duck_amount`: coefficient `0.001409`, |coef| `0.001409`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `0.001341`, |coef| `0.001341`
- `lag_10__CT1__duck_amount`: coefficient `0.001337`, |coef| `0.001337`
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_00__CT_damage_last_5s`: coefficient `0.001259`, |coef| `0.001259`
- `lag_08__CT2__is_walking`: coefficient `-0.001254`, |coef| `0.001254`
- `lag_15__CT1__duck_amount`: coefficient `0.001228`, |coef| `0.001228`
- `lag_14__CT5__is_scoped`: coefficient `0.001198`, |coef| `0.001198`
- `lag_13__T3__duck_amount`: coefficient `-0.001197`, |coef| `0.001197`

## Top 10 utility ridge features

- `lag_13__CT_B_site_active_smokes`: coefficient `0.000721` (raises CT win probability)
- `lag_03__T5__smoke`: coefficient `-0.000706` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000584` (raises CT win probability)
- `lag_06__T1__molly`: coefficient `-0.000532` (lowers CT win probability)
- `lag_15__CT_A_site_active_smokes`: coefficient `-0.000521` (lowers CT win probability)
- `lag_13__CT_active_smokes`: coefficient `0.000503` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `0.000499` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `-0.000484` (lowers CT win probability)
- `lag_04__T5__smoke`: coefficient `-0.000478` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000475` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002861` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002122` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002007` (raises CT win probability)
- `lag_06__T_place_WATER`: coefficient `0.001661` (raises CT win probability)
- `lag_14__CT_place_HOUSE`: coefficient `-0.001486` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001437` (lowers CT win probability)
- `lag_08__T1__duck_amount`: coefficient `0.001409` (raises CT win probability)
- `lag_02__T_place_SIDEENTRANCE`: coefficient `0.001341` (raises CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `0.001337` (raises CT win probability)
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.001329` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `70452`, seconds `77.00`, LSTM delta `-0.2046`

Top all feature movements:
- `lag_06__T_place_WATER`: contribution `-0.009479`
- `lag_00__kill_diff_last_3s`: contribution `-0.006887`
- `lag_10__T_place_SIDEENTRANCE`: contribution `-0.006485`
- `lag_09__T_place_TUNNEL`: contribution `-0.006410`
- `lag_08__T1__duck_amount`: contribution `-0.005517`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70356`, seconds `75.50`, LSTM delta `+0.2003`

Top all feature movements:
- `lag_06__T_place_WATER`: contribution `+0.009479`
- `lag_00__kill_diff_last_3s`: contribution `+0.006887`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.006542`
- `lag_11__T_place_TUNNEL`: contribution `+0.006400`
- `lag_00__CT_kills_last_3s`: contribution `+0.006127`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70900`, seconds `84.00`, LSTM delta `+0.1716`

Top all feature movements:
- `lag_15__CT1__duck_amount`: contribution `+0.004684`
- `lag_13__T3__duck_amount`: contribution `+0.004512`
- `lag_14__CT5__is_scoped`: contribution `+0.004286`
- `lag_00__damage_diff_last_5s`: contribution `+0.004029`
- `lag_10__CT_place_HOUSE`: contribution `+0.003424`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71028`, seconds `86.00`, LSTM delta `+0.1449`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.006887`
- `lag_00__CT_kills_last_3s`: contribution `+0.006127`
- `lag_02__T_bomb_zone_count`: contribution `+0.006086`
- `lag_14__CT_place_HOUSE`: contribution `+0.005250`
- `lag_01__T_shots_fired_sum`: contribution `+0.003791`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70804`, seconds `82.50`, LSTM delta `-0.1244`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.006887`
- `lag_10__CT1__duck_amount`: contribution `-0.004631`
- `lag_00__T_kills_last_3s`: contribution `-0.004554`
- `lag_14__CT5__is_scoped`: contribution `-0.004286`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.003950`

Top utility-only movements:
- No utility movement among the top local contributors.
