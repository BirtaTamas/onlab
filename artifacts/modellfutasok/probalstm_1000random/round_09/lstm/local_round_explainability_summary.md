# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `48932`, seconds `54.00`, LSTM `0.2290`, delta `+0.1599`
- tick `48996`, seconds `55.00`, LSTM `0.0809`, delta `-0.1493`
- tick `49284`, seconds `59.50`, LSTM `0.0412`, delta `-0.0892`
- tick `45508`, seconds `0.50`, LSTM `0.1785`, delta `-0.0829`
- tick `48676`, seconds `50.00`, LSTM `0.1804`, delta `-0.0601`
- tick `48612`, seconds `49.00`, LSTM `0.2351`, delta `-0.0567`
- tick `48516`, seconds `47.50`, LSTM `0.2411`, delta `-0.0562`
- tick `46436`, seconds `15.00`, LSTM `0.2634`, delta `+0.0478`
- tick `48548`, seconds `48.00`, LSTM `0.2825`, delta `+0.0414`
- tick `48452`, seconds `46.50`, LSTM `0.2891`, delta `-0.0386`

## Top 15 local ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003701`, |coef| `0.003701`
- `lag_12__T_place_SIDEENTRANCE`: coefficient `-0.002465`, |coef| `0.002465`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001678`, |coef| `0.001678`
- `lag_06__T3__is_walking`: coefficient `0.001464`, |coef| `0.001464`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001400`, |coef| `0.001400`
- `lag_00__kill_diff_last_3s`: coefficient `0.001221`, |coef| `0.001221`
- `lag_00__CT_burning_players`: coefficient `0.001206`, |coef| `0.001206`
- `lag_08__T_B_site_active_infernos`: coefficient `-0.001203`, |coef| `0.001203`
- `lag_09__T5__is_walking`: coefficient `0.001167`, |coef| `0.001167`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001120`, |coef| `0.001120`
- `lag_12__CT3__is_walking`: coefficient `-0.001117`, |coef| `0.001117`
- `lag_02__T_place_TSIDEUPPER`: coefficient `0.001111`, |coef| `0.001111`
- `lag_15__T_place_TSIDEUPPER`: coefficient `-0.001095`, |coef| `0.001095`
- `lag_03__T_B_site_active_infernos`: coefficient `0.001092`, |coef| `0.001092`

## Top 10 utility ridge features

- `lag_08__T_B_site_active_infernos`: coefficient `-0.001203` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.001092` (raises CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `-0.000819` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000805` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000796` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.000755` (lowers CT win probability)
- `lag_08__T_active_infernos`: coefficient `-0.000750` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000747` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.000706` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.000677` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003701` (lowers CT win probability)
- `lag_12__T_place_SIDEENTRANCE`: coefficient `-0.002465` (lowers CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001678` (lowers CT win probability)
- `lag_06__T3__is_walking`: coefficient `0.001464` (raises CT win probability)
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001400` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001221` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.001206` (raises CT win probability)
- `lag_09__T5__is_walking`: coefficient `0.001167` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001120` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `48932`, seconds `54.00`, LSTM delta `+0.1599`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.018063`
- `lag_12__T_place_SIDEENTRANCE`: contribution `+0.012031`
- `lag_15__T_place_SIDEENTRANCE`: contribution `+0.004381`
- `lag_02__T4__is_scoped`: contribution `+0.003588`
- `lag_10__CT5__duck_amount`: contribution `+0.003566`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.003089`

### tick `48996`, seconds `55.00`, LSTM delta `-0.1493`

Top all feature movements:
- `lag_12__T_place_SIDEENTRANCE`: contribution `-0.012031`
- `lag_00__T_shots_fired_sum`: contribution `-0.005039`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.004787`
- `lag_04__T4__is_scoped`: contribution `-0.004436`
- `lag_08__CT5__duck_amount`: contribution `-0.003414`

Top utility-only movements:
- `lag_08__T_B_site_active_infernos`: contribution `-0.003401`

### tick `49284`, seconds `59.50`, LSTM delta `-0.0892`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.005325`
- `lag_13__T_place_TSIDELOWER`: contribution `-0.003333`
- `lag_08__T2__shots_fired`: contribution `-0.003323`
- `lag_00__CT_burning_players`: contribution `-0.003096`
- `lag_03__T_B_site_active_infernos`: contribution `-0.003089`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.003089`
- `lag_00__CT2__flash_duration`: contribution `-0.002753`
- `lag_10__T_utility_damage_last_5s`: contribution `-0.002573`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002499`
- `lag_14__T_B_site_active_infernos`: contribution `-0.002135`

### tick `45508`, seconds `0.50`, LSTM delta `-0.0829`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.058881`
- `lag_00__CT_velocity_mean`: contribution `-0.002043`
- `lag_00__T_velocity_mean`: contribution `-0.002001`
- `lag_01__T_place_TSPAWN`: contribution `-0.001634`
- `lag_01__T2__has_bomb`: contribution `-0.001439`

Top utility-only movements:
- `lag_01__T4__molly`: contribution `+0.000521`
- `lag_01__flash_inv_diff`: contribution `-0.000408`
- `lag_01__T1__utility_total`: contribution `-0.000371`
- `lag_01__T4__flash`: contribution `-0.000360`
- `lag_01__T5__molly`: contribution `+0.000359`

### tick `48676`, seconds `50.00`, LSTM delta `-0.0601`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.018063`
- `lag_02__T_place_SIDEENTRANCE`: contribution `-0.004787`
- `lag_07__T_place_SIDEENTRANCE`: contribution `-0.004306`
- `lag_13__T_place_TSIDELOWER`: contribution `+0.003333`
- `lag_15__T_place_TSIDELOWER`: contribution `-0.003057`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.001704`
