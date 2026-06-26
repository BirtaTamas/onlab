# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `18`

## Largest probability jumps

- tick `148406`, seconds `82.50`, LSTM `0.4466`, delta `+0.1760`
- tick `148342`, seconds `81.50`, LSTM `0.2532`, delta `-0.1198`
- tick `144854`, seconds `27.00`, LSTM `0.5342`, delta `-0.0686`
- tick `147894`, seconds `74.50`, LSTM `0.4840`, delta `+0.0548`
- tick `147990`, seconds `76.00`, LSTM `0.4504`, delta `-0.0478`
- tick `148438`, seconds `83.00`, LSTM `0.4933`, delta `+0.0467`
- tick `146134`, seconds `47.00`, LSTM `0.5369`, delta `+0.0431`
- tick `149622`, seconds `101.50`, LSTM `0.4254`, delta `-0.0379`
- tick `148118`, seconds `78.00`, LSTM `0.3974`, delta `-0.0355`
- tick `148054`, seconds `77.00`, LSTM `0.4442`, delta `-0.0354`

## Top 15 local ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.001764`, |coef| `0.001764`
- `lag_00__T_place_VENTS`: coefficient `-0.001751`, |coef| `0.001751`
- `lag_01__T_place_OBSERVATION`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_02__CT_place_HUT`: coefficient `-0.001479`, |coef| `0.001479`
- `lag_02__T_place_VENTS`: coefficient `0.001180`, |coef| `0.001180`
- `lag_02__T_place_OBSERVATION`: coefficient `-0.001156`, |coef| `0.001156`
- `lag_12__T2__duck_amount`: coefficient `-0.001129`, |coef| `0.001129`
- `lag_13__T_place_OBSERVATION`: coefficient `0.000951`, |coef| `0.000951`
- `lag_13__T2__duck_amount`: coefficient `-0.000950`, |coef| `0.000950`
- `lag_00__CT2__is_scoped`: coefficient `0.000922`, |coef| `0.000922`
- `lag_15__CT_place_MINI`: coefficient `-0.000900`, |coef| `0.000900`
- `lag_14__CT2__duck_amount`: coefficient `-0.000891`, |coef| `0.000891`
- `lag_00__T_place_GARAGE`: coefficient `-0.000858`, |coef| `0.000858`
- `lag_00__kill_diff_last_3s`: coefficient `0.000822`, |coef| `0.000822`
- `lag_01__CT_place_MINI`: coefficient `0.000785`, |coef| `0.000785`

## Top 10 utility ridge features

- `lag_10__T_A_site_active_infernos`: coefficient `0.000486` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000463` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.000396` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `0.000352` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.000349` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000346` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000331` (raises CT win probability)
- `lag_11__CT1__molly`: coefficient `0.000319` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000314` (lowers CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `0.000312` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.001764` (lowers CT win probability)
- `lag_00__T_place_VENTS`: coefficient `-0.001751` (lowers CT win probability)
- `lag_01__T_place_OBSERVATION`: coefficient `-0.001676` (lowers CT win probability)
- `lag_02__CT_place_HUT`: coefficient `-0.001479` (lowers CT win probability)
- `lag_02__T_place_VENTS`: coefficient `0.001180` (raises CT win probability)
- `lag_02__T_place_OBSERVATION`: coefficient `-0.001156` (lowers CT win probability)
- `lag_12__T2__duck_amount`: coefficient `-0.001129` (lowers CT win probability)
- `lag_13__T_place_OBSERVATION`: coefficient `0.000951` (raises CT win probability)
- `lag_13__T2__duck_amount`: coefficient `-0.000950` (lowers CT win probability)
- `lag_00__CT2__is_scoped`: coefficient `0.000922` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `148406`, seconds `82.50`, LSTM delta `+0.1760`

Top all feature movements:
- `lag_01__T_place_OBSERVATION`: contribution `+0.028386`
- `lag_13__T_place_OBSERVATION`: contribution `+0.016111`
- `lag_02__T_place_VENTS`: contribution `+0.015916`
- `lag_02__CT_place_HUT`: contribution `+0.014421`
- `lag_00__CT2__is_scoped`: contribution `+0.005640`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `148342`, seconds `81.50`, LSTM delta `-0.1198`

Top all feature movements:
- `lag_00__T_place_VENTS`: contribution `-0.023614`
- `lag_11__T_place_OBSERVATION`: contribution `-0.012338`
- `lag_00__CT_place_HUT`: contribution `-0.005723`
- `lag_15__CT_place_MINI`: contribution `-0.005518`
- `lag_14__T_place_DECON`: contribution `-0.005364`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `-0.001448`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001310`

### tick `144854`, seconds `27.00`, LSTM delta `-0.0686`

Top all feature movements:
- `lag_12__T2__duck_amount`: contribution `-0.003478`
- `lag_00__CT_place_HEAVEN`: contribution `-0.003155`
- `lag_06__T_place_SQUEAKY`: contribution `-0.003046`
- `lag_10__CT_place_HEAVEN`: contribution `-0.002788`
- `lag_03__T_place_SQUEAKY`: contribution `-0.002012`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `-0.001206`

### tick `147894`, seconds `74.50`, LSTM delta `+0.0548`

Top all feature movements:
- `lag_01__CT_place_MINI`: contribution `+0.004811`
- `lag_00__T_place_DECON`: contribution `+0.004697`
- `lag_12__T2__duck_amount`: contribution `-0.003233`
- `lag_10__CT_place_ADMIN`: contribution `+0.002554`
- `lag_03__T_bomb_zone_count`: contribution `+0.002552`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `+0.001448`
- `lag_10__T_B_site_active_infernos`: contribution `+0.001310`

### tick `147990`, seconds `76.00`, LSTM delta `-0.0478`

Top all feature movements:
- `lag_00__T_place_OBSERVATION`: contribution `-0.029864`
- `lag_12__T2__duck_amount`: contribution `+0.004316`
- `lag_11__T_place_DECON`: contribution `-0.004122`
- `lag_14__CT2__duck_amount`: contribution `-0.002702`
- `lag_03__T_place_DECON`: contribution `-0.002513`

Top utility-only movements:
- No utility movement among the top local contributors.
