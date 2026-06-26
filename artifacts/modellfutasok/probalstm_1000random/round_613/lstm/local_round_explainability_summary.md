# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `17148`, seconds `52.00`, LSTM `0.8623`, delta `+0.1985`
- tick `17660`, seconds `60.00`, LSTM `0.9593`, delta `+0.0946`
- tick `16060`, seconds `35.00`, LSTM `0.6116`, delta `+0.0811`
- tick `17084`, seconds `51.00`, LSTM `0.7106`, delta `+0.0623`
- tick `15388`, seconds `24.50`, LSTM `0.4377`, delta `-0.0530`
- tick `16668`, seconds `44.50`, LSTM `0.6652`, delta `+0.0475`
- tick `17116`, seconds `51.50`, LSTM `0.6638`, delta `-0.0468`
- tick `15068`, seconds `19.50`, LSTM `0.5053`, delta `-0.0406`
- tick `15324`, seconds `23.50`, LSTM `0.4979`, delta `-0.0393`
- tick `15420`, seconds `25.00`, LSTM `0.4756`, delta `+0.0378`

## Top 15 local ridge features

- `lag_15__CT_place_SIDEALLEY`: coefficient `0.002637`, |coef| `0.002637`
- `lag_01__CT_place_SIDEALLEY`: coefficient `-0.002545`, |coef| `0.002545`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.002497`, |coef| `0.002497`
- `lag_00__CT_kills_last_3s`: coefficient `0.001715`, |coef| `0.001715`
- `lag_00__CT_place_SIDEALLEY`: coefficient `0.001602`, |coef| `0.001602`
- `lag_05__CT_place_CATWALK`: coefficient `-0.001446`, |coef| `0.001446`
- `lag_00__kill_diff_last_3s`: coefficient `0.001430`, |coef| `0.001430`
- `lag_13__CT_place_SIDEALLEY`: coefficient `0.001328`, |coef| `0.001328`
- `lag_12__CT_place_SHOP`: coefficient `0.001253`, |coef| `0.001253`
- `lag_02__T_place_UNDERPASS`: coefficient `0.001249`, |coef| `0.001249`
- `lag_01__CT_place_HOUSE`: coefficient `0.001184`, |coef| `0.001184`
- `lag_15__CT_place_CONNECTOR`: coefficient `0.001149`, |coef| `0.001149`
- `lag_09__CT4__is_scoped`: coefficient `0.001130`, |coef| `0.001130`
- `lag_05__T_place_UNDERPASS`: coefficient `0.001125`, |coef| `0.001125`
- `lag_15__CT_place_JUNGLE`: coefficient `-0.001080`, |coef| `0.001080`

## Top 10 utility ridge features

- `lag_00__T3__molly`: coefficient `-0.000858` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000789` (lowers CT win probability)
- `lag_15__T3__smoke`: coefficient `-0.000726` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000690` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000569` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000539` (raises CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000506` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.000499` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000490` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.000470` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SIDEALLEY`: coefficient `0.002637` (raises CT win probability)
- `lag_01__CT_place_SIDEALLEY`: coefficient `-0.002545` (lowers CT win probability)
- `lag_00__CT_place_BACKALLEY`: coefficient `0.002497` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001715` (raises CT win probability)
- `lag_00__CT_place_SIDEALLEY`: coefficient `0.001602` (raises CT win probability)
- `lag_05__CT_place_CATWALK`: coefficient `-0.001446` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001430` (raises CT win probability)
- `lag_13__CT_place_SIDEALLEY`: coefficient `0.001328` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.001253` (raises CT win probability)
- `lag_02__T_place_UNDERPASS`: coefficient `0.001249` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17148`, seconds `52.00`, LSTM delta `+0.1985`

Top all feature movements:
- `lag_15__CT_place_SIDEALLEY`: contribution `+0.048116`
- `lag_01__CT_place_SIDEALLEY`: contribution `+0.046436`
- `lag_05__CT_place_CATWALK`: contribution `+0.005760`
- `lag_00__CT_kills_last_3s`: contribution `+0.004953`
- `lag_05__T_place_UNDERPASS`: contribution `+0.004405`

Top utility-only movements:
- `lag_00__T3__molly`: contribution `+0.001905`

### tick `17660`, seconds `60.00`, LSTM delta `+0.0946`

Top all feature movements:
- `lag_00__CT_place_BACKALLEY`: contribution `+0.037427`
- `lag_00__CT_kills_last_3s`: contribution `+0.004953`
- `lag_02__T_place_UNDERPASS`: contribution `+0.004892`
- `lag_14__CT_place_SHOP`: contribution `+0.003923`
- `lag_00__kill_diff_last_3s`: contribution `+0.003442`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.001525`
- `lag_00__T2__flash`: contribution `+0.001443`

### tick `16060`, seconds `35.00`, LSTM delta `+0.0811`

Top all feature movements:
- `lag_15__CT_place_JUNGLE`: contribution `+0.006928`
- `lag_00__CT_kills_last_3s`: contribution `+0.004953`
- `lag_15__CT_place_CONNECTOR`: contribution `+0.004109`
- `lag_00__kill_diff_last_3s`: contribution `+0.003442`
- `lag_07__T_place_HOUSE`: contribution `+0.002439`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17084`, seconds `51.00`, LSTM delta `+0.0623`

Top all feature movements:
- `lag_13__CT_place_SIDEALLEY`: contribution `+0.024235`
- `lag_03__T_place_UNDERPASS`: contribution `+0.004112`
- `lag_00__CT2__duck_amount`: contribution `+0.003782`
- `lag_03__T_place_BACKALLEY`: contribution `+0.003006`
- `lag_15__T_place_UNDERPASS`: contribution `+0.002963`

Top utility-only movements:
- `lag_13__T3__smoke`: contribution `+0.000729`

### tick `15388`, seconds `24.50`, LSTM delta `-0.0530`

Top all feature movements:
- `lag_12__CT_place_LADDER`: contribution `-0.005423`
- `lag_10__T_shots_fired_sum`: contribution `-0.005185`
- `lag_09__CT_place_JUNGLE`: contribution `-0.004423`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.003445`
- `lag_03__CT_place_STAIRS`: contribution `-0.003236`

Top utility-only movements:
- No utility movement among the top local contributors.
