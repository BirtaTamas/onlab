# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `32457`, seconds `73.50`, LSTM `0.8195`, delta `+0.2287`
- tick `31657`, seconds `61.00`, LSTM `0.3674`, delta `-0.1583`
- tick `29225`, seconds `23.00`, LSTM `0.8136`, delta `+0.1526`
- tick `32393`, seconds `72.50`, LSTM `0.5063`, delta `+0.1234`
- tick `29257`, seconds `23.50`, LSTM `0.7071`, delta `-0.1066`
- tick `32361`, seconds `72.00`, LSTM `0.3829`, delta `+0.0979`
- tick `32969`, seconds `81.50`, LSTM `0.8661`, delta `-0.0848`
- tick `32425`, seconds `73.00`, LSTM `0.5908`, delta `+0.0845`
- tick `31689`, seconds `61.50`, LSTM `0.2897`, delta `-0.0776`
- tick `32553`, seconds `75.00`, LSTM `0.9457`, delta `+0.0635`

## Top 15 local ridge features

- `lag_02__T_place_STAIRS`: coefficient `-0.005249`, |coef| `0.005249`
- `lag_00__T_place_STAIRS`: coefficient `-0.002878`, |coef| `0.002878`
- `lag_03__T_place_STAIRS`: coefficient `-0.002520`, |coef| `0.002520`
- `lag_01__T_place_STAIRS`: coefficient `-0.002205`, |coef| `0.002205`
- `lag_00__kill_diff_last_3s`: coefficient `0.002105`, |coef| `0.002105`
- `lag_11__CT_place_TRUCK`: coefficient `-0.001824`, |coef| `0.001824`
- `lag_00__damage_diff_last_5s`: coefficient `0.001793`, |coef| `0.001793`
- `lag_00__CT_kills_last_3s`: coefficient `0.001754`, |coef| `0.001754`
- `lag_07__CT_place_CATWALK`: coefficient `0.001682`, |coef| `0.001682`
- `lag_03__T_bomb_zone_count`: coefficient `0.001636`, |coef| `0.001636`
- `lag_10__CT_place_SNIPERSNEST`: coefficient `0.001571`, |coef| `0.001571`
- `lag_00__CT_damage_last_5s`: coefficient `0.001495`, |coef| `0.001495`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001494`, |coef| `0.001494`
- `lag_07__CT_place_SNIPERSNEST`: coefficient `0.001469`, |coef| `0.001469`
- `lag_08__CT_place_SNIPERSNEST`: coefficient `0.001348`, |coef| `0.001348`

## Top 10 utility ridge features

- `lag_02__T_A_site_active_infernos`: coefficient `-0.001295` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001015` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000948` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000773` (lowers CT win probability)
- `lag_09__T4__molly`: coefficient `-0.000742` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000653` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000638` (lowers CT win probability)
- `lag_03__T4__flash`: coefficient `-0.000633` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000631` (raises CT win probability)
- `lag_10__T4__molly`: coefficient `-0.000616` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_STAIRS`: coefficient `-0.005249` (lowers CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `-0.002878` (lowers CT win probability)
- `lag_03__T_place_STAIRS`: coefficient `-0.002520` (lowers CT win probability)
- `lag_01__T_place_STAIRS`: coefficient `-0.002205` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002105` (raises CT win probability)
- `lag_11__CT_place_TRUCK`: coefficient `-0.001824` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001793` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001754` (raises CT win probability)
- `lag_07__CT_place_CATWALK`: coefficient `0.001682` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.001636` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `32457`, seconds `73.50`, LSTM delta `+0.2287`

Top all feature movements:
- `lag_02__T_place_STAIRS`: contribution `+0.100479`
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.008414`
- `lag_06__T_bomb_zone_count`: contribution `+0.006534`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005190`
- `lag_00__T_place_CONNECTOR`: contribution `+0.005079`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31657`, seconds `61.00`, LSTM delta `-0.1583`

Top all feature movements:
- `lag_02__T_place_STAIRS`: contribution `-0.100479`
- `lag_00__kill_diff_last_3s`: contribution `-0.005066`
- `lag_00__CT_place_STAIRS`: contribution `-0.004593`
- `lag_00__damage_diff_last_5s`: contribution `-0.004044`
- `lag_12__CT_place_TRUCK`: contribution `-0.002928`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29225`, seconds `23.00`, LSTM delta `+0.1526`

Top all feature movements:
- `lag_11__CT_place_TRUCK`: contribution `+0.011767`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005190`
- `lag_00__kill_diff_last_3s`: contribution `+0.005066`
- `lag_00__CT_kills_last_3s`: contribution `+0.005063`
- `lag_00__damage_diff_last_5s`: contribution `+0.004449`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.003166`
- `lag_00__T1__flash_duration`: contribution `+0.001744`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.001625`
- `lag_08__T1__flash_duration`: contribution `+0.001538`

### tick `32393`, seconds `72.50`, LSTM delta `+0.1234`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.055091`
- `lag_04__T_bomb_zone_count`: contribution `+0.007434`
- `lag_08__CT_place_SNIPERSNEST`: contribution `+0.007218`
- `lag_00__kill_diff_last_3s`: contribution `+0.005066`
- `lag_00__CT_kills_last_3s`: contribution `+0.005063`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.003022`
- `lag_10__T4__molly`: contribution `+0.001343`

### tick `29257`, seconds `23.50`, LSTM delta `-0.1066`

Top all feature movements:
- `lag_11__CT_place_TRUCK`: contribution `-0.011767`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005190`
- `lag_00__kill_diff_last_3s`: contribution `-0.005066`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.003904`
- `lag_00__T_shots_fired_sum`: contribution `-0.003776`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.002879`
- `lag_09__CT2__flash_duration`: contribution `-0.001513`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001414`
