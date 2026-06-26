# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `21`

## Largest probability jumps

- tick `190013`, seconds `89.00`, LSTM `0.8931`, delta `+0.2162`
- tick `189949`, seconds `88.00`, LSTM `0.6147`, delta `+0.1625`
- tick `189501`, seconds `81.00`, LSTM `0.7512`, delta `-0.1379`
- tick `185405`, seconds `17.00`, LSTM `0.8487`, delta `+0.0998`
- tick `185309`, seconds `15.50`, LSTM `0.6934`, delta `+0.0923`
- tick `189693`, seconds `84.00`, LSTM `0.5101`, delta `-0.0903`
- tick `189533`, seconds `81.50`, LSTM `0.6818`, delta `-0.0694`
- tick `185437`, seconds `17.50`, LSTM `0.7832`, delta `-0.0655`
- tick `185373`, seconds `16.50`, LSTM `0.7489`, delta `+0.0640`
- tick `189981`, seconds `88.50`, LSTM `0.6769`, delta `+0.0622`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001874`, |coef| `0.001874`
- `lag_00__kill_diff_last_3s`: coefficient `0.001816`, |coef| `0.001816`
- `lag_00__damage_diff_last_5s`: coefficient `0.001655`, |coef| `0.001655`
- `lag_02__kill_diff_last_3s`: coefficient `0.001493`, |coef| `0.001493`
- `lag_09__CT2__flash_duration`: coefficient `-0.001412`, |coef| `0.001412`
- `lag_00__CT_kills_last_3s`: coefficient `0.001371`, |coef| `0.001371`
- `lag_04__CT_place_WATER`: coefficient `0.001322`, |coef| `0.001322`
- `lag_01__CT_place_BACKOFA`: coefficient `-0.001319`, |coef| `0.001319`
- `lag_03__T_bomb_zone_count`: coefficient `0.001310`, |coef| `0.001310`
- `lag_13__CT_place_WATER`: coefficient `-0.001301`, |coef| `0.001301`
- `lag_00__CT_place_WALKWAY`: coefficient `0.001215`, |coef| `0.001215`
- `lag_12__CT2__flash_duration`: coefficient `-0.001158`, |coef| `0.001158`
- `lag_00__CT2__duck_amount`: coefficient `0.001141`, |coef| `0.001141`
- `lag_14__CT2__flash_duration`: coefficient `-0.001120`, |coef| `0.001120`
- `lag_11__CT_place_WATER`: coefficient `-0.001116`, |coef| `0.001116`

## Top 10 utility ridge features

- `lag_09__CT2__flash_duration`: coefficient `-0.001412` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.001158` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001120` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001082` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001059` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `-0.001005` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000873` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.000825` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `-0.000749` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001874` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001816` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001655` (raises CT win probability)
- `lag_02__kill_diff_last_3s`: coefficient `0.001493` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001371` (raises CT win probability)
- `lag_04__CT_place_WATER`: coefficient `0.001322` (raises CT win probability)
- `lag_01__CT_place_BACKOFA`: coefficient `-0.001319` (lowers CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.001310` (raises CT win probability)
- `lag_13__CT_place_WATER`: coefficient `-0.001301` (lowers CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `0.001215` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `190013`, seconds `89.00`, LSTM delta `+0.2162`

Top all feature movements:
- `lag_04__CT_place_WATER`: contribution `+0.008035`
- `lag_13__CT_place_WATER`: contribution `+0.007907`
- `lag_03__T_bomb_zone_count`: contribution `+0.007624`
- `lag_14__CT2__flash_duration`: contribution `+0.005690`
- `lag_04__T_place_WATER`: contribution `+0.005460`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `+0.005690`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.005361`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.003626`

### tick `189949`, seconds `88.00`, LSTM delta `+0.1625`

Top all feature movements:
- `lag_13__CT_place_BACKOFA`: contribution `+0.009138`
- `lag_08__CT_shots_fired_sum`: contribution `+0.007292`
- `lag_11__CT_place_WATER`: contribution `+0.006779`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006509`
- `lag_02__T_place_WATER`: contribution `+0.006129`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `+0.005886`

### tick `189501`, seconds `81.00`, LSTM delta `-0.1379`

Top all feature movements:
- `lag_01__CT_place_BACKOFA`: contribution `-0.012741`
- `lag_09__CT2__flash_duration`: contribution `-0.007179`
- `lag_00__CT_place_WALKWAY`: contribution `-0.005962`
- `lag_00__kill_diff_last_3s`: contribution `-0.004371`
- `lag_00__damage_diff_last_5s`: contribution `-0.003697`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.007179`
- `lag_09__CT_flash_duration_sum`: contribution `-0.001764`
- `lag_00__T4__molly`: contribution `-0.001579`

### tick `185405`, seconds `17.00`, LSTM delta `+0.0998`

Top all feature movements:
- `lag_13__CT_place_WATER`: contribution `+0.007907`
- `lag_02__CT_place_FOUNTAIN`: contribution `+0.005127`
- `lag_00__kill_diff_last_3s`: contribution `+0.004371`
- `lag_00__CT_kills_last_3s`: contribution `+0.003958`
- `lag_02__CT_place_UPPERPARK`: contribution `+0.003579`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.001668`

### tick `185309`, seconds `15.50`, LSTM delta `+0.0923`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006509`
- `lag_00__kill_diff_last_3s`: contribution `+0.004371`
- `lag_00__CT_kills_last_3s`: contribution `+0.003958`
- `lag_00__T_place_FOUNTAIN`: contribution `+0.003269`
- `lag_07__CT_flashed_players`: contribution `+0.003210`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.003090`
