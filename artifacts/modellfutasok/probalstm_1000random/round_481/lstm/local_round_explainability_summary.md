# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `15`

## Largest probability jumps

- tick `129183`, seconds `12.00`, LSTM `0.0722`, delta `-0.0690`
- tick `132703`, seconds `67.00`, LSTM `0.0221`, delta `-0.0673`
- tick `132351`, seconds `61.50`, LSTM `0.0802`, delta `+0.0609`
- tick `132383`, seconds `62.00`, LSTM `0.1310`, delta `+0.0508`
- tick `128447`, seconds `0.50`, LSTM `0.2089`, delta `-0.0466`
- tick `129087`, seconds `10.50`, LSTM `0.1671`, delta `-0.0459`
- tick `131903`, seconds `54.50`, LSTM `0.0274`, delta `-0.0431`
- tick `132991`, seconds `71.50`, LSTM `0.0497`, delta `+0.0397`
- tick `128959`, seconds `8.50`, LSTM `0.1867`, delta `-0.0386`
- tick `128543`, seconds `2.00`, LSTM `0.2706`, delta `+0.0374`

## Top 15 local ridge features

- `lag_00__CT_place_WATER`: coefficient `0.000937`, |coef| `0.000937`
- `lag_00__CT_place_BACKOFA`: coefficient `0.000921`, |coef| `0.000921`
- `lag_14__T_place_CONSTRUCTION`: coefficient `-0.000859`, |coef| `0.000859`
- `lag_01__CT_place_BACKOFA`: coefficient `0.000756`, |coef| `0.000756`
- `lag_14__CT_place_WATER`: coefficient `-0.000742`, |coef| `0.000742`
- `lag_02__CT_place_BACKOFA`: coefficient `0.000672`, |coef| `0.000672`
- `lag_13__CT_place_WATER`: coefficient `-0.000637`, |coef| `0.000637`
- `lag_02__T_place_PLAYGROUND`: coefficient `-0.000630`, |coef| `0.000630`
- `lag_00__CT_place_WALKWAY`: coefficient `-0.000630`, |coef| `0.000630`
- `lag_13__T_place_CONSTRUCTION`: coefficient `-0.000613`, |coef| `0.000613`
- `lag_01__T_place_CONSTRUCTION`: coefficient `-0.000571`, |coef| `0.000571`
- `lag_04__T_place_CONSTRUCTION`: coefficient `-0.000547`, |coef| `0.000547`
- `lag_00__kill_diff_last_3s`: coefficient `0.000521`, |coef| `0.000521`
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000508`, |coef| `0.000508`
- `lag_15__CT_place_WATER`: coefficient `-0.000495`, |coef| `0.000495`

## Top 10 utility ridge features

- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000508` (raises CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.000449` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000424` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000424` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.000411` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000389` (raises CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.000369` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000355` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000348` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.000339` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_WATER`: coefficient `0.000937` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.000921` (raises CT win probability)
- `lag_14__T_place_CONSTRUCTION`: coefficient `-0.000859` (lowers CT win probability)
- `lag_01__CT_place_BACKOFA`: coefficient `0.000756` (raises CT win probability)
- `lag_14__CT_place_WATER`: coefficient `-0.000742` (lowers CT win probability)
- `lag_02__CT_place_BACKOFA`: coefficient `0.000672` (raises CT win probability)
- `lag_13__CT_place_WATER`: coefficient `-0.000637` (lowers CT win probability)
- `lag_02__T_place_PLAYGROUND`: coefficient `-0.000630` (lowers CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `-0.000630` (lowers CT win probability)
- `lag_13__T_place_CONSTRUCTION`: coefficient `-0.000613` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `129183`, seconds `12.00`, LSTM delta `-0.0690`

Top all feature movements:
- `lag_02__T_place_PLAYGROUND`: contribution `-0.009250`
- `lag_07__CT_place_WALKWAY`: contribution `-0.004559`
- `lag_03__CT_place_BRIDGE`: contribution `-0.003632`
- `lag_14__T_place_TSTAIRS`: contribution `-0.003319`
- `lag_11__T_place_TUNNELS`: contribution `-0.002273`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `132703`, seconds `67.00`, LSTM delta `-0.0673`

Top all feature movements:
- `lag_13__T_place_CONSTRUCTION`: contribution `-0.007625`
- `lag_04__T_place_CONSTRUCTION`: contribution `-0.006795`
- `lag_00__CT_place_WATER`: contribution `-0.005692`
- `lag_09__T_place_CONSTRUCTION`: contribution `-0.005522`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.004804`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.004804`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.004013`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.003024`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.002631`
- `lag_07__CT2__flash_duration`: contribution `-0.001704`

### tick `132351`, seconds `61.50`, LSTM delta `+0.0609`

Top all feature movements:
- `lag_00__CT_place_WATER`: contribution `+0.005692`
- `lag_14__CT_place_WATER`: contribution `+0.004507`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.004011`
- `lag_13__CT_place_WATER`: contribution `+0.003870`
- `lag_00__CT_place_WALKWAY`: contribution `+0.003091`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.004011`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002704`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.001398`
- `lag_14__CT1__flash_duration`: contribution `+0.001317`

### tick `132383`, seconds `62.00`, LSTM delta `+0.0508`

Top all feature movements:
- `lag_01__T_place_CONSTRUCTION`: contribution `+0.007094`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.004804`
- `lag_14__CT_place_WATER`: contribution `+0.004507`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.003024`
- `lag_15__CT_place_WATER`: contribution `+0.003008`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.004804`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.003024`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.000932`

### tick `128447`, seconds `0.50`, LSTM delta `-0.0466`

Top all feature movements:
- `lag_01__CT_macro_A`: contribution `-0.002522`
- `lag_01__CT_place_BOMBSITEA`: contribution `-0.002522`
- `lag_00__CT_velocity_mean`: contribution `-0.001258`
- `lag_01__T_place_TSPAWN`: contribution `-0.000980`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000829`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000787`
- `lag_01__T_molly_inv`: contribution `-0.000533`
- `lag_01__T_smoke_inv`: contribution `-0.000510`
- `lag_01__T2__utility_total`: contribution `-0.000456`
- `lag_01__T_utility_inv`: contribution `-0.000430`
