# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `125745`, seconds `87.50`, LSTM `0.8492`, delta `+0.1298`
- tick `122385`, seconds `35.00`, LSTM `0.8338`, delta `+0.0944`
- tick `125489`, seconds `83.50`, LSTM `0.7032`, delta `-0.0786`
- tick `120945`, seconds `12.50`, LSTM `0.7291`, delta `+0.0774`
- tick `124305`, seconds `65.00`, LSTM `0.8849`, delta `+0.0565`
- tick `122289`, seconds `33.50`, LSTM `0.7667`, delta `+0.0543`
- tick `125937`, seconds `90.50`, LSTM `0.9450`, delta `+0.0532`
- tick `121265`, seconds `17.50`, LSTM `0.7285`, delta `-0.0436`
- tick `125873`, seconds `89.50`, LSTM `0.8559`, delta `-0.0434`
- tick `121009`, seconds `13.50`, LSTM `0.8026`, delta `+0.0405`

## Top 15 local ridge features

- `lag_00__CT_place_SIDEALLEY`: coefficient `0.001545`, |coef| `0.001545`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001153`, |coef| `0.001153`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001110`, |coef| `0.001110`
- `lag_00__T_place_STAIRS`: coefficient `-0.001095`, |coef| `0.001095`
- `lag_00__CT_place_TRUCK`: coefficient `0.001042`, |coef| `0.001042`
- `lag_08__T_place_STAIRS`: coefficient `0.001024`, |coef| `0.001024`
- `lag_11__CT_place_PALACEALLEY`: coefficient `0.001011`, |coef| `0.001011`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.000855`, |coef| `0.000855`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000847`, |coef| `0.000847`
- `lag_06__T_place_JUNGLE`: coefficient `0.000842`, |coef| `0.000842`
- `lag_03__CT_place_TRUCK`: coefficient `0.000811`, |coef| `0.000811`
- `lag_03__T_place_JUNGLE`: coefficient `-0.000781`, |coef| `0.000781`
- `lag_06__CT_place_TRAMP`: coefficient `0.000696`, |coef| `0.000696`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000694`, |coef| `0.000694`
- `lag_03__CT_place_PALACEALLEY`: coefficient `-0.000663`, |coef| `0.000663`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000847` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000694` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000586` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000540` (raises CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.000417` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000402` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `-0.000387` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000361` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000359` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.000344` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SIDEALLEY`: coefficient `0.001545` (raises CT win probability)
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001153` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001110` (lowers CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `-0.001095` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001042` (raises CT win probability)
- `lag_08__T_place_STAIRS`: coefficient `0.001024` (raises CT win probability)
- `lag_11__CT_place_PALACEALLEY`: coefficient `0.001011` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.000855` (lowers CT win probability)
- `lag_06__T_place_JUNGLE`: coefficient `0.000842` (raises CT win probability)
- `lag_03__CT_place_TRUCK`: coefficient `0.000811` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `125745`, seconds `87.50`, LSTM delta `+0.1298`

Top all feature movements:
- `lag_08__T_place_STAIRS`: contribution `+0.019596`
- `lag_11__CT_place_PALACEALLEY`: contribution `+0.015431`
- `lag_06__T_place_JUNGLE`: contribution `+0.010912`
- `lag_03__T_place_JUNGLE`: contribution `+0.010116`
- `lag_08__T_shots_fired_sum`: contribution `+0.005443`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122385`, seconds `35.00`, LSTM delta `+0.0944`

Top all feature movements:
- `lag_03__CT_place_TRUCK`: contribution `+0.005230`
- `lag_14__CT_place_STAIRS`: contribution `+0.004172`
- `lag_13__T_flashed_players`: contribution `+0.003647`
- `lag_13__T4__flash_duration`: contribution `+0.003453`
- `lag_04__CT_place_TRUCK`: contribution `+0.002967`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.003453`
- `lag_13__T_flash_duration_sum`: contribution `+0.002380`
- `lag_02__T4__flash_duration`: contribution `+0.002127`

### tick `125489`, seconds `83.50`, LSTM delta `-0.0786`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `-0.020966`
- `lag_03__CT_place_PALACEALLEY`: contribution `-0.010127`
- `lag_00__T_shots_fired_sum`: contribution `-0.009153`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005148`
- `lag_03__CT_place_TSPAWN`: contribution `-0.004302`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120945`, seconds `12.50`, LSTM delta `+0.0774`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.009039`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.006081`
- `lag_04__CT3__flash_duration`: contribution `+0.003204`
- `lag_11__T_flashed_players`: contribution `+0.002542`
- `lag_01__CT_place_TRUCK`: contribution `+0.002452`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.009039`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.006081`
- `lag_04__CT3__flash_duration`: contribution `+0.003204`
- `lag_03__T5__flash_duration`: contribution `+0.001906`
- `lag_11__T_flash_duration_sum`: contribution `+0.001594`

### tick `124305`, seconds `65.00`, LSTM delta `+0.0565`

Top all feature movements:
- `lag_00__CT_place_SIDEALLEY`: contribution `+0.028194`
- `lag_00__CT_place_HOUSE`: contribution `+0.001326`
- `lag_14__T_place_MIDDLE`: contribution `+0.001243`
- `lag_02__T_place_CATWALK`: contribution `+0.001165`
- `lag_05__CT1__duck_amount`: contribution `+0.000912`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.000767`
- `lag_08__CT_active_infernos`: contribution `+0.000665`
