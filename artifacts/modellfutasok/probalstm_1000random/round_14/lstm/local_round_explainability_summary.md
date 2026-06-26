# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `13`

## Largest probability jumps

- tick `130305`, seconds `38.50`, LSTM `0.7950`, delta `+0.1905`
- tick `130369`, seconds `39.50`, LSTM `0.9300`, delta `+0.0920`
- tick `130081`, seconds `35.00`, LSTM `0.5879`, delta `+0.0721`
- tick `131393`, seconds `55.50`, LSTM `0.9041`, delta `-0.0595`
- tick `130337`, seconds `39.00`, LSTM `0.8379`, delta `+0.0429`
- tick `131201`, seconds `52.50`, LSTM `0.9481`, delta `+0.0385`
- tick `127905`, seconds `1.00`, LSTM `0.5157`, delta `+0.0349`
- tick `131425`, seconds `56.00`, LSTM `0.9380`, delta `+0.0339`
- tick `130113`, seconds `35.50`, LSTM `0.6150`, delta `+0.0272`
- tick `131105`, seconds `51.00`, LSTM `0.8939`, delta `+0.0271`

## Top 15 local ridge features

- `lag_10__CT_place_TSIDELOWER`: coefficient `-0.002812`, |coef| `0.002812`
- `lag_10__CT_place_TSIDEUPPER`: coefficient `0.001994`, |coef| `0.001994`
- `lag_00__damage_diff_last_5s`: coefficient `0.001901`, |coef| `0.001901`
- `lag_00__CT_kills_last_3s`: coefficient `0.001673`, |coef| `0.001673`
- `lag_00__kill_diff_last_3s`: coefficient `0.001672`, |coef| `0.001672`
- `lag_00__CT_damage_last_5s`: coefficient `0.001593`, |coef| `0.001593`
- `lag_03__CT_place_HOUSE`: coefficient `0.001555`, |coef| `0.001555`
- `lag_00__T_place_HOUSE`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_05__CT_place_HOUSE`: coefficient `0.001271`, |coef| `0.001271`
- `lag_04__CT_place_HOUSE`: coefficient `0.001193`, |coef| `0.001193`
- `lag_11__T2__is_walking`: coefficient `0.001089`, |coef| `0.001089`
- `lag_06__CT3__duck_amount`: coefficient `-0.001028`, |coef| `0.001028`
- `lag_00__T1__alive`: coefficient `-0.001008`, |coef| `0.001008`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001003`, |coef| `0.001003`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001002`, |coef| `0.001002`

## Top 10 utility ridge features

- `lag_13__T_A_site_active_infernos`: coefficient `-0.000900` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.000648` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000487` (lowers CT win probability)
- `lag_13__active_infernos_total`: coefficient `-0.000466` (lowers CT win probability)
- `lag_04__T_B_site_active_smokes`: coefficient `0.000399` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000392` (lowers CT win probability)
- `lag_10__T2__smoke`: coefficient `-0.000388` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `0.000372` (raises CT win probability)
- `lag_08__T2__smoke`: coefficient `-0.000367` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000366` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_TSIDELOWER`: coefficient `-0.002812` (lowers CT win probability)
- `lag_10__CT_place_TSIDEUPPER`: coefficient `0.001994` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001901` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001673` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001672` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001593` (raises CT win probability)
- `lag_03__CT_place_HOUSE`: coefficient `0.001555` (raises CT win probability)
- `lag_00__T_place_HOUSE`: coefficient `-0.001350` (lowers CT win probability)
- `lag_05__CT_place_HOUSE`: coefficient `0.001271` (raises CT win probability)
- `lag_04__CT_place_HOUSE`: coefficient `0.001193` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `130305`, seconds `38.50`, LSTM delta `+0.1905`

Top all feature movements:
- `lag_10__CT_place_TSIDELOWER`: contribution `+0.038200`
- `lag_10__CT_place_TSIDEUPPER`: contribution `+0.014992`
- `lag_00__T_place_HOUSE`: contribution `+0.005937`
- `lag_03__CT_place_HOUSE`: contribution `+0.005494`
- `lag_00__damage_diff_last_5s`: contribution `+0.005104`

Top utility-only movements:
- `lag_13__T_A_site_active_infernos`: contribution `+0.002678`

### tick `130369`, seconds `39.50`, LSTM delta `+0.0920`

Top all feature movements:
- `lag_12__CT_place_TSIDEUPPER`: contribution `+0.007365`
- `lag_12__CT_place_TSIDELOWER`: contribution `+0.006523`
- `lag_00__T_place_HOUSE`: contribution `+0.005937`
- `lag_00__CT_kills_last_3s`: contribution `+0.004829`
- `lag_05__CT_place_HOUSE`: contribution `+0.004490`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `130081`, seconds `35.00`, LSTM delta `+0.0721`

Top all feature movements:
- `lag_03__CT_place_TSIDELOWER`: contribution `+0.008122`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.007534`
- `lag_12__CT_place_TSIDELOWER`: contribution `-0.006523`
- `lag_00__CT_kills_last_3s`: contribution `+0.004829`
- `lag_00__kill_diff_last_3s`: contribution `+0.004023`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131393`, seconds `55.50`, LSTM delta `-0.0595`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008047`
- `lag_00__T_duck_amount_mean`: contribution `-0.005831`
- `lag_00__CT_kills_last_3s`: contribution `-0.004829`
- `lag_05__CT3__duck_amount`: contribution `-0.003111`
- `lag_00__CT3__duck_amount`: contribution `-0.003060`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `130337`, seconds `39.00`, LSTM delta `+0.0429`

Top all feature movements:
- `lag_11__CT_place_TSIDELOWER`: contribution `+0.006537`
- `lag_11__CT_place_TSIDEUPPER`: contribution `+0.006192`
- `lag_04__CT_place_HOUSE`: contribution `+0.004215`
- `lag_07__CT3__duck_amount`: contribution `-0.003091`
- `lag_15__CT3__duck_amount`: contribution `+0.002487`

Top utility-only movements:
- No utility movement among the top local contributors.
