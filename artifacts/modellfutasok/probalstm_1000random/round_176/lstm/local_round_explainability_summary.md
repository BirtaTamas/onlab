# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-metizport-inferno-qyaWW06KtkktSDfICHvaab/wildcard-vs-metizport-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `55770`, seconds `75.50`, LSTM `0.8010`, delta `+0.1412`
- tick `55802`, seconds `76.00`, LSTM `0.9025`, delta `+0.1015`
- tick `55962`, seconds `78.50`, LSTM `0.9624`, delta `+0.0655`
- tick `52090`, seconds `18.00`, LSTM `0.6110`, delta `-0.0354`
- tick `52570`, seconds `25.50`, LSTM `0.6063`, delta `-0.0319`
- tick `51994`, seconds `16.50`, LSTM `0.6582`, delta `+0.0301`
- tick `55578`, seconds `72.50`, LSTM `0.6475`, delta `+0.0265`
- tick `54458`, seconds `55.00`, LSTM `0.6513`, delta `+0.0246`
- tick `53082`, seconds `33.50`, LSTM `0.5682`, delta `-0.0243`
- tick `51610`, seconds `10.50`, LSTM `0.6351`, delta `+0.0234`

## Top 15 local ridge features

- `lag_01__T3__flash_duration`: coefficient `0.001703`, |coef| `0.001703`
- `lag_00__CT_kills_last_3s`: coefficient `0.001439`, |coef| `0.001439`
- `lag_02__T3__flash_duration`: coefficient `0.001242`, |coef| `0.001242`
- `lag_00__kill_diff_last_3s`: coefficient `0.001199`, |coef| `0.001199`
- `lag_01__T_flashed_players`: coefficient `0.001015`, |coef| `0.001015`
- `lag_00__damage_diff_last_5s`: coefficient `0.000994`, |coef| `0.000994`
- `lag_14__bomb_events_last_5s`: coefficient `0.000986`, |coef| `0.000986`
- `lag_00__T_place_TOPOFMID`: coefficient `-0.000964`, |coef| `0.000964`
- `lag_00__CT_damage_last_5s`: coefficient `0.000947`, |coef| `0.000947`
- `lag_15__T_place_TOPOFMID`: coefficient `0.000907`, |coef| `0.000907`
- `lag_02__T_flashed_players`: coefficient `0.000905`, |coef| `0.000905`
- `lag_13__T3__has_bomb`: coefficient `0.000888`, |coef| `0.000888`
- `lag_14__T1__has_bomb`: coefficient `-0.000883`, |coef| `0.000883`
- `lag_09__T1__duck_amount`: coefficient `0.000878`, |coef| `0.000878`
- `lag_00__T5__alive`: coefficient `-0.000875`, |coef| `0.000875`

## Top 10 utility ridge features

- `lag_01__T3__flash_duration`: coefficient `0.001703` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001242` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.000875` (raises CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000793` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000722` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000666` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000616` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000515` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000511` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000509` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001439` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001199` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `0.001015` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000994` (raises CT win probability)
- `lag_14__bomb_events_last_5s`: coefficient `0.000986` (raises CT win probability)
- `lag_00__T_place_TOPOFMID`: coefficient `-0.000964` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000947` (raises CT win probability)
- `lag_15__T_place_TOPOFMID`: coefficient `0.000907` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `0.000905` (raises CT win probability)
- `lag_13__T3__has_bomb`: coefficient `0.000888` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `55770`, seconds `75.50`, LSTM delta `+0.1412`

Top all feature movements:
- `lag_01__T3__flash_duration`: contribution `+0.011028`
- `lag_01__T_flashed_players`: contribution `+0.005876`
- `lag_00__CT_kills_last_3s`: contribution `+0.004154`
- `lag_01__T_flash_duration_sum`: contribution `+0.003700`
- `lag_06__CT3__is_scoped`: contribution `+0.003636`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.011028`
- `lag_01__T_flash_duration_sum`: contribution `+0.003700`

### tick `55802`, seconds `76.00`, LSTM delta `+0.1015`

Top all feature movements:
- `lag_02__T3__flash_duration`: contribution `+0.008042`
- `lag_02__T_flashed_players`: contribution `+0.005238`
- `lag_00__CT_kills_last_3s`: contribution `+0.004154`
- `lag_02__T_flash_duration_sum`: contribution `+0.003052`
- `lag_07__CT3__is_scoped`: contribution `+0.002987`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.008042`
- `lag_02__T_flash_duration_sum`: contribution `+0.003052`

### tick `55962`, seconds `78.50`, LSTM delta `+0.0655`

Top all feature movements:
- `lag_00__CT_place_ARCH`: contribution `+0.003531`
- `lag_00__T3__flash_duration`: contribution `-0.003295`
- `lag_00__T2__duck_amount`: contribution `+0.003199`
- `lag_03__CT_place_ARCH`: contribution `+0.002972`
- `lag_00__damage_diff_last_5s`: contribution `+0.002242`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.003295`
- `lag_07__T3__flash_duration`: contribution `+0.002174`
- `lag_00__T3__flash`: contribution `+0.001119`
- `lag_00__T3__utility_total`: contribution `+0.000905`
- `lag_07__T_flash_duration_sum`: contribution `+0.000860`

### tick `52090`, seconds `18.00`, LSTM delta `-0.0354`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `-0.007311`
- `lag_09__T1__duck_amount`: contribution `-0.003439`
- `lag_00__T1__is_scoped`: contribution `-0.003351`
- `lag_02__T_flashed_players`: contribution `+0.001746`
- `lag_00__T3__is_walking`: contribution `-0.001723`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52570`, seconds `25.50`, LSTM delta `-0.0319`

Top all feature movements:
- `lag_14__T_place_BALCONY`: contribution `-0.004850`
- `lag_14__T_place_APARTMENTS`: contribution `-0.003930`
- `lag_15__T1__is_scoped`: contribution `-0.002361`
- `lag_05__CT_place_ARCH`: contribution `-0.002113`
- `lag_15__CT_place_TOPOFMID`: contribution `-0.001590`

Top utility-only movements:
- No utility movement among the top local contributors.
