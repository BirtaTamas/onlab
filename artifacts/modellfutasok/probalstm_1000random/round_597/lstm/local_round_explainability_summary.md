# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `27146`, seconds `63.50`, LSTM `0.4692`, delta `+0.3326`
- tick `27530`, seconds `69.50`, LSTM `0.7712`, delta `+0.2569`
- tick `25514`, seconds `38.00`, LSTM `0.3101`, delta `-0.2432`
- tick `26922`, seconds `60.00`, LSTM `0.3373`, delta `-0.2034`
- tick `27082`, seconds `62.50`, LSTM `0.1591`, delta `-0.1082`
- tick `26986`, seconds `61.00`, LSTM `0.2099`, delta `-0.0838`
- tick `25834`, seconds `43.00`, LSTM `0.4186`, delta `+0.0815`
- tick `27306`, seconds `66.00`, LSTM `0.5344`, delta `+0.0734`
- tick `26634`, seconds `55.50`, LSTM `0.4295`, delta `-0.0669`
- tick `27210`, seconds `64.50`, LSTM `0.4910`, delta `+0.0579`

## Top 15 local ridge features

- `lag_02__T_place_QUAD`: coefficient `-0.003267`, |coef| `0.003267`
- `lag_13__T_place_BALCONY`: coefficient `0.002769`, |coef| `0.002769`
- `lag_00__T_place_BALCONY`: coefficient `-0.002269`, |coef| `0.002269`
- `lag_00__kill_diff_last_3s`: coefficient `0.002234`, |coef| `0.002234`
- `lag_14__T_place_BALCONY`: coefficient `0.002192`, |coef| `0.002192`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001988`, |coef| `0.001988`
- `lag_15__CT2__duck_amount`: coefficient `-0.001958`, |coef| `0.001958`
- `lag_14__T_place_QUAD`: coefficient `-0.001891`, |coef| `0.001891`
- `lag_04__T_place_QUAD`: coefficient `0.001851`, |coef| `0.001851`
- `lag_00__damage_diff_last_5s`: coefficient `0.001714`, |coef| `0.001714`
- `lag_00__T_kills_last_3s`: coefficient `-0.001652`, |coef| `0.001652`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001640`, |coef| `0.001640`
- `lag_00__T3__shots_fired`: coefficient `-0.001616`, |coef| `0.001616`
- `lag_04__CT5__is_scoped`: coefficient `0.001496`, |coef| `0.001496`
- `lag_00__T_damage_last_5s`: coefficient `-0.001398`, |coef| `0.001398`

## Top 10 utility ridge features

- `lag_01__CT1__flash_duration`: coefficient `-0.001071` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.000827` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000819` (lowers CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.000791` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.000763` (lowers CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.000714` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000698` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000679` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000648` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.000642` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_QUAD`: coefficient `-0.003267` (lowers CT win probability)
- `lag_13__T_place_BALCONY`: coefficient `0.002769` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.002269` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002234` (raises CT win probability)
- `lag_14__T_place_BALCONY`: coefficient `0.002192` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001988` (lowers CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `-0.001958` (lowers CT win probability)
- `lag_14__T_place_QUAD`: coefficient `-0.001891` (lowers CT win probability)
- `lag_04__T_place_QUAD`: coefficient `0.001851` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001714` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `27146`, seconds `63.50`, LSTM delta `+0.3326`

Top all feature movements:
- `lag_02__T_place_QUAD`: contribution `+0.078698`
- `lag_04__T_place_QUAD`: contribution `+0.044580`
- `lag_00__T_place_BALCONY`: contribution `+0.031199`
- `lag_15__CT2__duck_amount`: contribution `+0.007427`
- `lag_02__T_place_BALCONY`: contribution `+0.007044`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27530`, seconds `69.50`, LSTM delta `+0.2569`

Top all feature movements:
- `lag_14__T_place_QUAD`: contribution `+0.045546`
- `lag_00__T_place_BALCONY`: contribution `+0.031199`
- `lag_14__T_place_BALCONY`: contribution `+0.030141`
- `lag_11__T_place_BALCONY`: contribution `+0.017321`
- `lag_08__T_place_BALCONY`: contribution `+0.012857`

Top utility-only movements:
- `lag_09__T4__flash_duration`: contribution `+0.005039`
- `lag_04__T3__flash_duration`: contribution `+0.003294`
- `lag_07__T3__flash_duration`: contribution `+0.002748`

### tick `25514`, seconds `38.00`, LSTM delta `-0.2432`

Top all feature movements:
- `lag_13__T_place_BALCONY`: contribution `-0.038080`
- `lag_00__T_shots_fired_sum`: contribution `+0.010436`
- `lag_01__T_shots_fired_sum`: contribution `-0.006149`
- `lag_15__CT2__duck_amount`: contribution `-0.006144`
- `lag_11__CT_place_PIT`: contribution `-0.005453`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.003332`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.002694`

### tick `26922`, seconds `60.00`, LSTM delta `-0.2034`

Top all feature movements:
- `lag_15__CT2__duck_amount`: contribution `-0.007459`
- `lag_00__T_shots_fired_sum`: contribution `-0.005963`
- `lag_07__T_shots_fired_sum`: contribution `-0.005698`
- `lag_07__T3__shots_fired`: contribution `-0.005560`
- `lag_00__kill_diff_last_3s`: contribution `-0.005377`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.002437`

### tick `27082`, seconds `62.50`, LSTM delta `-0.1082`

Top all feature movements:
- `lag_02__T_place_QUAD`: contribution `-0.078698`
- `lag_00__T_place_BALCONY`: contribution `-0.031199`
- `lag_00__T_place_QUAD`: contribution `+0.010648`
- `lag_02__T_shots_fired_sum`: contribution `+0.006101`
- `lag_04__CT_flashed_players`: contribution `-0.003554`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `-0.002454`
