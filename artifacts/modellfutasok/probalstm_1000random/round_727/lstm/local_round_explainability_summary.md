# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `15`

## Largest probability jumps

- tick `121002`, seconds `88.50`, LSTM `0.4278`, delta `-0.2552`
- tick `121130`, seconds `90.50`, LSTM `0.0312`, delta `-0.2083`
- tick `121034`, seconds `89.00`, LSTM `0.3075`, delta `-0.1204`
- tick `121450`, seconds `95.50`, LSTM `0.0747`, delta `+0.0635`
- tick `120298`, seconds `77.50`, LSTM `0.6189`, delta `-0.0458`
- tick `121066`, seconds `89.50`, LSTM `0.2624`, delta `-0.0451`
- tick `117642`, seconds `36.00`, LSTM `0.6601`, delta `-0.0423`
- tick `116426`, seconds `17.00`, LSTM `0.5855`, delta `+0.0418`
- tick `116106`, seconds `12.00`, LSTM `0.5787`, delta `+0.0386`
- tick `116586`, seconds `19.50`, LSTM `0.6163`, delta `+0.0371`

## Top 15 local ridge features

- `lag_01__T_place_EXTENDEDA`: coefficient `-0.003060`, |coef| `0.003060`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `0.002219`, |coef| `0.002219`
- `lag_06__T_place_SHORTSTAIRS`: coefficient `-0.002180`, |coef| `0.002180`
- `lag_09__T5__flash_duration`: coefficient `0.002159`, |coef| `0.002159`
- `lag_05__T_shots_fired_sum`: coefficient `0.001886`, |coef| `0.001886`
- `lag_00__T_kills_last_3s`: coefficient `-0.001874`, |coef| `0.001874`
- `lag_07__T_place_SHORTSTAIRS`: coefficient `-0.001792`, |coef| `0.001792`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_15__CT_place_BDOORS`: coefficient `0.001672`, |coef| `0.001672`
- `lag_00__CT_place_ARAMP`: coefficient `0.001649`, |coef| `0.001649`
- `lag_00__T_place_SHORTSTAIRS`: coefficient `0.001577`, |coef| `0.001577`
- `lag_14__T_place_EXTENDEDA`: coefficient `-0.001555`, |coef| `0.001555`
- `lag_00__CT4__duck_amount`: coefficient `0.001553`, |coef| `0.001553`
- `lag_00__damage_diff_last_5s`: coefficient `0.001542`, |coef| `0.001542`
- `lag_00__kill_diff_last_3s`: coefficient `0.001525`, |coef| `0.001525`

## Top 10 utility ridge features

- `lag_09__T5__flash_duration`: coefficient `0.002159` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.001516` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.001261` (raises CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.001059` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.001031` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001014` (lowers CT win probability)
- `lag_11__CT1__molly`: coefficient `0.000967` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.000872` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.000854` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.000776` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_EXTENDEDA`: coefficient `-0.003060` (lowers CT win probability)
- `lag_01__T_place_SHORTSTAIRS`: coefficient `0.002219` (raises CT win probability)
- `lag_06__T_place_SHORTSTAIRS`: coefficient `-0.002180` (lowers CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `0.001886` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001874` (lowers CT win probability)
- `lag_07__T_place_SHORTSTAIRS`: coefficient `-0.001792` (lowers CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001779` (lowers CT win probability)
- `lag_15__CT_place_BDOORS`: coefficient `0.001672` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `0.001649` (raises CT win probability)
- `lag_00__T_place_SHORTSTAIRS`: coefficient `0.001577` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121002`, seconds `88.50`, LSTM delta `-0.2552`

Top all feature movements:
- `lag_09__T5__flash_duration`: contribution `-0.015633`
- `lag_01__T_place_EXTENDEDA`: contribution `-0.015171`
- `lag_05__T_shots_fired_sum`: contribution `-0.011315`
- `lag_00__CT_place_ARAMP`: contribution `-0.010272`
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.009324`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.015633`

### tick `121130`, seconds `90.50`, LSTM delta `-0.2083`

Top all feature movements:
- `lag_01__T_place_EXTENDEDA`: contribution `-0.015171`
- `lag_13__T5__flash_duration`: contribution `-0.010975`
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.009324`
- `lag_06__T_place_SHORTSTAIRS`: contribution `-0.009163`
- `lag_14__T_place_EXTENDEDA`: contribution `-0.007711`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.010975`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.003637`

### tick `121034`, seconds `89.00`, LSTM delta `-0.1204`

Top all feature movements:
- `lag_10__T5__flash_duration`: contribution `-0.009127`
- `lag_10__CT_place_ARAMP`: contribution `+0.009087`
- `lag_00__T_place_EXTENDEDA`: contribution `-0.008818`
- `lag_07__T_place_SHORTSTAIRS`: contribution `-0.007532`
- `lag_02__T_place_EXTENDEDA`: contribution `-0.007431`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.009127`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.002300`

### tick `121450`, seconds `95.50`, LSTM delta `+0.0635`

Top all feature movements:
- `lag_00__T_place_EXTENDEDA`: contribution `+0.008818`
- `lag_05__T_shots_fired_sum`: contribution `+0.008486`
- `lag_14__CT_place_ARAMP`: contribution `-0.005261`
- `lag_00__CT4__duck_amount`: contribution `+0.005223`
- `lag_04__T2__is_scoped`: contribution `+0.005184`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120298`, seconds `77.50`, LSTM delta `-0.0458`

Top all feature movements:
- `lag_00__T5__flash_duration`: contribution `-0.007343`
- `lag_00__T3__duck_amount`: contribution `-0.003845`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.003638`
- `lag_07__T_place_CATWALK`: contribution `-0.003263`
- `lag_00__T5__is_walking`: contribution `+0.003098`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.007343`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.003638`
- `lag_14__CT_A_site_active_infernos`: contribution `-0.001238`
