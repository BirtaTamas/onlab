# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `120936`, seconds `107.50`, LSTM `0.1556`, delta `-0.1803`
- tick `120360`, seconds `98.50`, LSTM `0.6032`, delta `+0.1689`
- tick `120424`, seconds `99.50`, LSTM `0.4425`, delta `-0.1678`
- tick `116744`, seconds `42.00`, LSTM `0.4774`, delta `-0.0813`
- tick `120712`, seconds `104.00`, LSTM `0.3157`, delta `-0.0694`
- tick `120808`, seconds `105.50`, LSTM `0.4012`, delta `+0.0659`
- tick `120680`, seconds `103.50`, LSTM `0.3851`, delta `-0.0638`
- tick `120840`, seconds `106.00`, LSTM `0.3395`, delta `-0.0617`
- tick `119944`, seconds `92.00`, LSTM `0.5322`, delta `-0.0540`
- tick `120552`, seconds `101.50`, LSTM `0.4092`, delta `-0.0503`

## Top 15 local ridge features

- `lag_15__T_place_QUAD`: coefficient `0.003019`, |coef| `0.003019`
- `lag_00__damage_diff_last_5s`: coefficient `0.002982`, |coef| `0.002982`
- `lag_00__kill_diff_last_3s`: coefficient `0.002838`, |coef| `0.002838`
- `lag_13__T_place_QUAD`: coefficient `-0.002766`, |coef| `0.002766`
- `lag_00__T_kills_last_3s`: coefficient `-0.002467`, |coef| `0.002467`
- `lag_07__T_place_BALCONY`: coefficient `-0.002438`, |coef| `0.002438`
- `lag_00__T_damage_last_5s`: coefficient `-0.002159`, |coef| `0.002159`
- `lag_04__T1__has_bomb`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_00__CT4__alive`: coefficient `0.001755`, |coef| `0.001755`
- `lag_10__T1__duck_amount`: coefficient `-0.001721`, |coef| `0.001721`
- `lag_12__CT_kills_last_3s`: coefficient `0.001713`, |coef| `0.001713`
- `lag_11__T_velocity_mean`: coefficient `0.001677`, |coef| `0.001677`
- `lag_08__CT_place_LIBRARY`: coefficient `0.001633`, |coef| `0.001633`
- `lag_04__bomb_events_last_5s`: coefficient `-0.001590`, |coef| `0.001590`
- `lag_03__T_bomb_zone_count`: coefficient `-0.001558`, |coef| `0.001558`

## Top 10 utility ridge features

- `lag_00__T3__flash_duration`: coefficient `-0.000624` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000395` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000353` (lowers CT win probability)
- `lag_02__CT2__smoke`: coefficient `0.000338` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000314` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000303` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000260` (lowers CT win probability)
- `lag_03__CT4__flash`: coefficient `0.000253` (raises CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.000221` (lowers CT win probability)
- `lag_06__CT4__flash`: coefficient `0.000207` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_QUAD`: coefficient `0.003019` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002982` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002838` (raises CT win probability)
- `lag_13__T_place_QUAD`: coefficient `-0.002766` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002467` (lowers CT win probability)
- `lag_07__T_place_BALCONY`: coefficient `-0.002438` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002159` (lowers CT win probability)
- `lag_04__T1__has_bomb`: coefficient `-0.002020` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001755` (raises CT win probability)
- `lag_10__T1__duck_amount`: coefficient `-0.001721` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `120936`, seconds `107.50`, LSTM delta `-0.1803`

Top all feature movements:
- `lag_07__T_place_BALCONY`: contribution `-0.033527`
- `lag_08__CT_place_LIBRARY`: contribution `-0.010470`
- `lag_00__T_kills_last_3s`: contribution `-0.007817`
- `lag_00__T_place_BALCONY`: contribution `-0.007677`
- `lag_00__kill_diff_last_3s`: contribution `-0.006832`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120360`, seconds `98.50`, LSTM delta `+0.1689`

Top all feature movements:
- `lag_13__T_place_QUAD`: contribution `+0.066621`
- `lag_00__kill_diff_last_3s`: contribution `+0.006832`
- `lag_00__damage_diff_last_5s`: contribution `+0.006726`
- `lag_06__T_duck_amount_mean`: contribution `+0.005838`
- `lag_06__T4__duck_amount`: contribution `+0.005417`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120424`, seconds `99.50`, LSTM delta `-0.1678`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.072719`
- `lag_00__T_kills_last_3s`: contribution `-0.007817`
- `lag_00__kill_diff_last_3s`: contribution `-0.006832`
- `lag_00__damage_diff_last_5s`: contribution `-0.006726`
- `lag_00__T_damage_last_5s`: contribution `-0.005177`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116744`, seconds `42.00`, LSTM delta `-0.0813`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007817`
- `lag_00__kill_diff_last_3s`: contribution `-0.006832`
- `lag_00__damage_diff_last_5s`: contribution `-0.006726`
- `lag_00__T2__duck_amount`: contribution `-0.005217`
- `lag_00__T_damage_last_5s`: contribution `-0.005177`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.002143`
- `lag_01__T3__flash_duration`: contribution `-0.001839`

### tick `120712`, seconds `104.00`, LSTM delta `-0.0694`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.007677`
- `lag_15__CT4__duck_amount`: contribution `-0.005207`
- `lag_03__CT_place_RUINS`: contribution `-0.004732`
- `lag_01__CT_place_LIBRARY`: contribution `-0.004570`
- `lag_12__T_kills_last_3s`: contribution `-0.003649`

Top utility-only movements:
- No utility movement among the top local contributors.
