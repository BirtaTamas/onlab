# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m3-ancient.csv`
- round_num: `5`

## Largest probability jumps

- tick `27780`, seconds `62.00`, LSTM `0.8112`, delta `+0.2624`
- tick `27812`, seconds `62.50`, LSTM `0.5740`, delta `-0.2372`
- tick `27908`, seconds `64.00`, LSTM `0.4149`, delta `-0.1591`
- tick `25156`, seconds `21.00`, LSTM `0.6475`, delta `-0.1515`
- tick `28292`, seconds `70.00`, LSTM `0.1256`, delta `-0.1447`
- tick `24964`, seconds `18.00`, LSTM `0.8141`, delta `+0.0858`
- tick `28228`, seconds `69.00`, LSTM `0.2910`, delta `+0.0581`
- tick `25764`, seconds `30.50`, LSTM `0.7617`, delta `-0.0533`
- tick `25412`, seconds `25.00`, LSTM `0.7388`, delta `+0.0521`
- tick `28132`, seconds `67.50`, LSTM `0.3179`, delta `-0.0492`

## Top 15 local ridge features

- `lag_00__T_place_RAMP`: coefficient `-0.003392`, |coef| `0.003392`
- `lag_00__kill_diff_last_3s`: coefficient `0.003233`, |coef| `0.003233`
- `lag_05__T_place_RUINS`: coefficient `-0.002863`, |coef| `0.002863`
- `lag_05__T_place_TSIDELOWER`: coefficient `0.002754`, |coef| `0.002754`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002726`, |coef| `0.002726`
- `lag_00__T3__flash_duration`: coefficient `-0.002592`, |coef| `0.002592`
- `lag_00__damage_diff_last_5s`: coefficient `0.002512`, |coef| `0.002512`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002379`, |coef| `0.002379`
- `lag_00__CT_kills_last_3s`: coefficient `0.002185`, |coef| `0.002185`
- `lag_03__T_place_TSIDELOWER`: coefficient `-0.001963`, |coef| `0.001963`
- `lag_10__CT2__is_walking`: coefficient `0.001948`, |coef| `0.001948`
- `lag_10__T5__is_walking`: coefficient `0.001860`, |coef| `0.001860`
- `lag_00__T_kills_last_3s`: coefficient `-0.001859`, |coef| `0.001859`
- `lag_00__CT2__flash_duration`: coefficient `-0.001824`, |coef| `0.001824`
- `lag_13__CT2__is_walking`: coefficient `0.001821`, |coef| `0.001821`

## Top 10 utility ridge features

- `lag_00__T3__flash_duration`: coefficient `-0.002592` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001824` (lowers CT win probability)
- `lag_15__T3__flash_duration`: coefficient `-0.001574` (lowers CT win probability)
- `lag_09__T2__smoke`: coefficient `-0.001508` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001507` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.001469` (lowers CT win probability)
- `lag_05__T2__smoke`: coefficient `-0.001412` (lowers CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.001206` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001153` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.001143` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_RAMP`: coefficient `-0.003392` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003233` (raises CT win probability)
- `lag_05__T_place_RUINS`: coefficient `-0.002863` (lowers CT win probability)
- `lag_05__T_place_TSIDELOWER`: coefficient `0.002754` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002726` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002512` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002379` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002185` (raises CT win probability)
- `lag_03__T_place_TSIDELOWER`: coefficient `-0.001963` (lowers CT win probability)
- `lag_10__CT2__is_walking`: coefficient `0.001948` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `27780`, seconds `62.00`, LSTM delta `+0.2624`

Top all feature movements:
- `lag_00__T_place_RAMP`: contribution `+0.023990`
- `lag_05__T_place_RUINS`: contribution `+0.015227`
- `lag_05__T_place_TSIDELOWER`: contribution `+0.010323`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008262`
- `lag_00__kill_diff_last_3s`: contribution `+0.007783`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27812`, seconds `62.50`, LSTM delta `-0.2372`

Top all feature movements:
- `lag_00__T3__flash_duration`: contribution `-0.017865`
- `lag_00__T5__flash_duration`: contribution `-0.010448`
- `lag_05__T_place_TSIDELOWER`: contribution `-0.010323`
- `lag_00__CT2__flash_duration`: contribution `-0.008850`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008262`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.017865`
- `lag_00__T5__flash_duration`: contribution `-0.010448`
- `lag_00__CT2__flash_duration`: contribution `-0.008850`

### tick `27908`, seconds `64.00`, LSTM delta `-0.1591`

Top all feature movements:
- `lag_00__T_place_RAMP`: contribution `-0.011995`
- `lag_03__T3__flash_duration`: contribution `-0.010128`
- `lag_04__T_place_RAMP`: contribution `-0.007832`
- `lag_00__kill_diff_last_3s`: contribution `-0.007783`
- `lag_00__T_kills_last_3s`: contribution `-0.005889`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.010128`
- `lag_03__CT2__flash_duration`: contribution `-0.004832`
- `lag_00__T5__flash_duration`: contribution `+0.003358`
- `lag_09__T2__smoke`: contribution `+0.003311`

### tick `25156`, seconds `21.00`, LSTM delta `-0.1515`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.015566`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.007015`
- `lag_00__CT_kills_last_3s`: contribution `-0.006307`
- `lag_00__T_kills_last_3s`: contribution `-0.005889`
- `lag_10__CT5__flash_duration`: contribution `-0.005366`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `-0.007015`
- `lag_10__CT5__flash_duration`: contribution `-0.005366`
- `lag_11__CT1__flash_duration`: contribution `-0.004446`
- `lag_03__CT_active_infernos`: contribution `-0.003797`
- `lag_01__CT5__flash_duration`: contribution `-0.003247`

### tick `28292`, seconds `70.00`, LSTM delta `-0.1447`

Top all feature movements:
- `lag_15__T3__flash_duration`: contribution `-0.010853`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008262`
- `lag_00__kill_diff_last_3s`: contribution `-0.007783`
- `lag_00__T_kills_last_3s`: contribution `-0.005889`
- `lag_15__CT2__flash_duration`: contribution `-0.005546`

Top utility-only movements:
- `lag_15__T3__flash_duration`: contribution `-0.010853`
- `lag_15__CT2__flash_duration`: contribution `-0.005546`
- `lag_15__T5__flash_duration`: contribution `-0.004903`
- `lag_04__T1__flash_duration`: contribution `-0.001846`
