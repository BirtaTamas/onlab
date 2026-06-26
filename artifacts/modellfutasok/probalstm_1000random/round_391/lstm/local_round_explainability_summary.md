# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `13`

## Largest probability jumps

- tick `116811`, seconds `64.00`, LSTM `0.5104`, delta `+0.2334`
- tick `117803`, seconds `79.50`, LSTM `0.2798`, delta `-0.1788`
- tick `117867`, seconds `80.50`, LSTM `0.0564`, delta `-0.1689`
- tick `117227`, seconds `70.50`, LSTM `0.4580`, delta `-0.1295`
- tick `116651`, seconds `61.50`, LSTM `0.3940`, delta `-0.1209`
- tick `116939`, seconds `66.00`, LSTM `0.5717`, delta `-0.0995`
- tick `116907`, seconds `65.50`, LSTM `0.6712`, delta `+0.0841`
- tick `117995`, seconds `82.50`, LSTM `0.0223`, delta `-0.0626`
- tick `117291`, seconds `71.50`, LSTM `0.3582`, delta `-0.0581`
- tick `117835`, seconds `80.00`, LSTM `0.2253`, delta `-0.0545`

## Top 15 local ridge features

- `lag_00__T5__flash_duration`: coefficient `-0.002320`, |coef| `0.002320`
- `lag_02__T_place_ELECTRICALBOX`: coefficient `0.002301`, |coef| `0.002301`
- `lag_07__CT_place_DUMPSTER`: coefficient `-0.001808`, |coef| `0.001808`
- `lag_00__CT_place_IVY`: coefficient `0.001558`, |coef| `0.001558`
- `lag_07__T_place_ELECTRICALBOX`: coefficient `-0.001520`, |coef| `0.001520`
- `lag_13__CT_place_DUMPSTER`: coefficient `-0.001505`, |coef| `0.001505`
- `lag_09__CT_place_DUMPSTER`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_15__T_place_ELECTRICALBOX`: coefficient `-0.001408`, |coef| `0.001408`
- `lag_00__damage_diff_last_5s`: coefficient `0.001373`, |coef| `0.001373`
- `lag_00__T_flash_duration_sum`: coefficient `-0.001333`, |coef| `0.001333`
- `lag_00__T_flashed_players`: coefficient `-0.001296`, |coef| `0.001296`
- `lag_09__CT5__flash_duration`: coefficient `0.001261`, |coef| `0.001261`
- `lag_05__CT5__flash_duration`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_04__CT5__flash_duration`: coefficient `-0.001248`, |coef| `0.001248`
- `lag_01__T5__flash_duration`: coefficient `-0.001239`, |coef| `0.001239`

## Top 10 utility ridge features

- `lag_00__T5__flash_duration`: coefficient `-0.002320` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001333` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.001261` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001259` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.001248` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.001239` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.001019` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.001015` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000959` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.000955` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_ELECTRICALBOX`: coefficient `0.002301` (raises CT win probability)
- `lag_07__CT_place_DUMPSTER`: coefficient `-0.001808` (lowers CT win probability)
- `lag_00__CT_place_IVY`: coefficient `0.001558` (raises CT win probability)
- `lag_07__T_place_ELECTRICALBOX`: coefficient `-0.001520` (lowers CT win probability)
- `lag_13__CT_place_DUMPSTER`: coefficient `-0.001505` (lowers CT win probability)
- `lag_09__CT_place_DUMPSTER`: coefficient `-0.001436` (lowers CT win probability)
- `lag_15__T_place_ELECTRICALBOX`: coefficient `-0.001408` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001373` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001296` (lowers CT win probability)
- `lag_03__CT_place_IVY`: coefficient `0.001237` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `116811`, seconds `64.00`, LSTM delta `+0.2334`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `+0.060407`
- `lag_00__T5__flash_duration`: contribution `+0.015267`
- `lag_09__CT5__flash_duration`: contribution `+0.007792`
- `lag_05__T5__flash_duration`: contribution `+0.006706`
- `lag_09__CT_flash_duration_sum`: contribution `+0.005122`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.015267`
- `lag_09__CT5__flash_duration`: contribution `+0.007792`
- `lag_05__T5__flash_duration`: contribution `+0.006706`
- `lag_09__CT_flash_duration_sum`: contribution `+0.005122`
- `lag_00__T_flash_duration_sum`: contribution `+0.005009`

### tick `117803`, seconds `79.50`, LSTM delta `-0.1788`

Top all feature movements:
- `lag_07__CT_place_DUMPSTER`: contribution `-0.093245`
- `lag_02__CT_place_DUMPSTER`: contribution `-0.054368`
- `lag_02__CT_place_TMAIN`: contribution `-0.013020`
- `lag_03__T_place_ELECTRICALBOX`: contribution `+0.007494`
- `lag_00__damage_diff_last_5s`: contribution `-0.003097`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117867`, seconds `80.50`, LSTM delta `-0.1689`

Top all feature movements:
- `lag_09__CT_place_DUMPSTER`: contribution `-0.074100`
- `lag_04__CT_place_DUMPSTER`: contribution `-0.050596`
- `lag_04__CT_place_TMAIN`: contribution `-0.010461`
- `lag_00__damage_diff_last_5s`: contribution `-0.003097`
- `lag_06__T_place_LONGDOG`: contribution `+0.002452`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117227`, seconds `70.50`, LSTM delta `-0.1295`

Top all feature movements:
- `lag_15__T_place_ELECTRICALBOX`: contribution `-0.036950`
- `lag_00__CT_place_IVY`: contribution `-0.017786`
- `lag_10__CT_place_IVY`: contribution `-0.012044`
- `lag_09__T_place_ELECTRICALBOX`: contribution `-0.009439`
- `lag_01__T_bomb_zone_count`: contribution `-0.004141`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.001897`

### tick `116651`, seconds `61.50`, LSTM delta `-0.1209`

Top all feature movements:
- `lag_00__T5__flash_duration`: contribution `-0.015267`
- `lag_04__CT5__flash_duration`: contribution `-0.007716`
- `lag_04__CT_place_BACKOFB`: contribution `-0.005169`
- `lag_00__T_flash_duration_sum`: contribution `-0.005009`
- `lag_00__T_flashed_players`: contribution `-0.005002`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.015267`
- `lag_04__CT5__flash_duration`: contribution `-0.007716`
- `lag_00__T_flash_duration_sum`: contribution `-0.005009`
- `lag_04__CT_flash_duration_sum`: contribution `-0.004217`
- `lag_04__CT1__flash_duration`: contribution `-0.002870`
