# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `18210`, seconds `40.50`, LSTM `0.8890`, delta `+0.1455`
- tick `17922`, seconds `36.00`, LSTM `0.6333`, delta `+0.0913`
- tick `19074`, seconds `54.00`, LSTM `0.9535`, delta `+0.0818`
- tick `18402`, seconds `43.50`, LSTM `0.9388`, delta `+0.0708`
- tick `16578`, seconds `15.00`, LSTM `0.5955`, delta `-0.0623`
- tick `17666`, seconds `32.00`, LSTM `0.5051`, delta `-0.0560`
- tick `18114`, seconds `39.00`, LSTM `0.7394`, delta `+0.0539`
- tick `16994`, seconds `21.50`, LSTM `0.5996`, delta `-0.0476`
- tick `18082`, seconds `38.50`, LSTM `0.6856`, delta `+0.0430`
- tick `16258`, seconds `10.00`, LSTM `0.6439`, delta `+0.0379`

## Top 15 local ridge features

- `lag_00__CT_place_SCAFFOLDING`: coefficient `0.002170`, |coef| `0.002170`
- `lag_00__CT_place_LADDER`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_00__CT_place_TRUCK`: coefficient `0.001245`, |coef| `0.001245`
- `lag_03__T_place_TRUCK`: coefficient `0.001224`, |coef| `0.001224`
- `lag_15__CT_place_JUNGLE`: coefficient `0.001092`, |coef| `0.001092`
- `lag_08__CT_place_LADDER`: coefficient `0.001064`, |coef| `0.001064`
- `lag_03__CT_place_LADDER`: coefficient `-0.001010`, |coef| `0.001010`
- `lag_00__CT_place_CATWALK`: coefficient `0.000993`, |coef| `0.000993`
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.000990`, |coef| `0.000990`
- `lag_00__CT_kills_last_3s`: coefficient `0.000943`, |coef| `0.000943`
- `lag_13__T4__flash_duration`: coefficient `-0.000938`, |coef| `0.000938`
- `lag_15__T_place_UNDERPASS`: coefficient `-0.000913`, |coef| `0.000913`
- `lag_01__CT1__is_scoped`: coefficient `0.000888`, |coef| `0.000888`
- `lag_00__CT_damage_last_5s`: coefficient `0.000887`, |coef| `0.000887`
- `lag_10__T4__flash_duration`: coefficient `-0.000840`, |coef| `0.000840`

## Top 10 utility ridge features

- `lag_13__T4__flash_duration`: coefficient `-0.000938` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.000840` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000659` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000489` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000488` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.000463` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000461` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000458` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.000447` (raises CT win probability)
- `lag_14__CT2__smoke`: coefficient `-0.000437` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SCAFFOLDING`: coefficient `0.002170` (raises CT win probability)
- `lag_00__CT_place_LADDER`: coefficient `-0.001712` (lowers CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001245` (raises CT win probability)
- `lag_03__T_place_TRUCK`: coefficient `0.001224` (raises CT win probability)
- `lag_15__CT_place_JUNGLE`: coefficient `0.001092` (raises CT win probability)
- `lag_08__CT_place_LADDER`: coefficient `0.001064` (raises CT win probability)
- `lag_03__CT_place_LADDER`: coefficient `-0.001010` (lowers CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `0.000993` (raises CT win probability)
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.000990` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000943` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18210`, seconds `40.50`, LSTM delta `+0.1455`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.045278`
- `lag_03__CT_place_LADDER`: contribution `+0.010503`
- `lag_15__CT_place_JUNGLE`: contribution `+0.007009`
- `lag_06__CT_place_JUNGLE`: contribution `+0.004006`
- `lag_01__CT1__is_scoped`: contribution `+0.003801`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.001680`

### tick `17922`, seconds `36.00`, LSTM delta `+0.0913`

Top all feature movements:
- `lag_08__CT_place_LADDER`: contribution `+0.011063`
- `lag_00__CT_place_TRUCK`: contribution `-0.008031`
- `lag_13__T4__flash_duration`: contribution `+0.007577`
- `lag_06__CT_place_TRUCK`: contribution `+0.005254`
- `lag_06__CT_place_JUNGLE`: contribution `-0.004006`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.007577`

### tick `19074`, seconds `54.00`, LSTM delta `+0.0818`

Top all feature movements:
- `lag_03__T_place_TRUCK`: contribution `+0.021259`
- `lag_00__CT_place_CATWALK`: contribution `+0.003956`
- `lag_00__CT_kills_last_3s`: contribution `+0.002723`
- `lag_08__T_shots_fired_sum`: contribution `+0.002578`
- `lag_00__damage_diff_last_5s`: contribution `+0.002201`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18402`, seconds `43.50`, LSTM delta `+0.0708`

Top all feature movements:
- `lag_06__CT_place_SCAFFOLDING`: contribution `+0.020658`
- `lag_02__CT_place_SCAFFOLDING`: contribution `+0.013086`
- `lag_01__CT1__is_scoped`: contribution `-0.003801`
- `lag_09__CT_place_LADDER`: contribution `-0.003732`
- `lag_12__CT_place_JUNGLE`: contribution `-0.002285`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.001121`
- `lag_00__T3__flash`: contribution `+0.001005`
- `lag_08__CT_active_smokes`: contribution `+0.001002`

### tick `16578`, seconds `15.00`, LSTM delta `-0.0623`

Top all feature movements:
- `lag_10__CT_place_LADDER`: contribution `-0.005503`
- `lag_06__CT_place_TRUCK`: contribution `-0.005254`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.004372`
- `lag_14__CT_place_LADDER`: contribution `+0.002815`
- `lag_00__T2__duck_amount`: contribution `-0.002311`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `-0.001321`
