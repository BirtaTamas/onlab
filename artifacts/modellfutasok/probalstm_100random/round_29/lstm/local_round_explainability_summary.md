# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `19338`, seconds `23.50`, LSTM `0.8530`, delta `+0.0873`
- tick `19306`, seconds `23.00`, LSTM `0.7657`, delta `+0.0495`
- tick `19914`, seconds `32.50`, LSTM `0.9599`, delta `+0.0488`
- tick `19178`, seconds `21.00`, LSTM `0.6580`, delta `+0.0436`
- tick `19882`, seconds `32.00`, LSTM `0.9112`, delta `+0.0359`
- tick `21066`, seconds `50.50`, LSTM `0.9721`, delta `+0.0294`
- tick `19210`, seconds `21.50`, LSTM `0.6834`, delta `+0.0254`
- tick `19530`, seconds `26.50`, LSTM `0.8384`, delta `-0.0228`
- tick `18186`, seconds `5.50`, LSTM `0.5717`, delta `+0.0221`
- tick `19050`, seconds `19.00`, LSTM `0.5925`, delta `+0.0212`

## Top 15 local ridge features

- `lag_02__T5__flash_duration`: coefficient `0.001175`, |coef| `0.001175`
- `lag_01__T5__flash_duration`: coefficient `0.001027`, |coef| `0.001027`
- `lag_00__CT_kills_last_3s`: coefficient `0.000873`, |coef| `0.000873`
- `lag_00__T5__flash_duration`: coefficient `0.000869`, |coef| `0.000869`
- `lag_00__kill_diff_last_3s`: coefficient `0.000794`, |coef| `0.000794`
- `lag_09__T_place_UPSTAIRS`: coefficient `0.000776`, |coef| `0.000776`
- `lag_03__T5__flash_duration`: coefficient `0.000752`, |coef| `0.000752`
- `lag_11__T5__flash_duration`: coefficient `0.000667`, |coef| `0.000667`
- `lag_12__T5__flash_duration`: coefficient `0.000664`, |coef| `0.000664`
- `lag_04__T5__flash_duration`: coefficient `0.000606`, |coef| `0.000606`
- `lag_07__T5__flash_duration`: coefficient `0.000592`, |coef| `0.000592`
- `lag_00__CT_damage_last_5s`: coefficient `0.000589`, |coef| `0.000589`
- `lag_13__T_shots_fired_sum`: coefficient `-0.000588`, |coef| `0.000588`
- `lag_01__CT_place_QUAD`: coefficient `-0.000586`, |coef| `0.000586`
- `lag_06__T5__flash_duration`: coefficient `0.000585`, |coef| `0.000585`

## Top 10 utility ridge features

- `lag_02__T5__flash_duration`: coefficient `0.001175` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001027` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000869` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000752` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000667` (raises CT win probability)
- `lag_12__T5__flash_duration`: coefficient `0.000664` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.000606` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.000592` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000585` (raises CT win probability)
- `lag_10__T5__flash_duration`: coefficient `0.000571` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.000873` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000794` (raises CT win probability)
- `lag_09__T_place_UPSTAIRS`: coefficient `0.000776` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000589` (raises CT win probability)
- `lag_13__T_shots_fired_sum`: coefficient `-0.000588` (lowers CT win probability)
- `lag_01__CT_place_QUAD`: coefficient `-0.000586` (lowers CT win probability)
- `lag_10__CT_place_PIT`: coefficient `0.000584` (raises CT win probability)
- `lag_08__T_place_UPSTAIRS`: coefficient `0.000570` (raises CT win probability)
- `lag_13__T3__shots_fired`: coefficient `-0.000569` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000565` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `19338`, seconds `23.50`, LSTM delta `+0.0873`

Top all feature movements:
- `lag_09__T_place_UPSTAIRS`: contribution `+0.013082`
- `lag_02__T5__flash_duration`: contribution `+0.007823`
- `lag_05__T_place_UPSTAIRS`: contribution `+0.007803`
- `lag_00__CT_kills_last_3s`: contribution `+0.002521`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002096`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.007823`

### tick `19306`, seconds `23.00`, LSTM delta `+0.0495`

Top all feature movements:
- `lag_08__T_place_UPSTAIRS`: contribution `+0.009617`
- `lag_01__T5__flash_duration`: contribution `+0.006841`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002096`
- `lag_04__T2__duck_amount`: contribution `+0.001486`
- `lag_05__CT_place_QUAD`: contribution `+0.001316`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `+0.006841`
- `lag_15__T_B_site_active_infernos`: contribution `+0.000884`
- `lag_01__T_B_site_active_infernos`: contribution `+0.000616`

### tick `19914`, seconds `32.50`, LSTM delta `+0.0488`

Top all feature movements:
- `lag_14__T_shots_fired_sum`: contribution `+0.004210`
- `lag_14__T3__shots_fired`: contribution `+0.003076`
- `lag_00__CT_kills_last_3s`: contribution `+0.002521`
- `lag_10__CT_place_PIT`: contribution `+0.002516`
- `lag_00__kill_diff_last_3s`: contribution `+0.001911`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `+0.000915`

### tick `19178`, seconds `21.00`, LSTM delta `+0.0436`

Top all feature movements:
- `lag_01__CT_place_QUAD`: contribution `+0.004622`
- `lag_00__CT_kills_last_3s`: contribution `+0.002521`
- `lag_00__kill_diff_last_3s`: contribution `+0.001911`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001797`
- `lag_00__T2__duck_amount`: contribution `+0.001776`

Top utility-only movements:
- `lag_00__T5__smoke`: contribution `+0.000804`

### tick `19882`, seconds `32.00`, LSTM delta `+0.0359`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `+0.006611`
- `lag_13__T3__shots_fired`: contribution `+0.005169`
- `lag_09__CT_place_PIT`: contribution `+0.002137`
- `lag_08__CT_place_BANANA`: contribution `+0.001565`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001497`

Top utility-only movements:
- `lag_14__CT2__flash_duration`: contribution `+0.000824`
