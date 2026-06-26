# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-3dmax-vs-lynn-vision-bo3-0ZNMTlQ0ZfadRgwA0Ax5fN/3dmax-vs-lynn-vision-m2-anubis.csv`
- round_num: `3`

## Largest probability jumps

- tick `22703`, seconds `0.50`, LSTM `0.0224`, delta `-0.0314`
- tick `26927`, seconds `66.50`, LSTM `0.0308`, delta `-0.0113`
- tick `26959`, seconds `67.00`, LSTM `0.0226`, delta `-0.0082`
- tick `26319`, seconds `57.00`, LSTM `0.0221`, delta `+0.0068`
- tick `27215`, seconds `71.00`, LSTM `0.0229`, delta `+0.0064`
- tick `26287`, seconds `56.50`, LSTM `0.0153`, delta `-0.0061`
- tick `26831`, seconds `65.00`, LSTM `0.0388`, delta `+0.0058`
- tick `23855`, seconds `18.50`, LSTM `0.0188`, delta `+0.0057`
- tick `29647`, seconds `109.00`, LSTM `0.0055`, delta `-0.0054`
- tick `23887`, seconds `19.00`, LSTM `0.0241`, delta `+0.0054`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000563`, |coef| `0.000563`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000273`, |coef| `0.000273`
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000168`, |coef| `0.000168`
- `lag_00__CT_place_WALKWAY`: coefficient `-0.000163`, |coef| `0.000163`
- `lag_00__CT_place_HEAVEN`: coefficient `0.000161`, |coef| `0.000161`
- `lag_01__CT_place_BRIDGE`: coefficient `0.000161`, |coef| `0.000161`
- `lag_00__T_velocity_mean`: coefficient `-0.000152`, |coef| `0.000152`
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000143`, |coef| `0.000143`
- `lag_00__CT_smokes_last_5s`: coefficient `0.000135`, |coef| `0.000135`
- `lag_00__CT_velocity_mean`: coefficient `-0.000134`, |coef| `0.000134`
- `lag_01__CT_place_HEAVEN`: coefficient `0.000121`, |coef| `0.000121`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000119`, |coef| `0.000119`
- `lag_01__CT_place_WALKWAY`: coefficient `-0.000115`, |coef| `0.000115`
- `lag_15__CT_place_BRIDGE`: coefficient `0.000110`, |coef| `0.000110`
- `lag_00__CT5__is_walking`: coefficient `0.000105`, |coef| `0.000105`

## Top 10 utility ridge features

- `lag_01__CT_flash_alpha_mean`: coefficient `0.000168` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000135` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000105` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000099` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.000087` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000087` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000086` (raises CT win probability)
- `lag_02__CT_flash_alpha_mean`: coefficient `0.000083` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000082` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000076` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000563` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000273` (raises CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `-0.000163` (lowers CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.000161` (raises CT win probability)
- `lag_01__CT_place_BRIDGE`: coefficient `0.000161` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000152` (lowers CT win probability)
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000143` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000134` (lowers CT win probability)
- `lag_01__CT_place_HEAVEN`: coefficient `0.000121` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000119` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22703`, seconds `0.50`, LSTM delta `-0.0314`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.014507`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.000735`
- `lag_00__T_velocity_mean`: contribution `-0.000563`
- `lag_01__T_place_TSPAWN`: contribution `-0.000526`
- `lag_00__CT_velocity_mean`: contribution `-0.000442`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.000735`
- `lag_01__utility_inv_diff`: contribution `-0.000371`
- `lag_01__smoke_inv_diff`: contribution `-0.000314`
- `lag_01__molly_inv_diff`: contribution `-0.000240`
- `lag_01__flash_inv_diff`: contribution `-0.000203`

### tick `26927`, seconds `66.50`, LSTM delta `-0.0113`

Top all feature movements:
- `lag_01__CT_place_BRIDGE`: contribution `-0.001849`
- `lag_00__CT_place_HEAVEN`: contribution `-0.000872`
- `lag_00__CT_place_WALKWAY`: contribution `-0.000803`
- `lag_10__T_place_MAIN`: contribution `-0.000583`
- `lag_14__T_place_MAIN`: contribution `-0.000577`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.000143`

### tick `26959`, seconds `67.00`, LSTM delta `-0.0082`

Top all feature movements:
- `lag_00__CT_place_BRIDGE`: contribution `-0.000756`
- `lag_01__CT_place_HEAVEN`: contribution `-0.000654`
- `lag_01__CT_place_WALKWAY`: contribution `-0.000564`
- `lag_11__T_place_MAIN`: contribution `-0.000533`
- `lag_15__T_place_MAIN`: contribution `-0.000531`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.000106`

### tick `26319`, seconds `57.00`, LSTM delta `+0.0068`

Top all feature movements:
- `lag_01__CT_place_BRIDGE`: contribution `+0.001849`
- `lag_00__CT_place_BRIDGE`: contribution `+0.001512`
- `lag_00__T1__duck_amount`: contribution `+0.000319`
- `lag_04__T5__duck_amount`: contribution `+0.000235`
- `lag_00__CT_place_MIDDLE`: contribution `+0.000234`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27215`, seconds `71.00`, LSTM delta `+0.0064`

Top all feature movements:
- `lag_01__CT_place_BRIDGE`: contribution `+0.001849`
- `lag_06__CT_place_BRIDGE`: contribution `+0.001284`
- `lag_00__CT_place_BRIDGE`: contribution `+0.000756`
- `lag_00__CT_place_RUINS`: contribution `+0.000692`
- `lag_06__CT_place_HEAVEN`: contribution `+0.000355`

Top utility-only movements:
- No utility movement among the top local contributors.
