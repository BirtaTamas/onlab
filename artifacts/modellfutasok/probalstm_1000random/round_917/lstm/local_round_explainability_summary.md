# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `1`

## Largest probability jumps

- tick `2177`, seconds `34.00`, LSTM `0.4143`, delta `+0.2130`
- tick `2017`, seconds `31.50`, LSTM `0.3378`, delta `-0.1896`
- tick `2113`, seconds `33.00`, LSTM `0.2448`, delta `-0.1642`
- tick `2785`, seconds `43.50`, LSTM `0.6809`, delta `+0.1635`
- tick `3041`, seconds `47.50`, LSTM `0.8698`, delta `+0.1370`
- tick `2049`, seconds `32.00`, LSTM `0.4239`, delta `+0.0861`
- tick `4481`, seconds `70.00`, LSTM `0.9415`, delta `+0.0451`
- tick `2145`, seconds `33.50`, LSTM `0.2013`, delta `-0.0435`
- tick `3073`, seconds `48.00`, LSTM `0.9031`, delta `+0.0333`
- tick `2817`, seconds `44.00`, LSTM `0.7117`, delta `+0.0308`

## Top 15 local ridge features

- `lag_05__CT_place_TSIDELOWER`: coefficient `-0.004128`, |coef| `0.004128`
- `lag_00__kill_diff_last_3s`: coefficient `0.002936`, |coef| `0.002936`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002520`, |coef| `0.002520`
- `lag_10__T3__duck_amount`: coefficient `-0.002431`, |coef| `0.002431`
- `lag_00__CT_kills_last_3s`: coefficient `0.002375`, |coef| `0.002375`
- `lag_00__T_place_ALLEY`: coefficient `-0.002361`, |coef| `0.002361`
- `lag_02__CT_place_TSIDELOWER`: coefficient `-0.002140`, |coef| `0.002140`
- `lag_00__damage_diff_last_5s`: coefficient `0.002027`, |coef| `0.002027`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001950`, |coef| `0.001950`
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001892`, |coef| `0.001892`
- `lag_04__T1__flash_duration`: coefficient `-0.001691`, |coef| `0.001691`
- `lag_11__T_place_ALLEY`: coefficient `0.001505`, |coef| `0.001505`
- `lag_11__T_place_HOUSE`: coefficient `-0.001498`, |coef| `0.001498`
- `lag_00__CT_defusing_count`: coefficient `0.001490`, |coef| `0.001490`
- `lag_06__CT_place_TSIDELOWER`: coefficient `-0.001388`, |coef| `0.001388`

## Top 10 utility ridge features

- `lag_04__T1__flash_duration`: coefficient `-0.001691` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.001010` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.000908` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000868` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000836` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.000801` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.000771` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000765` (lowers CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000644` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000586` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_TSIDELOWER`: coefficient `-0.004128` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002936` (raises CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.002520` (raises CT win probability)
- `lag_10__T3__duck_amount`: coefficient `-0.002431` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002375` (raises CT win probability)
- `lag_00__T_place_ALLEY`: coefficient `-0.002361` (lowers CT win probability)
- `lag_02__CT_place_TSIDELOWER`: coefficient `-0.002140` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002027` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001950` (raises CT win probability)
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.001892` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2177`, seconds `34.00`, LSTM delta `+0.2130`

Top all feature movements:
- `lag_05__CT_place_TSIDELOWER`: contribution `+0.056078`
- `lag_02__CT_place_TSIDELOWER`: contribution `+0.029072`
- `lag_08__CT_place_TSIDELOWER`: contribution `-0.013089`
- `lag_10__CT_place_TSIDELOWER`: contribution `+0.013020`
- `lag_10__T3__duck_amount`: contribution `+0.009164`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.008445`
- `lag_09__T1__flash_duration`: contribution `+0.002612`

### tick `2017`, seconds `31.50`, LSTM delta `-0.1896`

Top all feature movements:
- `lag_05__CT_place_TSIDELOWER`: contribution `-0.056078`
- `lag_00__CT_place_TSIDELOWER`: contribution `-0.034238`
- `lag_10__T3__duck_amount`: contribution `-0.009164`
- `lag_04__T1__flash_duration`: contribution `-0.008445`
- `lag_00__kill_diff_last_3s`: contribution `-0.007067`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.008445`
- `lag_04__T_flash_alpha_mean`: contribution `-0.001944`

### tick `2113`, seconds `33.00`, LSTM delta `-0.1642`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `-0.034238`
- `lag_06__CT_place_TSIDELOWER`: contribution `-0.018858`
- `lag_08__CT_place_TSIDELOWER`: contribution `-0.013089`
- `lag_00__kill_diff_last_3s`: contribution `-0.007067`
- `lag_02__T1__flash_duration`: contribution `-0.005047`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.005047`
- `lag_07__T1__flash_duration`: contribution `-0.003850`
- `lag_07__T_flash_alpha_mean`: contribution `-0.002107`

### tick `2785`, seconds `43.50`, LSTM delta `+0.1635`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.014658`
- `lag_00__T_place_ALLEY`: contribution `+0.010004`
- `lag_12__CT_place_TSIDEUPPER`: contribution `+0.007622`
- `lag_00__kill_diff_last_3s`: contribution `+0.007067`
- `lag_00__CT_kills_last_3s`: contribution `+0.006857`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3041`, seconds `47.50`, LSTM delta `+0.1370`

Top all feature movements:
- `lag_04__CT_place_TSIDEUPPER`: contribution `+0.014220`
- `lag_00__T_place_ALLEY`: contribution `+0.010004`
- `lag_10__T3__duck_amount`: contribution `+0.009164`
- `lag_00__kill_diff_last_3s`: contribution `+0.007067`
- `lag_00__CT_kills_last_3s`: contribution `+0.006857`

Top utility-only movements:
- No utility movement among the top local contributors.
