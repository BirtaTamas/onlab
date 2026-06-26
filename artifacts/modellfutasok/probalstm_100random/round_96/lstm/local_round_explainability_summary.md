# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `116886`, seconds `69.00`, LSTM `0.1934`, delta `-0.2769`
- tick `115894`, seconds `53.50`, LSTM `0.6074`, delta `+0.2599`
- tick `115862`, seconds `53.00`, LSTM `0.3475`, delta `-0.2372`
- tick `115702`, seconds `50.50`, LSTM `0.5443`, delta `-0.2300`
- tick `115670`, seconds `50.00`, LSTM `0.7743`, delta `+0.1575`
- tick `116182`, seconds `58.00`, LSTM `0.4928`, delta `-0.1318`
- tick `115478`, seconds `47.00`, LSTM `0.6619`, delta `+0.1011`
- tick `115414`, seconds `46.00`, LSTM `0.5650`, delta `-0.0628`
- tick `115350`, seconds `45.00`, LSTM `0.5779`, delta `+0.0622`
- tick `116214`, seconds `58.50`, LSTM `0.4325`, delta `-0.0603`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002909`, |coef| `0.002909`
- `lag_00__T_kills_last_3s`: coefficient `-0.002706`, |coef| `0.002706`
- `lag_11__CT_place_LIBRARY`: coefficient `0.002194`, |coef| `0.002194`
- `lag_00__T_place_QUAD`: coefficient `0.002068`, |coef| `0.002068`
- `lag_14__CT_place_LIBRARY`: coefficient `0.002049`, |coef| `0.002049`
- `lag_10__T_bomb_zone_count`: coefficient `0.002009`, |coef| `0.002009`
- `lag_14__T2__duck_amount`: coefficient `-0.001989`, |coef| `0.001989`
- `lag_15__T_place_QUAD`: coefficient `-0.001845`, |coef| `0.001845`
- `lag_01__T2__duck_amount`: coefficient `0.001819`, |coef| `0.001819`
- `lag_05__T_place_QUAD`: coefficient `0.001762`, |coef| `0.001762`
- `lag_12__T_place_QUAD`: coefficient `-0.001734`, |coef| `0.001734`
- `lag_08__T2__duck_amount`: coefficient `-0.001655`, |coef| `0.001655`
- `lag_00__damage_diff_last_5s`: coefficient `0.001654`, |coef| `0.001654`
- `lag_10__T1__duck_amount`: coefficient `0.001630`, |coef| `0.001630`
- `lag_07__T_place_BALCONY`: coefficient `-0.001616`, |coef| `0.001616`

## Top 10 utility ridge features

- `lag_15__CT2__flash_duration`: coefficient `0.001177` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `0.000791` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000590` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.000587` (raises CT win probability)
- `lag_00__T_active_smokes`: coefficient `0.000581` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000483` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000457` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `0.000454` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `0.000418` (raises CT win probability)
- `lag_00__active_smokes_total`: coefficient `0.000391` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002909` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002706` (lowers CT win probability)
- `lag_11__CT_place_LIBRARY`: coefficient `0.002194` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.002068` (raises CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `0.002049` (raises CT win probability)
- `lag_10__T_bomb_zone_count`: coefficient `0.002009` (raises CT win probability)
- `lag_14__T2__duck_amount`: coefficient `-0.001989` (lowers CT win probability)
- `lag_15__T_place_QUAD`: coefficient `-0.001845` (lowers CT win probability)
- `lag_01__T2__duck_amount`: coefficient `0.001819` (raises CT win probability)
- `lag_05__T_place_QUAD`: coefficient `0.001762` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `116886`, seconds `69.00`, LSTM delta `-0.2769`

Top all feature movements:
- `lag_11__CT_place_LIBRARY`: contribution `-0.014065`
- `lag_14__CT_place_LIBRARY`: contribution `-0.013137`
- `lag_10__T_bomb_zone_count`: contribution `-0.011694`
- `lag_00__T_kills_last_3s`: contribution `-0.008572`
- `lag_14__T2__duck_amount`: contribution `-0.007605`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115894`, seconds `53.50`, LSTM delta `+0.2599`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `+0.044450`
- `lag_07__T_place_BALCONY`: contribution `+0.022222`
- `lag_08__T_place_BALCONY`: contribution `+0.018260`
- `lag_11__T_place_QUAD`: contribution `+0.016102`
- `lag_00__kill_diff_last_3s`: contribution `+0.014003`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `+0.002765`

### tick `115862`, seconds `53.00`, LSTM delta `-0.2372`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.044450`
- `lag_12__T_place_QUAD`: contribution `-0.041762`
- `lag_07__T_place_BALCONY`: contribution `-0.022222`
- `lag_14__T_place_QUAD`: contribution `-0.016016`
- `lag_00__kill_diff_last_3s`: contribution `-0.014003`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115702`, seconds `50.50`, LSTM delta `-0.2300`

Top all feature movements:
- `lag_05__T_place_QUAD`: contribution `-0.042447`
- `lag_09__T_place_QUAD`: contribution `-0.038838`
- `lag_07__T_place_QUAD`: contribution `-0.031154`
- `lag_02__T_place_BALCONY`: contribution `-0.014894`
- `lag_00__T_kills_last_3s`: contribution `-0.008572`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.002274`

### tick `115670`, seconds `50.00`, LSTM delta `+0.1575`

Top all feature movements:
- `lag_09__T_place_QUAD`: contribution `+0.038838`
- `lag_06__T_place_QUAD`: contribution `+0.028691`
- `lag_00__T_place_BALCONY`: contribution `+0.013375`
- `lag_00__kill_diff_last_3s`: contribution `+0.007001`
- `lag_01__T_place_BALCONY`: contribution `-0.006295`

Top utility-only movements:
- No utility movement among the top local contributors.
