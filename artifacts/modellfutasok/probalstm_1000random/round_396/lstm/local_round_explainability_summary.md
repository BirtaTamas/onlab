# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `9658`, seconds `34.00`, LSTM `0.4836`, delta `-0.2154`
- tick `9914`, seconds `38.00`, LSTM `0.5735`, delta `+0.2099`
- tick `10138`, seconds `41.50`, LSTM `0.5467`, delta `-0.1651`
- tick `11162`, seconds `57.50`, LSTM `0.3408`, delta `-0.1607`
- tick `11194`, seconds `58.00`, LSTM `0.1967`, delta `-0.1441`
- tick `9594`, seconds `33.00`, LSTM `0.6508`, delta `+0.1156`
- tick `11226`, seconds `58.50`, LSTM `0.0933`, delta `-0.1034`
- tick `11386`, seconds `61.00`, LSTM `0.1241`, delta `+0.0826`
- tick `8890`, seconds `22.00`, LSTM `0.5590`, delta `-0.0820`
- tick `11642`, seconds `65.00`, LSTM `0.2183`, delta `+0.0776`

## Top 15 local ridge features

- `lag_08__CT_place_ARCH`: coefficient `0.003117`, |coef| `0.003117`
- `lag_00__T_kills_last_3s`: coefficient `-0.003039`, |coef| `0.003039`
- `lag_00__kill_diff_last_3s`: coefficient `0.002916`, |coef| `0.002916`
- `lag_00__damage_diff_last_5s`: coefficient `0.002763`, |coef| `0.002763`
- `lag_07__CT_place_ARCH`: coefficient `0.002689`, |coef| `0.002689`
- `lag_09__CT_place_ARCH`: coefficient `0.002415`, |coef| `0.002415`
- `lag_00__T_damage_last_5s`: coefficient `-0.002413`, |coef| `0.002413`
- `lag_12__CT_place_LIBRARY`: coefficient `0.002388`, |coef| `0.002388`
- `lag_07__T_place_DECK`: coefficient `0.002359`, |coef| `0.002359`
- `lag_01__T_kills_last_3s`: coefficient `-0.002221`, |coef| `0.002221`
- `lag_12__T_place_BALCONY`: coefficient `0.002198`, |coef| `0.002198`
- `lag_13__CT_place_LIBRARY`: coefficient `0.002177`, |coef| `0.002177`
- `lag_15__T_place_BALCONY`: coefficient `0.001879`, |coef| `0.001879`
- `lag_01__kill_diff_last_3s`: coefficient `0.001845`, |coef| `0.001845`
- `lag_01__T_damage_last_5s`: coefficient `-0.001719`, |coef| `0.001719`

## Top 10 utility ridge features

- `lag_00__CT_flash_alpha_mean`: coefficient `-0.001052` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000878` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `0.000797` (raises CT win probability)
- `lag_01__T_A_site_active_smokes`: coefficient `0.000745` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.000599` (raises CT win probability)
- `lag_00__T_active_smokes`: coefficient `0.000586` (raises CT win probability)
- `lag_02__T_A_site_active_smokes`: coefficient `0.000560` (raises CT win probability)
- `lag_01__T_active_smokes`: coefficient `0.000553` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `0.000530` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.000494` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_ARCH`: coefficient `0.003117` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003039` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002916` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002763` (raises CT win probability)
- `lag_07__CT_place_ARCH`: coefficient `0.002689` (raises CT win probability)
- `lag_09__CT_place_ARCH`: coefficient `0.002415` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002413` (lowers CT win probability)
- `lag_12__CT_place_LIBRARY`: coefficient `0.002388` (raises CT win probability)
- `lag_07__T_place_DECK`: coefficient `0.002359` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.002221` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9658`, seconds `34.00`, LSTM delta `-0.2154`

Top all feature movements:
- `lag_07__T_place_DECK`: contribution `-0.057215`
- `lag_03__T_place_BALCONY`: contribution `-0.018247`
- `lag_00__T_place_BALCONY`: contribution `-0.016479`
- `lag_00__T_kills_last_3s`: contribution `-0.009628`
- `lag_04__T_place_BALCONY`: contribution `-0.009036`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9914`, seconds `38.00`, LSTM delta `+0.2099`

Top all feature movements:
- `lag_15__T_place_DECK`: contribution `+0.033052`
- `lag_12__T_place_BALCONY`: contribution `+0.030232`
- `lag_11__T_place_BALCONY`: contribution `+0.018979`
- `lag_14__T_place_BALCONY`: contribution `+0.016833`
- `lag_05__T_place_BALCONY`: contribution `-0.008826`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10138`, seconds `41.50`, LSTM delta `-0.1651`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `-0.030232`
- `lag_15__T_place_BALCONY`: contribution `-0.025838`
- `lag_01__T_place_BALCONY`: contribution `+0.015718`
- `lag_07__CT_place_ARCH`: contribution `-0.010971`
- `lag_00__T_kills_last_3s`: contribution `-0.009628`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11162`, seconds `57.50`, LSTM delta `-0.1607`

Top all feature movements:
- `lag_12__CT_place_LIBRARY`: contribution `-0.015309`
- `lag_08__CT_place_ARCH`: contribution `-0.012718`
- `lag_07__CT_place_ARCH`: contribution `-0.010971`
- `lag_00__T_kills_last_3s`: contribution `-0.009628`
- `lag_00__kill_diff_last_3s`: contribution `-0.007018`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11194`, seconds `58.00`, LSTM delta `-0.1441`

Top all feature movements:
- `lag_13__CT_place_LIBRARY`: contribution `-0.013961`
- `lag_08__CT_place_ARCH`: contribution `-0.012718`
- `lag_09__CT_place_ARCH`: contribution `-0.009855`
- `lag_01__T_kills_last_3s`: contribution `-0.007035`
- `lag_01__CT_duck_amount_mean`: contribution `-0.004768`

Top utility-only movements:
- No utility movement among the top local contributors.
