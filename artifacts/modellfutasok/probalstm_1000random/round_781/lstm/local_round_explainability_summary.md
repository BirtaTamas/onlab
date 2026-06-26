# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `9`

## Largest probability jumps

- tick `65589`, seconds `0.50`, LSTM `0.1231`, delta `-0.0655`
- tick `66869`, seconds `20.50`, LSTM `0.0529`, delta `-0.0317`
- tick `67989`, seconds `38.00`, LSTM `0.0151`, delta `-0.0292`
- tick `67925`, seconds `37.00`, LSTM `0.0429`, delta `-0.0289`
- tick `66325`, seconds `12.00`, LSTM `0.1033`, delta `-0.0284`
- tick `66517`, seconds `15.00`, LSTM `0.1204`, delta `+0.0273`
- tick `66837`, seconds `20.00`, LSTM `0.0847`, delta `-0.0254`
- tick `66613`, seconds `16.50`, LSTM `0.0997`, delta `-0.0217`
- tick `65973`, seconds `6.50`, LSTM `0.1150`, delta `-0.0208`
- tick `65749`, seconds `3.00`, LSTM `0.1326`, delta `+0.0173`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000607`, |coef| `0.000607`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000531`, |coef| `0.000531`
- `lag_01__T_money_sum`: coefficient `-0.000530`, |coef| `0.000530`
- `lag_01__T_start_balance_sum`: coefficient `-0.000527`, |coef| `0.000527`
- `lag_00__CT_velocity_mean`: coefficient `-0.000503`, |coef| `0.000503`
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `0.000478`, |coef| `0.000478`
- `lag_00__T_velocity_mean`: coefficient `-0.000460`, |coef| `0.000460`
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000428`, |coef| `0.000428`
- `lag_00__T_kills_last_3s`: coefficient `-0.000397`, |coef| `0.000397`
- `lag_01__money_diff`: coefficient `0.000392`, |coef| `0.000392`
- `lag_01__T_shots_fired_sum`: coefficient `-0.000380`, |coef| `0.000380`
- `lag_01__flash_inv_diff`: coefficient `0.000378`, |coef| `0.000378`
- `lag_01__utility_inv_diff`: coefficient `0.000375`, |coef| `0.000375`
- `lag_01__CT_walking_count`: coefficient `0.000374`, |coef| `0.000374`
- `lag_10__CT_place_LOWERTUNNEL`: coefficient `-0.000366`, |coef| `0.000366`

## Top 10 utility ridge features

- `lag_01__flash_inv_diff`: coefficient `0.000378` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000375` (raises CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000312` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000278` (raises CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000277` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000269` (lowers CT win probability)
- `lag_01__T_flash_inv`: coefficient `-0.000266` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000264` (lowers CT win probability)
- `lag_01__T3__flash`: coefficient `-0.000256` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000235` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000607` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000531` (lowers CT win probability)
- `lag_01__T_money_sum`: coefficient `-0.000530` (lowers CT win probability)
- `lag_01__T_start_balance_sum`: coefficient `-0.000527` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000503` (lowers CT win probability)
- `lag_05__CT_place_LOWERTUNNEL`: coefficient `0.000478` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000460` (lowers CT win probability)
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000428` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000397` (lowers CT win probability)
- `lag_01__money_diff`: coefficient `0.000392` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `65589`, seconds `0.50`, LSTM delta `-0.0655`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002901`
- `lag_01__T_place_TSPAWN`: contribution `-0.002352`
- `lag_01__T_money_sum`: contribution `-0.002005`
- `lag_01__T_start_balance_sum`: contribution `-0.001992`
- `lag_00__CT_velocity_mean`: contribution `-0.001752`

Top utility-only movements:
- `lag_01__flash_inv_diff`: contribution `-0.001168`
- `lag_01__utility_inv_diff`: contribution `-0.001154`
- `lag_01__T4__utility_total`: contribution `-0.000690`
- `lag_01__T_flash_inv`: contribution `-0.000628`
- `lag_01__T_utility_inv`: contribution `-0.000628`

### tick `66869`, seconds `20.50`, LSTM delta `-0.0317`

Top all feature movements:
- `lag_03__CT_place_BDOORS`: contribution `-0.001703`
- `lag_00__CT_place_MIDDOORS`: contribution `-0.001236`
- `lag_08__CT_place_BDOORS`: contribution `-0.001123`
- `lag_06__CT_place_LOWERTUNNEL`: contribution `-0.001089`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `-0.000859`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67989`, seconds `38.00`, LSTM delta `-0.0292`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `-0.002565`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002228`
- `lag_00__T_kills_last_3s`: contribution `-0.001256`
- `lag_00__CT4__shots_fired`: contribution `-0.000935`
- `lag_03__T_shots_fired_sum`: contribution `-0.000887`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67925`, seconds `37.00`, LSTM delta `-0.0289`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `-0.001425`
- `lag_00__T_kills_last_3s`: contribution `-0.001256`
- `lag_13__CT4__duck_amount`: contribution `-0.000914`
- `lag_01__CT5__is_walking`: contribution `-0.000733`
- `lag_00__CT_place_LONGA`: contribution `-0.000700`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66325`, seconds `12.00`, LSTM delta `-0.0284`

Top all feature movements:
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.003016`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.002747`
- `lag_10__CT_place_LOWERTUNNEL`: contribution `-0.002687`
- `lag_12__CT_place_BDOORS`: contribution `-0.002669`
- `lag_04__CT_place_LOWERTUNNEL`: contribution `-0.002201`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `-0.000706`
- `lag_09__CT1__flash_duration`: contribution `-0.000403`
