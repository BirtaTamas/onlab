# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `44671`, seconds `0.50`, LSTM `0.9462`, delta `+0.0193`
- tick `44927`, seconds `4.50`, LSTM `0.9267`, delta `-0.0192`
- tick `49663`, seconds `78.50`, LSTM `0.9545`, delta `+0.0149`
- tick `44959`, seconds `5.00`, LSTM `0.9136`, delta `-0.0132`
- tick `45119`, seconds `7.50`, LSTM `0.9021`, delta `-0.0120`
- tick `45791`, seconds `18.00`, LSTM `0.9534`, delta `+0.0102`
- tick `49791`, seconds `80.50`, LSTM `0.9682`, delta `+0.0102`
- tick `49567`, seconds `77.00`, LSTM `0.9454`, delta `-0.0093`
- tick `45151`, seconds `8.00`, LSTM `0.9107`, delta `+0.0086`
- tick `45599`, seconds `15.00`, LSTM `0.9509`, delta `+0.0084`

## Top 15 local ridge features

- `lag_00__T_place_HOUSE`: coefficient `-0.000477`, |coef| `0.000477`
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000297`, |coef| `0.000297`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000258`, |coef| `0.000258`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.000241`, |coef| `0.000241`
- `lag_01__T_place_HOUSE`: coefficient `-0.000239`, |coef| `0.000239`
- `lag_00__CT5__is_walking`: coefficient `-0.000231`, |coef| `0.000231`
- `lag_00__CT_walking_count`: coefficient `-0.000225`, |coef| `0.000225`
- `lag_00__T_walking_count`: coefficient `-0.000206`, |coef| `0.000206`
- `lag_08__CT3__duck_amount`: coefficient `-0.000185`, |coef| `0.000185`
- `lag_03__CT_place_CATWALK`: coefficient `0.000174`, |coef| `0.000174`
- `lag_15__CT_place_CONNECTOR`: coefficient `-0.000172`, |coef| `0.000172`
- `lag_00__CT_place_JUNGLE`: coefficient `0.000169`, |coef| `0.000169`
- `lag_15__CT_place_STAIRS`: coefficient `0.000161`, |coef| `0.000161`
- `lag_09__CT_place_CTSPAWN`: coefficient `-0.000160`, |coef| `0.000160`
- `lag_02__T_place_SCAFFOLDING`: coefficient `0.000158`, |coef| `0.000158`

## Top 10 utility ridge features

- `lag_01__T_flash_alpha_mean`: coefficient `-0.000109` (lowers CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `-0.000102` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000090` (raises CT win probability)
- `lag_09__CT1__utility_total`: coefficient `-0.000089` (lowers CT win probability)
- `lag_09__smoke_inv_diff`: coefficient `-0.000089` (lowers CT win probability)
- `lag_09__utility_inv_diff`: coefficient `-0.000088` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000085` (raises CT win probability)
- `lag_09__CT1__smoke`: coefficient `-0.000079` (lowers CT win probability)
- `lag_09__CT5__utility_total`: coefficient `-0.000078` (lowers CT win probability)
- `lag_09__CT_utility_inv`: coefficient `-0.000077` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HOUSE`: coefficient `-0.000477` (lowers CT win probability)
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000297` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000258` (raises CT win probability)
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.000241` (raises CT win probability)
- `lag_01__T_place_HOUSE`: coefficient `-0.000239` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000231` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000225` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000206` (lowers CT win probability)
- `lag_08__CT3__duck_amount`: coefficient `-0.000185` (lowers CT win probability)
- `lag_03__CT_place_CATWALK`: coefficient `0.000174` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `44671`, seconds `0.50`, LSTM delta `+0.0193`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `+0.000662`
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000460`
- `lag_00__T_velocity_mean`: contribution `+0.000455`
- `lag_00__CT_velocity_mean`: contribution `+0.000405`
- `lag_01__smoke_inv_diff`: contribution `+0.000290`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `+0.000290`
- `lag_01__CT_flash_alpha_mean`: contribution `+0.000290`
- `lag_01__utility_inv_diff`: contribution `+0.000284`
- `lag_01__T_flash_alpha_mean`: contribution `+0.000240`
- `lag_01__molly_inv_diff`: contribution `+0.000175`

### tick `44927`, seconds `4.50`, LSTM delta `-0.0192`

Top all feature movements:
- `lag_00__T_place_HOUSE`: contribution `-0.004191`
- `lag_00__T_place_SIDEALLEY`: contribution `-0.001894`
- `lag_09__CT_place_CTSPAWN`: contribution `-0.000763`
- `lag_09__T_place_TSPAWN`: contribution `-0.000518`
- `lag_08__CT_velocity_mean`: contribution `-0.000461`

Top utility-only movements:
- `lag_09__utility_inv_diff`: contribution `-0.000291`
- `lag_09__smoke_inv_diff`: contribution `-0.000287`
- `lag_09__CT_utility_inv`: contribution `-0.000182`
- `lag_09__molly_inv_diff`: contribution `-0.000182`
- `lag_09__CT_smoke_inv`: contribution `-0.000176`

### tick `49663`, seconds `78.50`, LSTM delta `+0.0149`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.001288`
- `lag_00__CT_shots_fired_sum`: contribution `+0.000897`
- `lag_08__CT3__duck_amount`: contribution `+0.000688`
- `lag_15__CT_place_CONNECTOR`: contribution `+0.000616`
- `lag_03__CT1__shots_fired`: contribution `+0.000576`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44959`, seconds `5.00`, LSTM delta `-0.0132`

Top all feature movements:
- `lag_01__T_place_HOUSE`: contribution `-0.002106`
- `lag_00__T_place_HOUSE`: contribution `-0.002095`
- `lag_00__T_place_SIDEALLEY`: contribution `-0.000947`
- `lag_10__CT_place_CTSPAWN`: contribution `-0.000543`
- `lag_01__T_place_SIDEALLEY`: contribution `-0.000529`

Top utility-only movements:
- `lag_10__utility_inv_diff`: contribution `-0.000210`
- `lag_10__smoke_inv_diff`: contribution `-0.000198`
- `lag_10__CT1__utility_total`: contribution `-0.000147`
- `lag_10__CT_utility_inv`: contribution `-0.000144`

### tick `45119`, seconds `7.50`, LSTM delta `-0.0120`

Top all feature movements:
- `lag_00__T_place_HOUSE`: contribution `-0.002095`
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.001288`
- `lag_00__T_place_SIDEALLEY`: contribution `-0.000947`
- `lag_00__CT5__is_walking`: contribution `+0.000553`
- `lag_06__CT4__duck_amount`: contribution `-0.000539`

Top utility-only movements:
- No utility movement among the top local contributors.
