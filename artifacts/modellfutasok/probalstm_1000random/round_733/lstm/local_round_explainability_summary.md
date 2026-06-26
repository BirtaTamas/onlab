# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `50680`, seconds `16.00`, LSTM `0.2453`, delta `-0.0945`
- tick `51512`, seconds `29.00`, LSTM `0.0222`, delta `-0.0807`
- tick `51224`, seconds `24.50`, LSTM `0.0759`, delta `-0.0541`
- tick `52536`, seconds `45.00`, LSTM `0.0148`, delta `-0.0448`
- tick `50776`, seconds `17.50`, LSTM `0.2358`, delta `-0.0397`
- tick `50744`, seconds `17.00`, LSTM `0.2754`, delta `+0.0390`
- tick `49688`, seconds `0.50`, LSTM `0.2255`, delta `-0.0387`
- tick `51448`, seconds `28.00`, LSTM `0.0832`, delta `+0.0307`
- tick `50968`, seconds `20.50`, LSTM `0.1749`, delta `-0.0261`
- tick `50392`, seconds `11.50`, LSTM `0.3163`, delta `+0.0255`

## Top 15 local ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001441`, |coef| `0.001441`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001175`, |coef| `0.001175`
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.000818`, |coef| `0.000818`
- `lag_08__T_place_SIDEENTRANCE`: coefficient `-0.000790`, |coef| `0.000790`
- `lag_09__T_place_SIDEENTRANCE`: coefficient `-0.000706`, |coef| `0.000706`
- `lag_10__T_place_TSIDELOWER`: coefficient `-0.000692`, |coef| `0.000692`
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.000665`, |coef| `0.000665`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000664`, |coef| `0.000664`
- `lag_11__T_place_SIDEENTRANCE`: coefficient `-0.000632`, |coef| `0.000632`
- `lag_04__T_place_SIDEENTRANCE`: coefficient `-0.000626`, |coef| `0.000626`
- `lag_11__CT_place_SIDEENTRANCE`: coefficient `-0.000601`, |coef| `0.000601`
- `lag_06__T_place_TUNNEL`: coefficient `0.000591`, |coef| `0.000591`
- `lag_03__T_place_SIDEENTRANCE`: coefficient `-0.000591`, |coef| `0.000591`
- `lag_13__CT_place_ALLEY`: coefficient `0.000579`, |coef| `0.000579`
- `lag_11__T_place_TSIDELOWER`: coefficient `-0.000570`, |coef| `0.000570`

## Top 10 utility ridge features

- `lag_05__T_B_site_active_infernos`: coefficient `-0.000568` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.000559` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000531` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000502` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000498` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.000425` (raises CT win probability)
- `lag_09__CT_flashes_last_5s`: coefficient `-0.000419` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `-0.000413` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000409` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000409` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001441` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001175` (raises CT win probability)
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.000818` (raises CT win probability)
- `lag_08__T_place_SIDEENTRANCE`: coefficient `-0.000790` (lowers CT win probability)
- `lag_09__T_place_SIDEENTRANCE`: coefficient `-0.000706` (lowers CT win probability)
- `lag_10__T_place_TSIDELOWER`: coefficient `-0.000692` (lowers CT win probability)
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.000665` (lowers CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000664` (lowers CT win probability)
- `lag_11__T_place_SIDEENTRANCE`: coefficient `-0.000632` (lowers CT win probability)
- `lag_04__T_place_SIDEENTRANCE`: coefficient `-0.000626` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `50680`, seconds `16.00`, LSTM delta `-0.0945`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.007032`
- `lag_06__T_place_TUNNEL`: contribution `-0.003592`
- `lag_06__T_place_WATER`: contribution `-0.002963`
- `lag_11__T_place_TUNNEL`: contribution `-0.002729`
- `lag_04__T_place_WATER`: contribution `-0.002601`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.001726`
- `lag_05__T_B_site_active_infernos`: contribution `-0.001606`
- `lag_09__T4__flash_duration`: contribution `-0.001216`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.000960`

### tick `51512`, seconds `29.00`, LSTM delta `-0.0807`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.004643`
- `lag_10__T_place_TSIDELOWER`: contribution `-0.002595`
- `lag_06__T_shots_fired_sum`: contribution `-0.002467`
- `lag_01__T3__is_scoped`: contribution `-0.001763`
- `lag_00__T_kills_last_3s`: contribution `-0.001567`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `-0.001502`

### tick `51224`, seconds `24.50`, LSTM delta `-0.0541`

Top all feature movements:
- `lag_08__T_place_SIDEENTRANCE`: contribution `-0.003854`
- `lag_14__T_B_site_active_infernos`: contribution `-0.001580`
- `lag_04__T_B_site_active_infernos`: contribution `-0.001409`
- `lag_00__T_shots_fired_sum`: contribution `-0.001176`
- `lag_01__T_B_site_active_infernos`: contribution `-0.001157`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.001580`
- `lag_04__T_B_site_active_infernos`: contribution `-0.001409`
- `lag_01__T_B_site_active_infernos`: contribution `-0.001157`
- `lag_14__T_active_infernos`: contribution `-0.000850`
- `lag_12__T_B_site_active_infernos`: contribution `-0.000836`

### tick `52536`, seconds `45.00`, LSTM delta `-0.0448`

Top all feature movements:
- `lag_11__T_place_SIDEENTRANCE`: contribution `-0.003083`
- `lag_05__T3__is_scoped`: contribution `-0.002616`
- `lag_01__T3__is_scoped`: contribution `+0.001763`
- `lag_00__T_kills_last_3s`: contribution `-0.001567`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.001501`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.001059`

### tick `50776`, seconds `17.50`, LSTM delta `-0.0397`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.007032`
- `lag_03__T_place_SIDEENTRANCE`: contribution `-0.002884`
- `lag_01__T_place_SIDEENTRANCE`: contribution `+0.002715`
- `lag_02__T_shots_fired_sum`: contribution `-0.002110`
- `lag_00__T_place_TSIDEUPPER`: contribution `-0.002064`

Top utility-only movements:
- `lag_03__CT_B_site_active_infernos`: contribution `-0.000931`
- `lag_00__T_B_site_active_infernos`: contribution `+0.000766`
