# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-lynn-vision-bo3-KVSQ5iZB0TjTG70slfdqOB/furia-vs-lynn-vision-m2-overpass.csv`
- round_num: `4`

## Largest probability jumps

- tick `24427`, seconds `13.50`, LSTM `0.2072`, delta `-0.2124`
- tick `26923`, seconds `52.50`, LSTM `0.0550`, delta `-0.0956`
- tick `24747`, seconds `18.50`, LSTM `0.1349`, delta `-0.0338`
- tick `24491`, seconds `14.50`, LSTM `0.1604`, delta `-0.0309`
- tick `24523`, seconds `15.00`, LSTM `0.1889`, delta `+0.0285`
- tick `26699`, seconds `49.00`, LSTM `0.1049`, delta `+0.0237`
- tick `26859`, seconds `51.50`, LSTM `0.1507`, delta `+0.0231`
- tick `24587`, seconds `16.00`, LSTM `0.1515`, delta `-0.0215`
- tick `24651`, seconds `17.00`, LSTM `0.1665`, delta `+0.0189`
- tick `24395`, seconds `13.00`, LSTM `0.4195`, delta `-0.0187`

## Top 15 local ridge features

- `lag_00__CT_place_LOWERPARK`: coefficient `0.001337`, |coef| `0.001337`
- `lag_14__T_place_TSTAIRS`: coefficient `0.001311`, |coef| `0.001311`
- `lag_10__CT_place_WALKWAY`: coefficient `-0.001182`, |coef| `0.001182`
- `lag_11__CT_place_SNIPERSNEST`: coefficient `0.001005`, |coef| `0.001005`
- `lag_14__T_place_TUNNELS`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_00__T_kills_last_3s`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_05__CT_place_WATER`: coefficient `0.000886`, |coef| `0.000886`
- `lag_15__T_place_ALLEY`: coefficient `-0.000863`, |coef| `0.000863`
- `lag_05__T_place_PIPE`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_11__CT_place_WATER`: coefficient `-0.000801`, |coef| `0.000801`
- `lag_10__T_place_ALLEY`: coefficient `-0.000791`, |coef| `0.000791`
- `lag_09__CT_place_SNIPERSNEST`: coefficient `0.000787`, |coef| `0.000787`
- `lag_05__CT_place_WALKWAY`: coefficient `-0.000779`, |coef| `0.000779`
- `lag_06__T_place_LOWERPARK`: coefficient `-0.000777`, |coef| `0.000777`
- `lag_05__T_place_FOUNTAIN`: coefficient `-0.000776`, |coef| `0.000776`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.000732` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000593` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `-0.000486` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000479` (raises CT win probability)
- `lag_10__CT4__molly`: coefficient `0.000470` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000469` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000467` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000463` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000457` (raises CT win probability)
- `lag_06__T4__molly`: coefficient `0.000438` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_LOWERPARK`: coefficient `0.001337` (raises CT win probability)
- `lag_14__T_place_TSTAIRS`: coefficient `0.001311` (raises CT win probability)
- `lag_10__CT_place_WALKWAY`: coefficient `-0.001182` (lowers CT win probability)
- `lag_11__CT_place_SNIPERSNEST`: coefficient `0.001005` (raises CT win probability)
- `lag_14__T_place_TUNNELS`: coefficient `-0.000971` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000894` (lowers CT win probability)
- `lag_05__CT_place_WATER`: coefficient `0.000886` (raises CT win probability)
- `lag_15__T_place_ALLEY`: coefficient `-0.000863` (lowers CT win probability)
- `lag_05__T_place_PIPE`: coefficient `-0.000812` (lowers CT win probability)
- `lag_11__CT_place_WATER`: coefficient `-0.000801` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `24427`, seconds `13.50`, LSTM delta `-0.2124`

Top all feature movements:
- `lag_14__T_place_TSTAIRS`: contribution `-0.014862`
- `lag_05__T_place_PIPE`: contribution `-0.010367`
- `lag_00__T_place_PIPE`: contribution `-0.006544`
- `lag_00__CT_place_LOWERPARK`: contribution `-0.005975`
- `lag_10__CT_place_WALKWAY`: contribution `-0.005803`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `26923`, seconds `52.50`, LSTM delta `-0.0956`

Top all feature movements:
- `lag_00__CT_place_LOWERPARK`: contribution `-0.005975`
- `lag_07__CT_place_STAIRS`: contribution `-0.005574`
- `lag_02__CT_place_BACKOFA`: contribution `-0.004182`
- `lag_09__CT_place_BACKOFA`: contribution `-0.003903`
- `lag_15__CT_place_CANAL`: contribution `-0.003382`

Top utility-only movements:
- `lag_00__CT2__molly`: contribution `-0.001182`

### tick `24747`, seconds `18.50`, LSTM delta `-0.0338`

Top all feature movements:
- `lag_15__T_place_PIPE`: contribution `-0.005377`
- `lag_02__CT_place_BRIDGE`: contribution `-0.004893`
- `lag_10__T_place_WATER`: contribution `-0.002820`
- `lag_15__CT_place_WALKWAY`: contribution `-0.001910`
- `lag_15__T_place_TUNNELS`: contribution `+0.001801`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24491`, seconds `14.50`, LSTM delta `-0.0309`

Top all feature movements:
- `lag_11__CT_place_SNIPERSNEST`: contribution `-0.005383`
- `lag_11__CT_place_WATER`: contribution `-0.004866`
- `lag_07__CT_place_WATER`: contribution `-0.004554`
- `lag_09__CT_place_WATER`: contribution `+0.004230`
- `lag_10__T_place_ALLEY`: contribution `+0.003350`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `24523`, seconds `15.00`, LSTM delta `+0.0285`

Top all feature movements:
- `lag_03__T_place_PIPE`: contribution `+0.003862`
- `lag_08__T_place_PIPE`: contribution `+0.002331`
- `lag_10__CT_place_WATER`: contribution `+0.001748`
- `lag_08__CT_place_WATER`: contribution `+0.001731`
- `lag_11__T_place_ALLEY`: contribution `+0.001537`

Top utility-only movements:
- `lag_01__CT_active_infernos`: contribution `+0.000817`
