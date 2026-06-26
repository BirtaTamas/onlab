# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `18`

## Largest probability jumps

- tick `139212`, seconds `13.00`, LSTM `0.0632`, delta `-0.1272`
- tick `139116`, seconds `11.50`, LSTM `0.2817`, delta `-0.1109`
- tick `139148`, seconds `12.00`, LSTM `0.2142`, delta `-0.0674`
- tick `139084`, seconds `11.00`, LSTM `0.3926`, delta `-0.0670`
- tick `139660`, seconds `20.00`, LSTM `0.0129`, delta `-0.0336`
- tick `138668`, seconds `4.50`, LSTM `0.4662`, delta `+0.0279`
- tick `139180`, seconds `12.50`, LSTM `0.1905`, delta `-0.0238`
- tick `139436`, seconds `16.50`, LSTM `0.0462`, delta `-0.0162`
- tick `139404`, seconds `16.00`, LSTM `0.0624`, delta `+0.0106`
- tick `138604`, seconds `3.50`, LSTM `0.4462`, delta `-0.0098`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001859`, |coef| `0.001859`
- `lag_06__CT_place_BRIDGE`: coefficient `-0.001480`, |coef| `0.001480`
- `lag_07__CT_place_BRIDGE`: coefficient `-0.001348`, |coef| `0.001348`
- `lag_00__CT_place_BRIDGE`: coefficient `0.001210`, |coef| `0.001210`
- `lag_09__CT_place_BRIDGE`: coefficient `-0.001061`, |coef| `0.001061`
- `lag_05__CT_place_BRIDGE`: coefficient `-0.000965`, |coef| `0.000965`
- `lag_14__CT_place_LOWERTUNNEL`: coefficient `0.000859`, |coef| `0.000859`
- `lag_01__CT_place_BRIDGE`: coefficient `0.000826`, |coef| `0.000826`
- `lag_00__CT1__flash`: coefficient `0.000789`, |coef| `0.000789`
- `lag_01__CT_place_MAIN`: coefficient `-0.000775`, |coef| `0.000775`
- `lag_04__CT_place_MAIN`: coefficient `-0.000732`, |coef| `0.000732`
- `lag_14__T_place_STREET`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_05__CT_place_CTSIDEUPPER`: coefficient `0.000730`, |coef| `0.000730`
- `lag_00__CT1__utility_total`: coefficient `0.000717`, |coef| `0.000717`
- `lag_15__T_place_RUINS`: coefficient `-0.000707`, |coef| `0.000707`

## Top 10 utility ridge features

- `lag_00__CT1__flash`: coefficient `0.000789` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000717` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000572` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000549` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000459` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000457` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000453` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000429` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000400` (raises CT win probability)
- `lag_09__T3__smoke`: coefficient `0.000370` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001859` (raises CT win probability)
- `lag_06__CT_place_BRIDGE`: coefficient `-0.001480` (lowers CT win probability)
- `lag_07__CT_place_BRIDGE`: coefficient `-0.001348` (lowers CT win probability)
- `lag_00__CT_place_BRIDGE`: coefficient `0.001210` (raises CT win probability)
- `lag_09__CT_place_BRIDGE`: coefficient `-0.001061` (lowers CT win probability)
- `lag_05__CT_place_BRIDGE`: coefficient `-0.000965` (lowers CT win probability)
- `lag_14__CT_place_LOWERTUNNEL`: coefficient `0.000859` (raises CT win probability)
- `lag_01__CT_place_BRIDGE`: coefficient `0.000826` (raises CT win probability)
- `lag_01__CT_place_MAIN`: coefficient `-0.000775` (lowers CT win probability)
- `lag_04__CT_place_MAIN`: coefficient `-0.000732` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `139212`, seconds `13.00`, LSTM delta `-0.1272`

Top all feature movements:
- `lag_07__CT_place_BRIDGE`: contribution `-0.015449`
- `lag_00__CT_place_BRIDGE`: contribution `-0.013865`
- `lag_09__CT_place_BRIDGE`: contribution `-0.012167`
- `lag_04__CT_place_MAIN`: contribution `-0.004931`
- `lag_13__T_place_STREET`: contribution `-0.003009`

Top utility-only movements:
- `lag_00__CT1__flash`: contribution `-0.002824`
- `lag_00__CT1__utility_total`: contribution `-0.002019`
- `lag_00__CT1__molly`: contribution `-0.001366`

### tick `139116`, seconds `11.50`, LSTM delta `-0.1109`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `-0.016959`
- `lag_14__CT_place_LOWERTUNNEL`: contribution `-0.012627`
- `lag_01__CT_place_BRIDGE`: contribution `-0.009473`
- `lag_01__CT_place_MAIN`: contribution `-0.005217`
- `lag_14__T_place_STREET`: contribution `-0.004020`

Top utility-only movements:
- `lag_00__CT2__molly`: contribution `-0.001411`

### tick `139148`, seconds `12.00`, LSTM delta `-0.0674`

Top all feature movements:
- `lag_07__CT_place_BRIDGE`: contribution `-0.015449`
- `lag_05__CT_place_BRIDGE`: contribution `-0.011066`
- `lag_15__CT_place_LOWERTUNNEL`: contribution `-0.004397`
- `lag_02__CT_place_BRIDGE`: contribution `-0.003916`
- `lag_07__T_place_BRIDGE`: contribution `-0.002249`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139084`, seconds `11.00`, LSTM delta `-0.0670`

Top all feature movements:
- `lag_00__CT_place_BRIDGE`: contribution `-0.013865`
- `lag_05__CT_place_BRIDGE`: contribution `-0.011066`
- `lag_13__T_place_STREET`: contribution `-0.003009`
- `lag_03__CT_place_BRIDGE`: contribution `-0.002811`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `-0.002062`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139660`, seconds `20.00`, LSTM delta `-0.0336`

Top all feature movements:
- `lag_13__T_place_STREET`: contribution `+0.003009`
- `lag_00__T_place_BRIDGE`: contribution `-0.002852`
- `lag_09__CT_place_OUTSIDELONG`: contribution `-0.002815`
- `lag_14__CT_place_BRIDGE`: contribution `+0.002041`
- `lag_06__CT_place_OUTSIDELONG`: contribution `-0.001832`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.000659`
