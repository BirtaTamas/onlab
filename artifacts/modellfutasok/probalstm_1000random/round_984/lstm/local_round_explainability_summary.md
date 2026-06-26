# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `80261`, seconds `102.50`, LSTM `0.5267`, delta `+0.4161`
- tick `77765`, seconds `63.50`, LSTM `0.2510`, delta `-0.2605`
- tick `79237`, seconds `86.50`, LSTM `0.0479`, delta `-0.2514`
- tick `77605`, seconds `61.00`, LSTM `0.5019`, delta `+0.2470`
- tick `77541`, seconds `60.00`, LSTM `0.2550`, delta `-0.2412`
- tick `78981`, seconds `82.50`, LSTM `0.3367`, delta `-0.2153`
- tick `77893`, seconds `65.50`, LSTM `0.4029`, delta `+0.2123`
- tick `80293`, seconds `103.00`, LSTM `0.7227`, delta `+0.1960`
- tick `78565`, seconds `76.00`, LSTM `0.5837`, delta `+0.0864`
- tick `77925`, seconds `66.00`, LSTM `0.4744`, delta `+0.0715`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.006808`, |coef| `0.006808`
- `lag_00__CT_defusing_count`: coefficient `0.006629`, |coef| `0.006629`
- `lag_01__CT_defusing_count`: coefficient `0.004435`, |coef| `0.004435`
- `lag_00__kill_diff_last_3s`: coefficient `0.004370`, |coef| `0.004370`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004139`, |coef| `0.004139`
- `lag_03__CT_place_RAFTERS`: coefficient `-0.004032`, |coef| `0.004032`
- `lag_14__CT_place_RAFTERS`: coefficient `0.003904`, |coef| `0.003904`
- `lag_02__CT_duck_amount_mean`: coefficient `0.003847`, |coef| `0.003847`
- `lag_11__T_duck_amount_mean`: coefficient `0.003508`, |coef| `0.003508`
- `lag_02__CT_defusing_count`: coefficient `0.003380`, |coef| `0.003380`
- `lag_00__T4__flash`: coefficient `-0.003265`, |coef| `0.003265`
- `lag_00__CT_kills_last_3s`: coefficient `0.003264`, |coef| `0.003264`
- `lag_14__CT_place_HEAVEN`: coefficient `-0.003164`, |coef| `0.003164`
- `lag_01__T_place_MINI`: coefficient `-0.003133`, |coef| `0.003133`
- `lag_00__T4__shots_fired`: coefficient `0.002844`, |coef| `0.002844`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.006808` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004139` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.003265` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.002772` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002111` (lowers CT win probability)
- `lag_01__T4__flash`: coefficient `-0.002018` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001905` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001522` (lowers CT win probability)
- `lag_02__T4__flash`: coefficient `-0.001359` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.001163` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.006629` (raises CT win probability)
- `lag_01__CT_defusing_count`: coefficient `0.004435` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004370` (raises CT win probability)
- `lag_03__CT_place_RAFTERS`: coefficient `-0.004032` (lowers CT win probability)
- `lag_14__CT_place_RAFTERS`: coefficient `0.003904` (raises CT win probability)
- `lag_02__CT_duck_amount_mean`: coefficient `0.003847` (raises CT win probability)
- `lag_11__T_duck_amount_mean`: coefficient `0.003508` (raises CT win probability)
- `lag_02__CT_defusing_count`: coefficient `0.003380` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003264` (raises CT win probability)
- `lag_14__CT_place_HEAVEN`: coefficient `-0.003164` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `80261`, seconds `102.50`, LSTM delta `+0.4161`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.041306`
- `lag_03__CT_place_RAFTERS`: contribution `+0.021543`
- `lag_02__CT_duck_amount_mean`: contribution `+0.021226`
- `lag_14__CT_place_RAFTERS`: contribution `+0.020862`
- `lag_11__T_duck_amount_mean`: contribution `+0.020404`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.041306`
- `lag_00__T4__flash`: contribution `+0.008872`

### tick `77765`, seconds `63.50`, LSTM delta `-0.2605`

Top all feature movements:
- `lag_02__T_place_MINI`: contribution `-0.023740`
- `lag_09__CT_place_DECON`: contribution `-0.023601`
- `lag_05__T_place_MINI`: contribution `-0.016971`
- `lag_06__CT_place_DECON`: contribution `-0.012810`
- `lag_00__kill_diff_last_3s`: contribution `-0.010518`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `-0.006202`

### tick `79237`, seconds `86.50`, LSTM delta `-0.2514`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `-0.028943`
- `lag_11__T4__shots_fired`: contribution `-0.023428`
- `lag_08__CT_place_HUT`: contribution `-0.013176`
- `lag_03__T_place_HUT`: contribution `-0.012035`
- `lag_00__kill_diff_last_3s`: contribution `-0.010518`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77605`, seconds `61.00`, LSTM delta `+0.2470`

Top all feature movements:
- `lag_00__T_place_MINI`: contribution `+0.036542`
- `lag_14__CT_place_DECON`: contribution `+0.019395`
- `lag_14__CT_place_HEAVEN`: contribution `-0.017081`
- `lag_13__CT_place_DECON`: contribution `-0.014116`
- `lag_11__CT_place_DECON`: contribution `+0.013188`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77541`, seconds `60.00`, LSTM delta `-0.2412`

Top all feature movements:
- `lag_01__T_place_MINI`: contribution `-0.043593`
- `lag_09__CT_place_DECON`: contribution `-0.023601`
- `lag_13__CT_place_DECON`: contribution `-0.014116`
- `lag_11__CT_place_DECON`: contribution `-0.013188`
- `lag_12__CT_place_HEAVEN`: contribution `-0.011431`

Top utility-only movements:
- No utility movement among the top local contributors.
