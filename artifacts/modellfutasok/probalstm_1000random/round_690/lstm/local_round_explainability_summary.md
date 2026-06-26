# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `1`

## Largest probability jumps

- tick `4371`, seconds `48.50`, LSTM `0.5377`, delta `+0.3274`
- tick `5075`, seconds `59.50`, LSTM `0.8551`, delta `+0.1870`
- tick `4211`, seconds `46.00`, LSTM `0.3419`, delta `-0.1683`
- tick `4115`, seconds `44.50`, LSTM `0.4256`, delta `+0.1613`
- tick `4051`, seconds `43.50`, LSTM `0.2595`, delta `-0.1045`
- tick `4403`, seconds `49.00`, LSTM `0.6100`, delta `+0.0723`
- tick `4307`, seconds `47.50`, LSTM `0.1979`, delta `-0.0692`
- tick `4147`, seconds `45.00`, LSTM `0.4947`, delta `+0.0691`
- tick `3955`, seconds `42.00`, LSTM `0.4428`, delta `-0.0669`
- tick `4243`, seconds `46.50`, LSTM `0.2759`, delta `-0.0660`

## Top 15 local ridge features

- `lag_00__T_place_WALKWAY`: coefficient `-0.003950`, |coef| `0.003950`
- `lag_01__CT_place_TUNNELSTAIRS`: coefficient `0.003531`, |coef| `0.003531`
- `lag_08__T_place_WALKWAY`: coefficient `-0.002575`, |coef| `0.002575`
- `lag_04__CT_place_MAIN`: coefficient `0.002232`, |coef| `0.002232`
- `lag_10__T_place_WALKWAY`: coefficient `0.002047`, |coef| `0.002047`
- `lag_07__T_place_WALKWAY`: coefficient `-0.001990`, |coef| `0.001990`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001608`, |coef| `0.001608`
- `lag_01__T_place_WALKWAY`: coefficient `-0.001575`, |coef| `0.001575`
- `lag_00__damage_diff_last_5s`: coefficient `0.001559`, |coef| `0.001559`
- `lag_00__T_place_HEAVEN`: coefficient `-0.001534`, |coef| `0.001534`
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `0.001412`, |coef| `0.001412`
- `lag_02__T_place_MIDDOORS`: coefficient `-0.001395`, |coef| `0.001395`
- `lag_06__T_place_HEAVEN`: coefficient `0.001388`, |coef| `0.001388`

## Top 10 utility ridge features

- `lag_02__T1__flash_duration`: coefficient `-0.000784` (lowers CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000713` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `-0.000660` (lowers CT win probability)
- `lag_14__T1__flash_duration`: coefficient `0.000587` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000584` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000448` (lowers CT win probability)
- `lag_14__CT1__smoke`: coefficient `-0.000432` (lowers CT win probability)
- `lag_06__CT1__smoke`: coefficient `-0.000412` (lowers CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000348` (raises CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.000347` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_WALKWAY`: coefficient `-0.003950` (lowers CT win probability)
- `lag_01__CT_place_TUNNELSTAIRS`: coefficient `0.003531` (raises CT win probability)
- `lag_08__T_place_WALKWAY`: coefficient `-0.002575` (lowers CT win probability)
- `lag_04__CT_place_MAIN`: coefficient `0.002232` (raises CT win probability)
- `lag_10__T_place_WALKWAY`: coefficient `0.002047` (raises CT win probability)
- `lag_07__T_place_WALKWAY`: coefficient `-0.001990` (lowers CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001711` (lowers CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001620` (lowers CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001608` (raises CT win probability)
- `lag_01__T_place_WALKWAY`: coefficient `-0.001575` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4371`, seconds `48.50`, LSTM delta `+0.3274`

Top all feature movements:
- `lag_08__T_place_WALKWAY`: contribution `+0.035019`
- `lag_10__T_place_WALKWAY`: contribution `+0.027842`
- `lag_07__T_place_WALKWAY`: contribution `+0.027062`
- `lag_00__T_place_HEAVEN`: contribution `+0.018828`
- `lag_09__CT_place_BRICKS`: contribution `+0.017907`

Top utility-only movements:
- `lag_08__T1__flash_duration`: contribution `+0.003452`
- `lag_14__T1__flash_duration`: contribution `+0.003071`

### tick `5075`, seconds `59.50`, LSTM delta `+0.1870`

Top all feature movements:
- `lag_01__CT_place_TUNNELSTAIRS`: contribution `+0.049740`
- `lag_04__CT_place_MAIN`: contribution `+0.015028`
- `lag_04__CT_place_CANAL`: contribution `+0.006741`
- `lag_05__T_bomb_zone_count`: contribution `+0.006183`
- `lag_06__CT_place_HEAVEN`: contribution `+0.005411`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4211`, seconds `46.00`, LSTM delta `-0.1683`

Top all feature movements:
- `lag_08__T_place_WALKWAY`: contribution `-0.035019`
- `lag_02__CT_place_BRICKS`: contribution `-0.024408`
- `lag_01__T_place_WALKWAY`: contribution `+0.021417`
- `lag_01__T_place_HEAVEN`: contribution `-0.016668`
- `lag_09__T_place_WALKWAY`: contribution `-0.013510`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.003057`

### tick `4115`, seconds `44.50`, LSTM delta `+0.1613`

Top all feature movements:
- `lag_00__T_place_WALKWAY`: contribution `+0.053708`
- `lag_01__CT_place_BRICKS`: contribution `+0.018551`
- `lag_09__CT_place_BRICKS`: contribution `-0.017907`
- `lag_06__T_place_WALKWAY`: contribution `-0.016645`
- `lag_04__CT_place_MAIN`: contribution `+0.015028`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.003728`

### tick `4051`, seconds `43.50`, LSTM delta `-0.1045`

Top all feature movements:
- `lag_00__T_place_WALKWAY`: contribution `-0.053708`
- `lag_04__T_place_WALKWAY`: contribution `+0.007100`
- `lag_03__T_place_WALKWAY`: contribution `+0.006210`
- `lag_07__CT_place_BRICKS`: contribution `+0.004445`
- `lag_00__CT_place_BACKOFB`: contribution `-0.003955`

Top utility-only movements:
- No utility movement among the top local contributors.
