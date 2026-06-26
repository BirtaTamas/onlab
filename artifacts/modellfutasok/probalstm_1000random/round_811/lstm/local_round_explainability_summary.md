# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `121079`, seconds `72.50`, LSTM `0.2150`, delta `-0.2940`
- tick `120983`, seconds `71.00`, LSTM `0.5337`, delta `-0.1305`
- tick `120919`, seconds `70.00`, LSTM `0.6331`, delta `+0.1108`
- tick `121111`, seconds `73.00`, LSTM `0.1259`, delta `-0.0891`
- tick `119831`, seconds `53.00`, LSTM `0.4801`, delta `+0.0854`
- tick `118327`, seconds `29.50`, LSTM `0.4355`, delta `+0.0789`
- tick `118231`, seconds `28.00`, LSTM `0.4476`, delta `-0.0606`
- tick `119767`, seconds `52.00`, LSTM `0.4261`, delta `-0.0524`
- tick `118263`, seconds `28.50`, LSTM `0.3990`, delta `-0.0485`
- tick `118359`, seconds `30.00`, LSTM `0.4824`, delta `+0.0469`

## Top 15 local ridge features

- `lag_08__T_place_ARCH`: coefficient `-0.003367`, |coef| `0.003367`
- `lag_00__T_place_BALCONY`: coefficient `-0.003305`, |coef| `0.003305`
- `lag_01__T_place_BALCONY`: coefficient `-0.002932`, |coef| `0.002932`
- `lag_02__T_place_ARCH`: coefficient `0.002681`, |coef| `0.002681`
- `lag_00__kill_diff_last_3s`: coefficient `0.001847`, |coef| `0.001847`
- `lag_00__T_kills_last_3s`: coefficient `-0.001802`, |coef| `0.001802`
- `lag_03__T_place_ARCH`: coefficient `0.001723`, |coef| `0.001723`
- `lag_05__T_place_ARCH`: coefficient `-0.001718`, |coef| `0.001718`
- `lag_00__CT_place_BALCONY`: coefficient `0.001696`, |coef| `0.001696`
- `lag_00__damage_diff_last_5s`: coefficient `0.001611`, |coef| `0.001611`
- `lag_12__T4__duck_amount`: coefficient `0.001504`, |coef| `0.001504`
- `lag_15__T2__duck_amount`: coefficient `0.001498`, |coef| `0.001498`
- `lag_00__T2__duck_amount`: coefficient `-0.001479`, |coef| `0.001479`
- `lag_10__T_place_ARCH`: coefficient `-0.001401`, |coef| `0.001401`
- `lag_01__damage_diff_last_5s`: coefficient `0.001392`, |coef| `0.001392`

## Top 10 utility ridge features

- `lag_00__CT4__smoke`: coefficient `0.000244` (raises CT win probability)
- `lag_01__CT4__smoke`: coefficient `0.000242` (raises CT win probability)
- `lag_10__CT4__smoke`: coefficient `0.000182` (raises CT win probability)
- `lag_05__CT4__smoke`: coefficient `0.000156` (raises CT win probability)
- `lag_11__CT4__smoke`: coefficient `0.000156` (raises CT win probability)
- `lag_04__CT4__smoke`: coefficient `0.000155` (raises CT win probability)
- `lag_01__CT4__utility_total`: coefficient `0.000120` (raises CT win probability)
- `lag_11__CT4__utility_total`: coefficient `0.000110` (raises CT win probability)
- `lag_10__CT4__utility_total`: coefficient `0.000109` (raises CT win probability)
- `lag_11__CT_smoke_inv`: coefficient `0.000107` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_ARCH`: coefficient `-0.003367` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.003305` (lowers CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `-0.002932` (lowers CT win probability)
- `lag_02__T_place_ARCH`: coefficient `0.002681` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001847` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001802` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.001723` (raises CT win probability)
- `lag_05__T_place_ARCH`: coefficient `-0.001718` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `0.001696` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001611` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121079`, seconds `72.50`, LSTM delta `-0.2940`

Top all feature movements:
- `lag_01__T_place_BALCONY`: contribution `-0.040323`
- `lag_08__T_place_ARCH`: contribution `-0.031322`
- `lag_02__T_place_ARCH`: contribution `-0.024944`
- `lag_07__T_place_BALCONY`: contribution `-0.016773`
- `lag_05__T_place_BALCONY`: contribution `-0.013958`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120983`, seconds `71.00`, LSTM delta `-0.1305`

Top all feature movements:
- `lag_05__T_place_ARCH`: contribution `-0.015986`
- `lag_04__T_place_BALCONY`: contribution `-0.014504`
- `lag_00__T_kills_last_3s`: contribution `-0.005708`
- `lag_12__T4__duck_amount`: contribution `-0.005562`
- `lag_02__T_place_BALCONY`: contribution `-0.004851`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120919`, seconds `70.00`, LSTM delta `+0.1108`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.045445`
- `lag_03__T_place_ARCH`: contribution `+0.016029`
- `lag_02__T_place_BALCONY`: contribution `+0.004851`
- `lag_11__T_place_TOPOFMID`: contribution `+0.004271`
- `lag_00__damage_diff_last_5s`: contribution `+0.003016`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121111`, seconds `73.00`, LSTM delta `-0.0891`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `-0.016029`
- `lag_09__T_place_ARCH`: contribution `-0.012901`
- `lag_06__T_place_BALCONY`: contribution `-0.009146`
- `lag_02__T_place_BALCONY`: contribution `+0.004851`
- `lag_09__T2__duck_amount`: contribution `-0.004473`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119831`, seconds `53.00`, LSTM delta `+0.0854`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.045445`
- `lag_02__T_place_BALCONY`: contribution `+0.004851`
- `lag_00__CT2__duck_amount`: contribution `+0.003882`
- `lag_15__T3__duck_amount`: contribution `+0.003198`
- `lag_13__CT_place_BALCONY`: contribution `+0.002998`

Top utility-only movements:
- No utility movement among the top local contributors.
