# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `47027`, seconds `73.00`, LSTM `0.1054`, delta `-0.3192`
- tick `46451`, seconds `64.00`, LSTM `0.4809`, delta `+0.3124`
- tick `46323`, seconds `62.00`, LSTM `0.1612`, delta `-0.2082`
- tick `46579`, seconds `66.00`, LSTM `0.6812`, delta `+0.1815`
- tick `46867`, seconds `70.50`, LSTM `0.5013`, delta `-0.1399`
- tick `45683`, seconds `52.00`, LSTM `0.4336`, delta `-0.0851`
- tick `46931`, seconds `71.50`, LSTM `0.4284`, delta `-0.0643`
- tick `45043`, seconds `42.00`, LSTM `0.5073`, delta `+0.0508`
- tick `47443`, seconds `79.50`, LSTM `0.0587`, delta `-0.0495`
- tick `43443`, seconds `17.00`, LSTM `0.4536`, delta `-0.0463`

## Top 15 local ridge features

- `lag_02__T_place_EXTENDEDA`: coefficient `-0.002076`, |coef| `0.002076`
- `lag_05__T_place_EXTENDEDA`: coefficient `-0.002068`, |coef| `0.002068`
- `lag_14__T3__flash_duration`: coefficient `0.002021`, |coef| `0.002021`
- `lag_09__T1__flash_duration`: coefficient `0.001919`, |coef| `0.001919`
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001805`, |coef| `0.001805`
- `lag_04__CT1__flash_duration`: coefficient `-0.001773`, |coef| `0.001773`
- `lag_13__CT_place_UNDERA`: coefficient `-0.001726`, |coef| `0.001726`
- `lag_00__kill_diff_last_3s`: coefficient `0.001688`, |coef| `0.001688`
- `lag_10__CT3__flash_duration`: coefficient `0.001675`, |coef| `0.001675`
- `lag_06__T5__is_scoped`: coefficient `-0.001631`, |coef| `0.001631`
- `lag_01__T1__flash_duration`: coefficient `0.001587`, |coef| `0.001587`
- `lag_05__CT3__flash_duration`: coefficient `0.001577`, |coef| `0.001577`
- `lag_08__CT_flashes_last_5s`: coefficient `0.001565`, |coef| `0.001565`
- `lag_10__CT_flash_duration_sum`: coefficient `0.001545`, |coef| `0.001545`
- `lag_02__CT5__is_scoped`: coefficient `0.001536`, |coef| `0.001536`

## Top 10 utility ridge features

- `lag_14__T3__flash_duration`: coefficient `0.002021` (raises CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.001919` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001773` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.001675` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.001587` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.001577` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `0.001565` (raises CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `0.001545` (raises CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.001498` (raises CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `0.001496` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_EXTENDEDA`: coefficient `-0.002076` (lowers CT win probability)
- `lag_05__T_place_EXTENDEDA`: coefficient `-0.002068` (lowers CT win probability)
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.001805` (lowers CT win probability)
- `lag_13__CT_place_UNDERA`: coefficient `-0.001726` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001688` (raises CT win probability)
- `lag_06__T5__is_scoped`: coefficient `-0.001631` (lowers CT win probability)
- `lag_02__CT5__is_scoped`: coefficient `0.001536` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001505` (lowers CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `-0.001504` (lowers CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.001484` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `47027`, seconds `73.00`, LSTM delta `-0.3192`

Top all feature movements:
- `lag_09__T1__flash_duration`: contribution `-0.010769`
- `lag_13__CT_place_UNDERA`: contribution `-0.010546`
- `lag_05__T_place_EXTENDEDA`: contribution `-0.010253`
- `lag_10__CT3__flash_duration`: contribution `-0.008677`
- `lag_06__T5__is_scoped`: contribution `-0.007777`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.010769`
- `lag_10__CT3__flash_duration`: contribution `-0.008677`
- `lag_10__CT2__flash_duration`: contribution `-0.007491`
- `lag_10__CT_flash_duration_sum`: contribution `-0.007146`
- `lag_07__T3__flash_duration`: contribution `-0.005360`

### tick `46451`, seconds `64.00`, LSTM delta `+0.3124`

Top all feature movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.017203`
- `lag_14__T3__flash_duration`: contribution `+0.013368`
- `lag_04__CT1__flash_duration`: contribution `+0.012637`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.010293`
- `lag_14__T_flash_duration_sum`: contribution `+0.009441`

Top utility-only movements:
- `lag_08__CT_flashes_last_5s`: contribution `+0.017203`
- `lag_14__T3__flash_duration`: contribution `+0.013368`
- `lag_04__CT1__flash_duration`: contribution `+0.012637`
- `lag_14__T_flash_duration_sum`: contribution `+0.009441`
- `lag_01__T1__flash_duration`: contribution `+0.008905`

### tick `46323`, seconds `62.00`, LSTM delta `-0.2082`

Top all feature movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.016451`
- `lag_04__CT_flashes_last_5s`: contribution `-0.013746`
- `lag_01__CT_place_ARAMP`: contribution `-0.011802`
- `lag_13__CT1__flash_duration`: contribution `-0.007940`
- `lag_06__T5__is_scoped`: contribution `-0.007777`

Top utility-only movements:
- `lag_14__CT_flashes_last_5s`: contribution `-0.016451`
- `lag_04__CT_flashes_last_5s`: contribution `-0.013746`
- `lag_13__CT1__flash_duration`: contribution `-0.007940`
- `lag_10__T_flash_duration_sum`: contribution `-0.005533`
- `lag_10__T3__flash_duration`: contribution `-0.004257`

### tick `46579`, seconds `66.00`, LSTM delta `+0.1815`

Top all feature movements:
- `lag_02__T_place_EXTENDEDA`: contribution `+0.010293`
- `lag_05__T_place_EXTENDEDA`: contribution `+0.010253`
- `lag_08__CT1__flash_duration`: contribution `+0.009048`
- `lag_12__CT_flashes_last_5s`: contribution `+0.008713`
- `lag_09__CT_place_ARAMP`: contribution `+0.007852`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `+0.009048`
- `lag_12__CT_flashes_last_5s`: contribution `+0.008713`
- `lag_05__CT2__flash_duration`: contribution `+0.005294`
- `lag_05__T3__flash_duration`: contribution `+0.004759`
- `lag_06__T5__flash_duration`: contribution `+0.004486`

### tick `46867`, seconds `70.50`, LSTM delta `-0.1399`

Top all feature movements:
- `lag_11__CT_flashes_last_5s`: contribution `-0.011998`
- `lag_14__T3__flash_duration`: contribution `-0.011163`
- `lag_05__CT3__flash_duration`: contribution `-0.008165`
- `lag_02__CT5__is_scoped`: contribution `-0.005494`
- `lag_02__CT_flashed_players`: contribution `+0.005456`

Top utility-only movements:
- `lag_11__CT_flashes_last_5s`: contribution `-0.011998`
- `lag_14__T3__flash_duration`: contribution `-0.011163`
- `lag_05__CT3__flash_duration`: contribution `-0.008165`
- `lag_05__CT2__flash_duration`: contribution `-0.005294`
- `lag_15__T5__flash_duration`: contribution `-0.004854`
