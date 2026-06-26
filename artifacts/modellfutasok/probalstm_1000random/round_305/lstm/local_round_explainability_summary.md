# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-inner-circle-bo3-YbhHiIk4CcU9clhSbtidF_/spirit-vs-inner-circle-m1-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `84728`, seconds `19.50`, LSTM `0.3381`, delta `-0.2782`
- tick `85336`, seconds `29.00`, LSTM `0.0754`, delta `-0.2179`
- tick `84440`, seconds `15.00`, LSTM `0.5924`, delta `+0.1784`
- tick `85112`, seconds `25.50`, LSTM `0.3390`, delta `+0.1468`
- tick `84248`, seconds `12.00`, LSTM `0.4985`, delta `-0.1193`
- tick `84760`, seconds `20.00`, LSTM `0.2403`, delta `-0.0977`
- tick `84280`, seconds `12.50`, LSTM `0.4430`, delta `-0.0555`
- tick `84696`, seconds `19.00`, LSTM `0.6162`, delta `-0.0481`
- tick `85304`, seconds `28.50`, LSTM `0.2933`, delta `-0.0429`
- tick `84536`, seconds `16.50`, LSTM `0.6412`, delta `+0.0408`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003517`, |coef| `0.003517`
- `lag_00__kill_diff_last_3s`: coefficient `0.002861`, |coef| `0.002861`
- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.002828`, |coef| `0.002828`
- `lag_13__CT_place_TSIDELOWER`: coefficient `0.002560`, |coef| `0.002560`
- `lag_00__T_kills_last_3s`: coefficient `-0.002458`, |coef| `0.002458`
- `lag_06__CT_place_TSIDELOWER`: coefficient `-0.002213`, |coef| `0.002213`
- `lag_12__CT_place_TSIDELOWER`: coefficient `0.002159`, |coef| `0.002159`
- `lag_09__T_place_MAINHALL`: coefficient `0.002007`, |coef| `0.002007`
- `lag_00__damage_diff_last_5s`: coefficient `0.001879`, |coef| `0.001879`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001872`, |coef| `0.001872`
- `lag_08__CT_B_site_active_infernos`: coefficient `0.001736`, |coef| `0.001736`
- `lag_04__T_place_MAINHALL`: coefficient `-0.001519`, |coef| `0.001519`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.001485`, |coef| `0.001485`
- `lag_07__CT_shots_fired_sum`: coefficient `0.001344`, |coef| `0.001344`
- `lag_00__CT1__flash_duration`: coefficient `0.001335`, |coef| `0.001335`

## Top 10 utility ridge features

- `lag_08__CT_B_site_active_infernos`: coefficient `0.001736` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001335` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001246` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `-0.001124` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001001` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000986` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000931` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000926` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `0.000922` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000878` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003517` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002861` (raises CT win probability)
- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.002828` (lowers CT win probability)
- `lag_13__CT_place_TSIDELOWER`: coefficient `0.002560` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002458` (lowers CT win probability)
- `lag_06__CT_place_TSIDELOWER`: coefficient `-0.002213` (lowers CT win probability)
- `lag_12__CT_place_TSIDELOWER`: coefficient `0.002159` (raises CT win probability)
- `lag_09__T_place_MAINHALL`: coefficient `0.002007` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001879` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001872` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `84728`, seconds `19.50`, LSTM delta `-0.2782`

Top all feature movements:
- `lag_00__CT_place_TSIDELOWER`: contribution `-0.038409`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.014074`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.008561`
- `lag_08__T_place_SIDEHALL`: contribution `-0.008468`
- `lag_00__T_kills_last_3s`: contribution `-0.007789`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `-0.008561`
- `lag_00__CT1__flash_duration`: contribution `-0.007353`
- `lag_06__CT1__flash_duration`: contribution `-0.006188`
- `lag_06__CT_active_infernos`: contribution `-0.003872`
- `lag_15__CT3__flash_duration`: contribution `-0.003618`

### tick `85336`, seconds `29.00`, LSTM delta `-0.2179`

Top all feature movements:
- `lag_13__CT_place_TSIDELOWER`: contribution `-0.034775`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.011163`
- `lag_05__CT_place_TSIDEUPPER`: contribution `-0.009123`
- `lag_00__T_kills_last_3s`: contribution `-0.007789`
- `lag_09__T_place_MAINHALL`: contribution `-0.007245`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.004719`
- `lag_00__T3__flash_duration`: contribution `-0.003483`
- `lag_00__T_flash_duration_sum`: contribution `-0.002810`

### tick `84440`, seconds `15.00`, LSTM delta `+0.1784`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013771`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.011928`
- `lag_00__T_kills_last_3s`: contribution `+0.007789`
- `lag_09__T_place_MAINHALL`: contribution `+0.007245`
- `lag_07__CT_place_SIDEENTRANCE`: contribution `+0.004971`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.011928`
- `lag_08__CT_active_infernos`: contribution `+0.003908`
- `lag_06__CT3__flash_duration`: contribution `+0.003653`
- `lag_11__CT3__flash_duration`: contribution `+0.003441`

### tick `85112`, seconds `25.50`, LSTM delta `+0.1468`

Top all feature movements:
- `lag_06__CT_place_TSIDELOWER`: contribution `+0.030056`
- `lag_12__CT_place_TSIDELOWER`: contribution `+0.029322`
- `lag_12__CT_place_TSIDEUPPER`: contribution `+0.008268`
- `lag_00__kill_diff_last_3s`: contribution `+0.006886`
- `lag_00__T_place_SIDEHALL`: contribution `+0.005981`

Top utility-only movements:
- `lag_12__CT1__flash_duration`: contribution `+0.004815`
- `lag_12__CT1__flash`: contribution `+0.001657`

### tick `84248`, seconds `12.00`, LSTM delta `-0.1193`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007789`
- `lag_00__kill_diff_last_3s`: contribution `-0.006886`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.006775`
- `lag_04__T_place_MAINHALL`: contribution `-0.005482`
- `lag_14__CT_place_HOUSE`: contribution `-0.004595`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `-0.006775`
- `lag_02__CT_active_infernos`: contribution `-0.003220`
- `lag_00__CT3__flash_duration`: contribution `-0.002274`
- `lag_08__CT_active_infernos`: contribution `+0.001954`
- `lag_05__CT3__flash_duration`: contribution `-0.001840`
