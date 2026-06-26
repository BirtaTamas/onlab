# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `6`

## Largest probability jumps

- tick `41147`, seconds `43.00`, LSTM `0.2627`, delta `-0.2043`
- tick `41339`, seconds `46.00`, LSTM `0.0701`, delta `-0.1356`
- tick `39835`, seconds `22.50`, LSTM `0.5531`, delta `-0.0684`
- tick `39611`, seconds `19.00`, LSTM `0.5934`, delta `+0.0442`
- tick `41179`, seconds `43.50`, LSTM `0.2228`, delta `-0.0399`
- tick `41211`, seconds `44.00`, LSTM `0.1870`, delta `-0.0358`
- tick `39643`, seconds `19.50`, LSTM `0.6209`, delta `+0.0275`
- tick `41275`, seconds `45.00`, LSTM `0.1914`, delta `+0.0270`
- tick `39739`, seconds `21.00`, LSTM `0.6325`, delta `+0.0226`
- tick `41243`, seconds `44.50`, LSTM `0.1644`, delta `-0.0226`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.002326`, |coef| `0.002326`
- `lag_04__CT_place_FOUNTAIN`: coefficient `0.001653`, |coef| `0.001653`
- `lag_06__CT_place_FOUNTAIN`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_15__T_place_CONNECTOR`: coefficient `-0.001311`, |coef| `0.001311`
- `lag_09__T_place_CONNECTOR`: coefficient `-0.001304`, |coef| `0.001304`
- `lag_05__T5__flash_duration`: coefficient `0.001285`, |coef| `0.001285`
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001280`, |coef| `0.001280`
- `lag_00__CT_place_BRICKS`: coefficient `0.001214`, |coef| `0.001214`
- `lag_02__CT_place_BRIDGE`: coefficient `-0.001196`, |coef| `0.001196`
- `lag_06__CT_place_MAIN`: coefficient `0.001165`, |coef| `0.001165`
- `lag_08__T_place_CONNECTOR`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_10__T_place_CONNECTOR`: coefficient `-0.001057`, |coef| `0.001057`
- `lag_00__CT_place_BACKOFB`: coefficient `0.000992`, |coef| `0.000992`
- `lag_11__T_place_CONNECTOR`: coefficient `-0.000976`, |coef| `0.000976`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.000931`, |coef| `0.000931`

## Top 10 utility ridge features

- `lag_05__T5__flash_duration`: coefficient `0.001285` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000762` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000728` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000657` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000607` (raises CT win probability)
- `lag_03__CT2__smoke`: coefficient `0.000585` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000576` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000570` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000542` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.000526` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.002326` (raises CT win probability)
- `lag_04__CT_place_FOUNTAIN`: coefficient `0.001653` (raises CT win probability)
- `lag_06__CT_place_FOUNTAIN`: coefficient `-0.001329` (lowers CT win probability)
- `lag_15__T_place_CONNECTOR`: coefficient `-0.001311` (lowers CT win probability)
- `lag_09__T_place_CONNECTOR`: coefficient `-0.001304` (lowers CT win probability)
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001280` (lowers CT win probability)
- `lag_00__CT_place_BRICKS`: coefficient `0.001214` (raises CT win probability)
- `lag_02__CT_place_BRIDGE`: coefficient `-0.001196` (lowers CT win probability)
- `lag_06__CT_place_MAIN`: coefficient `0.001165` (raises CT win probability)
- `lag_08__T_place_CONNECTOR`: coefficient `-0.001063` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `41147`, seconds `43.00`, LSTM delta `-0.2043`

Top all feature movements:
- `lag_04__CT_place_FOUNTAIN`: contribution `-0.017389`
- `lag_06__CT_place_FOUNTAIN`: contribution `-0.013983`
- `lag_05__T5__flash_duration`: contribution `-0.009386`
- `lag_06__CT_place_MAIN`: contribution `-0.007846`
- `lag_09__T_place_CONNECTOR`: contribution `-0.006313`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.009386`

### tick `41339`, seconds `46.00`, LSTM delta `-0.1356`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `-0.013711`
- `lag_10__CT_place_FOUNTAIN`: contribution `-0.008092`
- `lag_12__CT_place_FOUNTAIN`: contribution `-0.007387`
- `lag_15__T_place_CONNECTOR`: contribution `-0.006348`
- `lag_09__T_place_CONNECTOR`: contribution `+0.006313`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `-0.004206`

### tick `39835`, seconds `22.50`, LSTM delta `-0.0684`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.023320`
- `lag_07__CT_place_BRICKS`: contribution `-0.004340`
- `lag_11__T5__is_scoped`: contribution `-0.002204`
- `lag_03__T4__flash_duration`: contribution `-0.002121`
- `lag_15__T2__duck_amount`: contribution `-0.001502`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.002121`

### tick `39611`, seconds `19.00`, LSTM delta `+0.0442`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `+0.023320`
- `lag_06__T5__is_scoped`: contribution `+0.002498`
- `lag_01__CT_place_BACKOFB`: contribution `-0.002139`
- `lag_12__T5__is_scoped`: contribution `+0.002132`
- `lag_08__T4__flash_duration`: contribution `+0.001753`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.001753`
- `lag_11__CT3__flash_duration`: contribution `+0.000882`
- `lag_09__CT1__flash_duration`: contribution `+0.000801`
- `lag_08__T_flash_duration_sum`: contribution `+0.000605`

### tick `41179`, seconds `43.50`, LSTM delta `-0.0399`

Top all feature movements:
- `lag_07__CT_place_FOUNTAIN`: contribution `-0.007767`
- `lag_15__T_place_CONNECTOR`: contribution `-0.006348`
- `lag_10__T_place_CONNECTOR`: contribution `-0.005118`
- `lag_06__T5__flash_duration`: contribution `-0.004794`
- `lag_05__CT_place_FOUNTAIN`: contribution `-0.003900`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `-0.004794`
