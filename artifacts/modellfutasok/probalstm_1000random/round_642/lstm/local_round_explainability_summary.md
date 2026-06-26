# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `1`

## Largest probability jumps

- tick `2485`, seconds `19.00`, LSTM `0.3232`, delta `-0.0969`
- tick `4309`, seconds `47.50`, LSTM `0.0344`, delta `-0.0964`
- tick `2453`, seconds `18.50`, LSTM `0.4201`, delta `-0.0906`
- tick `2581`, seconds `20.50`, LSTM `0.1518`, delta `-0.0602`
- tick `3477`, seconds `34.50`, LSTM `0.3863`, delta `+0.0599`
- tick `2517`, seconds `19.50`, LSTM `0.2634`, delta `-0.0598`
- tick `3797`, seconds `39.50`, LSTM `0.3301`, delta `-0.0579`
- tick `2773`, seconds `23.50`, LSTM `0.1867`, delta `+0.0561`
- tick `2901`, seconds `25.50`, LSTM `0.3450`, delta `+0.0524`
- tick `2549`, seconds `20.00`, LSTM `0.2121`, delta `-0.0513`

## Top 15 local ridge features

- `lag_14__T_place_TSIDEUPPER`: coefficient `0.001549`, |coef| `0.001549`
- `lag_11__T_place_LONGDOG`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_13__T_place_TSIDEUPPER`: coefficient `0.001481`, |coef| `0.001481`
- `lag_06__T_place_LONGDOG`: coefficient `-0.001451`, |coef| `0.001451`
- `lag_14__T_place_BACKOFB`: coefficient `-0.001305`, |coef| `0.001305`
- `lag_09__CT_place_CONNECTOR`: coefficient `0.001304`, |coef| `0.001304`
- `lag_06__CT1__duck_amount`: coefficient `0.001285`, |coef| `0.001285`
- `lag_07__T_place_LONGDOG`: coefficient `-0.001279`, |coef| `0.001279`
- `lag_13__T_place_BACKOFB`: coefficient `-0.001278`, |coef| `0.001278`
- `lag_00__CT1__duck_amount`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_12__T_place_LONGDOG`: coefficient `-0.001191`, |coef| `0.001191`
- `lag_10__CT_place_CONNECTOR`: coefficient `0.001182`, |coef| `0.001182`
- `lag_08__CT_place_CONNECTOR`: coefficient `0.001180`, |coef| `0.001180`
- `lag_15__T_place_BACKOFB`: coefficient `-0.001177`, |coef| `0.001177`
- `lag_10__T_place_LONGDOG`: coefficient `-0.001177`, |coef| `0.001177`

## Top 10 utility ridge features

- `lag_01__CT2__flash_duration`: coefficient `0.000921` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000908` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000831` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `-0.000699` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000646` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.000634` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000558` (raises CT win probability)
- `lag_01__T4__molly`: coefficient `0.000543` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000541` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000528` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_TSIDEUPPER`: coefficient `0.001549` (raises CT win probability)
- `lag_11__T_place_LONGDOG`: coefficient `-0.001529` (lowers CT win probability)
- `lag_13__T_place_TSIDEUPPER`: coefficient `0.001481` (raises CT win probability)
- `lag_06__T_place_LONGDOG`: coefficient `-0.001451` (lowers CT win probability)
- `lag_14__T_place_BACKOFB`: coefficient `-0.001305` (lowers CT win probability)
- `lag_09__CT_place_CONNECTOR`: coefficient `0.001304` (raises CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `0.001285` (raises CT win probability)
- `lag_07__T_place_LONGDOG`: coefficient `-0.001279` (lowers CT win probability)
- `lag_13__T_place_BACKOFB`: coefficient `-0.001278` (lowers CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `-0.001227` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2485`, seconds `19.00`, LSTM delta `-0.0969`

Top all feature movements:
- `lag_11__T_place_LONGDOG`: contribution `-0.007115`
- `lag_07__T_place_LONGDOG`: contribution `-0.005950`
- `lag_12__T_place_LONGDOG`: contribution `-0.005542`
- `lag_06__T5__duck_amount`: contribution `-0.004057`
- `lag_14__T_place_TSIDEUPPER`: contribution `-0.003907`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4309`, seconds `47.50`, LSTM delta `-0.0964`

Top all feature movements:
- `lag_10__CT_place_LONGDOG`: contribution `-0.007612`
- `lag_09__T_flash_duration_sum`: contribution `-0.007352`
- `lag_05__CT_place_ENTRANCE`: contribution `-0.004570`
- `lag_14__T_place_TSIDEUPPER`: contribution `-0.003907`
- `lag_00__T5__duck_amount`: contribution `-0.003863`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `-0.007352`
- `lag_09__T5__flash_duration`: contribution `-0.003570`
- `lag_09__T3__flash_duration`: contribution `-0.002635`
- `lag_09__T4__flash_duration`: contribution `-0.002377`
- `lag_09__T2__flash_duration`: contribution `-0.002364`

### tick `2453`, seconds `18.50`, LSTM delta `-0.0906`

Top all feature movements:
- `lag_11__T_place_LONGDOG`: contribution `-0.007115`
- `lag_06__T_place_LONGDOG`: contribution `-0.006754`
- `lag_10__T_place_LONGDOG`: contribution `-0.005478`
- `lag_06__CT_place_CONNECTOR`: contribution `-0.003930`
- `lag_00__T5__duck_amount`: contribution `-0.003863`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `2581`, seconds `20.50`, LSTM delta `-0.0602`

Top all feature movements:
- `lag_10__T_place_LONGDOG`: contribution `-0.005478`
- `lag_04__CT_place_LONGDOG`: contribution `-0.005228`
- `lag_14__T_place_LONGDOG`: contribution `-0.004834`
- `lag_10__CT_place_CONNECTOR`: contribution `-0.004227`
- `lag_14__T_place_BACKOFB`: contribution `+0.003504`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3477`, seconds `34.50`, LSTM delta `+0.0599`

Top all feature movements:
- `lag_00__bomb_events_last_5s`: contribution `+0.004764`
- `lag_06__T5__duck_amount`: contribution `+0.004057`
- `lag_08__CT2__flash_duration`: contribution `+0.004006`
- `lag_00__CT4__duck_amount`: contribution `+0.003824`
- `lag_07__CT_place_CONNECTOR`: contribution `-0.003249`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `+0.004006`
