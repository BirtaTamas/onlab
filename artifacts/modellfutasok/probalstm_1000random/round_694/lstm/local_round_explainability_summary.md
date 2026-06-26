# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `17`

## Largest probability jumps

- tick `117378`, seconds `20.00`, LSTM `0.0592`, delta `-0.2805`
- tick `117314`, seconds `19.00`, LSTM `0.3654`, delta `-0.1735`
- tick `116386`, seconds `4.50`, LSTM `0.6198`, delta `+0.0555`
- tick `116354`, seconds `4.00`, LSTM `0.5643`, delta `+0.0366`
- tick `117282`, seconds `18.50`, LSTM `0.5389`, delta `-0.0310`
- tick `117250`, seconds `18.00`, LSTM `0.5699`, delta `+0.0293`
- tick `117346`, seconds `19.50`, LSTM `0.3398`, delta `-0.0256`
- tick `116578`, seconds `7.50`, LSTM `0.6045`, delta `+0.0247`
- tick `118466`, seconds `37.00`, LSTM `0.0151`, delta `-0.0213`
- tick `120578`, seconds `70.00`, LSTM `0.0494`, delta `+0.0212`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.002191`, |coef| `0.002191`
- `lag_15__CT_place_OUTSIDELONG`: coefficient `-0.002053`, |coef| `0.002053`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `0.001799`, |coef| `0.001799`
- `lag_10__CT_place_BRIDGE`: coefficient `-0.001773`, |coef| `0.001773`
- `lag_15__T1__flash_duration`: coefficient `0.001700`, |coef| `0.001700`
- `lag_00__CT_place_BRIDGE`: coefficient `0.001555`, |coef| `0.001555`
- `lag_15__CT_place_BRIDGE`: coefficient `0.001515`, |coef| `0.001515`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.001459`, |coef| `0.001459`
- `lag_15__CT1__duck_amount`: coefficient `-0.001138`, |coef| `0.001138`
- `lag_00__CT4__duck_amount`: coefficient `-0.001102`, |coef| `0.001102`
- `lag_10__T3__flash_duration`: coefficient `0.001064`, |coef| `0.001064`
- `lag_07__T_place_TSTAIRS`: coefficient `0.001054`, |coef| `0.001054`
- `lag_00__T_kills_last_3s`: coefficient `-0.001000`, |coef| `0.001000`
- `lag_08__CT_place_BRIDGE`: coefficient `-0.000981`, |coef| `0.000981`
- `lag_07__CT_place_OUTSIDELONG`: coefficient `-0.000969`, |coef| `0.000969`

## Top 10 utility ridge features

- `lag_15__T1__flash_duration`: coefficient `0.001700` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.001064` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000762` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000732` (raises CT win probability)
- `lag_13__T1__flash_duration`: coefficient `0.000729` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000711` (raises CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.000659` (raises CT win probability)
- `lag_15__T1__utility_total`: coefficient `0.000650` (raises CT win probability)
- `lag_15__CT1__molly`: coefficient `0.000647` (raises CT win probability)
- `lag_03__T4__molly`: coefficient `0.000633` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.002191` (raises CT win probability)
- `lag_15__CT_place_OUTSIDELONG`: coefficient `-0.002053` (lowers CT win probability)
- `lag_02__CT_place_OUTSIDELONG`: coefficient `0.001799` (raises CT win probability)
- `lag_10__CT_place_BRIDGE`: coefficient `-0.001773` (lowers CT win probability)
- `lag_00__CT_place_BRIDGE`: coefficient `0.001555` (raises CT win probability)
- `lag_15__CT_place_BRIDGE`: coefficient `0.001515` (raises CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.001459` (lowers CT win probability)
- `lag_15__CT1__duck_amount`: coefficient `-0.001138` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `-0.001102` (lowers CT win probability)
- `lag_07__T_place_TSTAIRS`: coefficient `0.001054` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `117378`, seconds `20.00`, LSTM delta `-0.2805`

Top all feature movements:
- `lag_10__CT_place_BRIDGE`: contribution `-0.020328`
- `lag_02__CT_place_OUTSIDELONG`: contribution `-0.018244`
- `lag_00__CT_place_BRIDGE`: contribution `-0.017822`
- `lag_15__CT_place_BRIDGE`: contribution `-0.017370`
- `lag_15__T1__flash_duration`: contribution `-0.012527`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `-0.012527`
- `lag_10__T3__flash_duration`: contribution `-0.004988`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.002690`

### tick `117314`, seconds `19.00`, LSTM delta `-0.1735`

Top all feature movements:
- `lag_15__CT_place_OUTSIDELONG`: contribution `-0.041647`
- `lag_08__CT_place_BRIDGE`: contribution `-0.011241`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.007021`
- `lag_13__CT_place_BRIDGE`: contribution `-0.006114`
- `lag_13__T1__flash_duration`: contribution `-0.005370`

Top utility-only movements:
- `lag_13__T1__flash_duration`: contribution `-0.005370`
- `lag_08__T3__flash_duration`: contribution `-0.002573`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.001944`
- `lag_00__CT5__molly`: contribution `-0.001479`

### tick `116386`, seconds `4.50`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.021441`
- `lag_09__CT_place_CTSIDEUPPER`: contribution `+0.014616`
- `lag_00__CT_place_ALLEY`: contribution `+0.003151`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.002944`
- `lag_02__T3__duck_amount`: contribution `-0.002250`

Top utility-only movements:
- `lag_09__CT1__molly`: contribution `+0.000362`
- `lag_09__CT3__molly`: contribution `-0.000341`
- `lag_09__T1__utility_total`: contribution `+0.000306`
- `lag_09__CT4__utility_total`: contribution `+0.000306`
- `lag_09__CT1__utility_total`: contribution `+0.000292`

### tick `116354`, seconds `4.00`, LSTM delta `+0.0366`

Top all feature movements:
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.010721`
- `lag_08__CT_place_CTSIDEUPPER`: contribution `+0.007951`
- `lag_03__CT_place_CTSIDEUPPER`: contribution `-0.004766`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.002944`
- `lag_01__T3__duck_amount`: contribution `+0.002484`

Top utility-only movements:
- `lag_08__CT4__utility_total`: contribution `+0.000379`
- `lag_08__CT3__molly`: contribution `-0.000352`

### tick `117282`, seconds `18.50`, LSTM delta `-0.0310`

Top all feature movements:
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.012112`
- `lag_07__CT_place_BRIDGE`: contribution `-0.004297`
- `lag_04__T_place_STREET`: contribution `+0.003322`
- `lag_12__CT1__duck_amount`: contribution `-0.001660`
- `lag_01__CT5__duck_amount`: contribution `+0.001501`

Top utility-only movements:
- `lag_12__T1__flash_duration`: contribution `-0.001219`
- `lag_07__T3__flash_duration`: contribution `-0.000753`
