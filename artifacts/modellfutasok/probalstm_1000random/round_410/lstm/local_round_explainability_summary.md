# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `119069`, seconds `52.50`, LSTM `0.0348`, delta `-0.1019`
- tick `116765`, seconds `16.50`, LSTM `0.1535`, delta `-0.0932`
- tick `116637`, seconds `14.50`, LSTM `0.2665`, delta `-0.0602`
- tick `118781`, seconds `48.00`, LSTM `0.1126`, delta `-0.0392`
- tick `116221`, seconds `8.00`, LSTM `0.3117`, delta `-0.0357`
- tick `116093`, seconds `6.00`, LSTM `0.3739`, delta `-0.0328`
- tick `116477`, seconds `12.00`, LSTM `0.3262`, delta `-0.0303`
- tick `118749`, seconds `47.50`, LSTM `0.1518`, delta `+0.0293`
- tick `118461`, seconds `43.00`, LSTM `0.1227`, delta `+0.0288`
- tick `115901`, seconds `3.00`, LSTM `0.3747`, delta `+0.0267`

## Top 15 local ridge features

- `lag_00__CT5__duck_amount`: coefficient `-0.001457`, |coef| `0.001457`
- `lag_08__CT_place_TOPOFMID`: coefficient `-0.001022`, |coef| `0.001022`
- `lag_08__T_place_UNDERPASS`: coefficient `-0.000952`, |coef| `0.000952`
- `lag_00__T_he_last_5s`: coefficient `0.000951`, |coef| `0.000951`
- `lag_00__CT1__is_walking`: coefficient `0.000919`, |coef| `0.000919`
- `lag_14__CT_place_TOPOFMID`: coefficient `-0.000873`, |coef| `0.000873`
- `lag_09__CT_place_TOPOFMID`: coefficient `-0.000856`, |coef| `0.000856`
- `lag_10__T_place_KITCHEN`: coefficient `0.000801`, |coef| `0.000801`
- `lag_00__T_kills_last_3s`: coefficient `-0.000758`, |coef| `0.000758`
- `lag_09__T2__duck_amount`: coefficient `-0.000726`, |coef| `0.000726`
- `lag_05__CT_place_TOPOFMID`: coefficient `-0.000716`, |coef| `0.000716`
- `lag_00__T_place_UNDERPASS`: coefficient `-0.000669`, |coef| `0.000669`
- `lag_06__CT_place_TOPOFMID`: coefficient `-0.000668`, |coef| `0.000668`
- `lag_03__CT_place_TOPOFMID`: coefficient `-0.000667`, |coef| `0.000667`
- `lag_13__T_place_UPSTAIRS`: coefficient `0.000664`, |coef| `0.000664`

## Top 10 utility ridge features

- `lag_00__T_he_last_5s`: coefficient `0.000951` (raises CT win probability)
- `lag_06__T_he_last_5s`: coefficient `-0.000536` (lowers CT win probability)
- `lag_10__T2__smoke`: coefficient `0.000377` (raises CT win probability)
- `lag_13__T_he_last_5s`: coefficient `0.000376` (raises CT win probability)
- `lag_14__T_he_last_5s`: coefficient `0.000327` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `0.000303` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.000292` (lowers CT win probability)
- `lag_02__T_he_last_5s`: coefficient `0.000289` (raises CT win probability)
- `lag_04__T_he_last_5s`: coefficient `0.000286` (raises CT win probability)
- `lag_01__T_he_last_5s`: coefficient `0.000281` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT5__duck_amount`: coefficient `-0.001457` (lowers CT win probability)
- `lag_08__CT_place_TOPOFMID`: coefficient `-0.001022` (lowers CT win probability)
- `lag_08__T_place_UNDERPASS`: coefficient `-0.000952` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000919` (raises CT win probability)
- `lag_14__CT_place_TOPOFMID`: coefficient `-0.000873` (lowers CT win probability)
- `lag_09__CT_place_TOPOFMID`: coefficient `-0.000856` (lowers CT win probability)
- `lag_10__T_place_KITCHEN`: coefficient `0.000801` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000758` (lowers CT win probability)
- `lag_09__T2__duck_amount`: coefficient `-0.000726` (lowers CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `-0.000716` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `119069`, seconds `52.50`, LSTM delta `-0.1019`

Top all feature movements:
- `lag_00__CT5__duck_amount`: contribution `-0.005498`
- `lag_08__T_place_UNDERPASS`: contribution `-0.003728`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.003710`
- `lag_14__CT_place_TOPOFMID`: contribution `-0.003169`
- `lag_02__T1__duck_amount`: contribution `-0.002486`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116765`, seconds `16.50`, LSTM delta `-0.0932`

Top all feature movements:
- `lag_10__T_place_KITCHEN`: contribution `-0.025592`
- `lag_13__T_place_KITCHEN`: contribution `-0.015120`
- `lag_13__T_place_UPSTAIRS`: contribution `-0.011207`
- `lag_10__T_place_DECK`: contribution `-0.007573`
- `lag_00__T_kills_last_3s`: contribution `-0.002402`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.001062`

### tick `116637`, seconds `14.50`, LSTM delta `-0.0602`

Top all feature movements:
- `lag_06__T_place_KITCHEN`: contribution `-0.015587`
- `lag_06__T_place_DECK`: contribution `-0.015135`
- `lag_14__T_place_UPSTAIRS`: contribution `-0.005299`
- `lag_09__T_place_KITCHEN`: contribution `-0.005075`
- `lag_13__T_he_last_5s`: contribution `-0.004908`

Top utility-only movements:
- `lag_13__T_he_last_5s`: contribution `-0.004908`
- `lag_00__T5__flash_duration`: contribution `-0.000906`

### tick `118781`, seconds `48.00`, LSTM delta `-0.0392`

Top all feature movements:
- `lag_00__CT5__duck_amount`: contribution `-0.005498`
- `lag_05__CT_place_TOPOFMID`: contribution `-0.002599`
- `lag_13__T4__duck_amount`: contribution `-0.002327`
- `lag_00__CT1__is_walking`: contribution `-0.002144`
- `lag_14__CT5__duck_amount`: contribution `-0.001730`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116221`, seconds `8.00`, LSTM delta `-0.0357`

Top all feature movements:
- `lag_00__T_he_last_5s`: contribution `-0.012408`
- `lag_01__T_place_UPSTAIRS`: contribution `-0.006002`
- `lag_10__T_he_last_5s`: contribution `-0.002976`
- `lag_04__CT_place_TOPOFMID`: contribution `-0.001755`
- `lag_06__CT_place_LIBRARY`: contribution `-0.001196`

Top utility-only movements:
- `lag_00__T_he_last_5s`: contribution `-0.012408`
- `lag_10__T_he_last_5s`: contribution `-0.002976`
