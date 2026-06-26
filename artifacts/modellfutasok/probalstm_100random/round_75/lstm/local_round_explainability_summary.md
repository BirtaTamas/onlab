# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `121734`, seconds `74.00`, LSTM `0.7768`, delta `+0.2314`
- tick `121702`, seconds `73.50`, LSTM `0.5454`, delta `+0.1059`
- tick `122758`, seconds `90.00`, LSTM `0.9062`, delta `+0.0995`
- tick `121510`, seconds `70.50`, LSTM `0.4576`, delta `-0.0768`
- tick `121766`, seconds `74.50`, LSTM `0.7086`, delta `-0.0682`
- tick `122694`, seconds `89.00`, LSTM `0.7684`, delta `-0.0605`
- tick `121798`, seconds `75.00`, LSTM `0.7687`, delta `+0.0601`
- tick `123398`, seconds `100.00`, LSTM `0.9569`, delta `+0.0580`
- tick `121830`, seconds `75.50`, LSTM `0.8241`, delta `+0.0553`
- tick `122598`, seconds `87.50`, LSTM `0.8175`, delta `+0.0470`

## Top 15 local ridge features

- `lag_13__T_place_CONTROL`: coefficient `-0.001522`, |coef| `0.001522`
- `lag_00__kill_diff_last_3s`: coefficient `0.001459`, |coef| `0.001459`
- `lag_11__CT_place_LOBBY`: coefficient `0.001416`, |coef| `0.001416`
- `lag_11__CT_place_HUT`: coefficient `-0.001390`, |coef| `0.001390`
- `lag_02__T_place_CONTROL`: coefficient `-0.001339`, |coef| `0.001339`
- `lag_00__CT_kills_last_3s`: coefficient `0.001200`, |coef| `0.001200`
- `lag_09__T_place_MINI`: coefficient `0.001186`, |coef| `0.001186`
- `lag_09__CT_place_VENTS`: coefficient `-0.001136`, |coef| `0.001136`
- `lag_00__damage_diff_last_5s`: coefficient `0.001129`, |coef| `0.001129`
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001108`, |coef| `0.001108`
- `lag_00__T_place_RAMP`: coefficient `-0.001097`, |coef| `0.001097`
- `lag_15__T_place_CONTROL`: coefficient `-0.001042`, |coef| `0.001042`
- `lag_07__CT_place_LOBBY`: coefficient `-0.001038`, |coef| `0.001038`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001033`, |coef| `0.001033`
- `lag_10__CT_place_HUT`: coefficient `-0.001023`, |coef| `0.001023`

## Top 10 utility ridge features

- `lag_01__T2__smoke`: coefficient `-0.000488` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000418` (lowers CT win probability)
- `lag_01__CT_he_last_5s`: coefficient `0.000312` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000307` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000270` (lowers CT win probability)
- `lag_03__CT_he_last_5s`: coefficient `0.000258` (raises CT win probability)
- `lag_06__CT_he_last_5s`: coefficient `0.000219` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000219` (raises CT win probability)
- `lag_09__CT_he_last_5s`: coefficient `0.000217` (raises CT win probability)
- `lag_14__CT_A_site_active_smokes`: coefficient `0.000214` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_CONTROL`: coefficient `-0.001522` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001459` (raises CT win probability)
- `lag_11__CT_place_LOBBY`: coefficient `0.001416` (raises CT win probability)
- `lag_11__CT_place_HUT`: coefficient `-0.001390` (lowers CT win probability)
- `lag_02__T_place_CONTROL`: coefficient `-0.001339` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001200` (raises CT win probability)
- `lag_09__T_place_MINI`: coefficient `0.001186` (raises CT win probability)
- `lag_09__CT_place_VENTS`: coefficient `-0.001136` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001129` (raises CT win probability)
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001108` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121734`, seconds `74.00`, LSTM delta `+0.2314`

Top all feature movements:
- `lag_11__CT_place_HUT`: contribution `+0.013559`
- `lag_11__CT_place_LOBBY`: contribution `+0.011590`
- `lag_13__T_place_CONTROL`: contribution `+0.010816`
- `lag_09__CT_place_VENTS`: contribution `+0.009529`
- `lag_02__T_place_CONTROL`: contribution `+0.009512`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121702`, seconds `73.50`, LSTM delta `+0.1059`

Top all feature movements:
- `lag_10__CT_place_HUT`: contribution `+0.009979`
- `lag_10__CT_place_LOBBY`: contribution `+0.007818`
- `lag_00__kill_diff_last_3s`: contribution `+0.007024`
- `lag_01__T_place_CONTROL`: contribution `+0.006420`
- `lag_00__CT_place_OBSERVATION`: contribution `+0.006267`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122758`, seconds `90.00`, LSTM delta `+0.0995`

Top all feature movements:
- `lag_09__T_place_MINI`: contribution `+0.016499`
- `lag_15__CT_place_OBSERVATION`: contribution `+0.012026`
- `lag_05__CT_place_CONTROL`: contribution `+0.007492`
- `lag_00__T_place_HUT`: contribution `+0.007015`
- `lag_07__T_place_HUT`: contribution `+0.005662`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121510`, seconds `70.50`, LSTM delta `-0.0768`

Top all feature movements:
- `lag_00__CT_place_LOBBY`: contribution `-0.007853`
- `lag_12__CT_place_HUT`: contribution `-0.004547`
- `lag_04__CT_place_HUT`: contribution `-0.004451`
- `lag_02__CT_place_VENTS`: contribution `-0.004175`
- `lag_00__kill_diff_last_3s`: contribution `-0.003512`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121766`, seconds `74.50`, LSTM delta `-0.0682`

Top all feature movements:
- `lag_00__CT_place_HUT`: contribution `-0.005868`
- `lag_08__CT_place_LOBBY`: contribution `-0.005564`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005025`
- `lag_14__T_place_CONTROL`: contribution `+0.004769`
- `lag_12__CT_place_HUT`: contribution `+0.004547`

Top utility-only movements:
- No utility movement among the top local contributors.
