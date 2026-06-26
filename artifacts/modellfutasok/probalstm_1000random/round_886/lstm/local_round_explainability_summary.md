# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `113275`, seconds `96.00`, LSTM `0.3723`, delta `+0.2791`
- tick `112891`, seconds `90.00`, LSTM `0.6221`, delta `+0.2665`
- tick `112859`, seconds `89.50`, LSTM `0.3556`, delta `+0.2626`
- tick `113051`, seconds `92.50`, LSTM `0.1237`, delta `-0.2605`
- tick `112955`, seconds `91.00`, LSTM `0.4247`, delta `-0.2216`
- tick `112603`, seconds `85.50`, LSTM `0.1012`, delta `-0.1949`
- tick `113499`, seconds `99.50`, LSTM `0.7073`, delta `+0.1871`
- tick `113403`, seconds `98.00`, LSTM `0.5303`, delta `+0.1200`
- tick `113019`, seconds `92.00`, LSTM `0.3842`, delta `-0.0695`
- tick `113595`, seconds `101.00`, LSTM `0.6453`, delta `-0.0690`

## Top 15 local ridge features

- `lag_11__T_place_TRUCK`: coefficient `-0.004617`, |coef| `0.004617`
- `lag_00__kill_diff_last_3s`: coefficient `0.002663`, |coef| `0.002663`
- `lag_00__damage_diff_last_5s`: coefficient `0.002650`, |coef| `0.002650`
- `lag_04__CT_place_BACKALLEY`: coefficient `-0.002639`, |coef| `0.002639`
- `lag_08__CT_place_SIDEALLEY`: coefficient `0.002610`, |coef| `0.002610`
- `lag_03__CT_place_SIDEALLEY`: coefficient `-0.002419`, |coef| `0.002419`
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.002363`, |coef| `0.002363`
- `lag_07__CT_place_TSPAWN`: coefficient `-0.002115`, |coef| `0.002115`
- `lag_10__CT_place_SIDEALLEY`: coefficient `-0.002099`, |coef| `0.002099`
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002017`, |coef| `0.002017`
- `lag_14__T4__duck_amount`: coefficient `-0.001924`, |coef| `0.001924`
- `lag_00__CT_place_SIDEALLEY`: coefficient `-0.001904`, |coef| `0.001904`
- `lag_05__T_place_TRUCK`: coefficient `0.001903`, |coef| `0.001903`
- `lag_07__CT_place_SIDEALLEY`: coefficient `0.001872`, |coef| `0.001872`
- `lag_15__CT_place_BACKALLEY`: coefficient `0.001845`, |coef| `0.001845`

## Top 10 utility ridge features

- `lag_06__T5__flash_duration`: coefficient `0.001489` (raises CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001488` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.001373` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.001249` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.000987` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.000980` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.000976` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `0.000936` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `-0.000932` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.000924` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_TRUCK`: coefficient `-0.004617` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002663` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002650` (raises CT win probability)
- `lag_04__CT_place_BACKALLEY`: coefficient `-0.002639` (lowers CT win probability)
- `lag_08__CT_place_SIDEALLEY`: coefficient `0.002610` (raises CT win probability)
- `lag_03__CT_place_SIDEALLEY`: coefficient `-0.002419` (lowers CT win probability)
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.002363` (lowers CT win probability)
- `lag_07__CT_place_TSPAWN`: coefficient `-0.002115` (lowers CT win probability)
- `lag_10__CT_place_SIDEALLEY`: coefficient `-0.002099` (lowers CT win probability)
- `lag_00__T_place_APARTMENTS`: coefficient `-0.002017` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `113275`, seconds `96.00`, LSTM delta `+0.2791`

Top all feature movements:
- `lag_11__T_place_TRUCK`: contribution `+0.080177`
- `lag_09__CT_place_BACKALLEY`: contribution `+0.024124`
- `lag_00__damage_diff_last_5s`: contribution `+0.010342`
- `lag_07__CT2__flash_duration`: contribution `+0.009404`
- `lag_13__T5__flash_duration`: contribution `+0.006749`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.009404`
- `lag_13__T5__flash_duration`: contribution `+0.006749`
- `lag_07__CT_flash_duration_sum`: contribution `+0.002310`

### tick `112891`, seconds `90.00`, LSTM delta `+0.2665`

Top all feature movements:
- `lag_08__CT_place_SIDEALLEY`: contribution `+0.047628`
- `lag_04__CT_place_SIDEALLEY`: contribution `+0.043112`
- `lag_06__T_place_TRUCK`: contribution `+0.027462`
- `lag_08__CT_place_TSPAWN`: contribution `+0.012606`
- `lag_06__T5__flash_duration`: contribution `+0.010296`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.010296`
- `lag_01__T5__flash_duration`: contribution `+0.006393`
- `lag_06__CT2__flash_duration`: contribution `+0.005109`
- `lag_08__CT1__flash_duration`: contribution `+0.004863`

### tick `112859`, seconds `89.50`, LSTM delta `+0.2626`

Top all feature movements:
- `lag_03__CT_place_SIDEALLEY`: contribution `+0.044145`
- `lag_07__CT_place_SIDEALLEY`: contribution `+0.034152`
- `lag_05__T_place_TRUCK`: contribution `+0.033048`
- `lag_07__CT_place_TSPAWN`: contribution `+0.015836`
- `lag_00__T5__flash_duration`: contribution `+0.008640`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.008640`
- `lag_05__T5__flash_duration`: contribution `+0.006782`
- `lag_05__CT2__flash_duration`: contribution `+0.006414`
- `lag_07__CT1__flash_duration`: contribution `+0.002870`

### tick `113051`, seconds `92.50`, LSTM delta `-0.2605`

Top all feature movements:
- `lag_11__T_place_TRUCK`: contribution `-0.080177`
- `lag_09__CT_place_SIDEALLEY`: contribution `-0.021897`
- `lag_13__CT_place_SIDEALLEY`: contribution `-0.021437`
- `lag_00__kill_diff_last_3s`: contribution `-0.012821`
- `lag_04__T_place_TRUCK`: contribution `-0.010837`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `-0.010296`
- `lag_11__CT2__flash_duration`: contribution `-0.006182`
- `lag_11__T5__flash_duration`: contribution `-0.003672`

### tick `112955`, seconds `91.00`, LSTM delta `-0.2216`

Top all feature movements:
- `lag_10__CT_place_SIDEALLEY`: contribution `-0.038295`
- `lag_01__T_place_TRUCK`: contribution `-0.027775`
- `lag_08__T_place_TRUCK`: contribution `-0.022496`
- `lag_06__CT_place_SIDEALLEY`: contribution `-0.015341`
- `lag_08__CT2__flash_duration`: contribution `-0.006760`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.006760`
- `lag_08__T5__flash_duration`: contribution `-0.006445`
- `lag_03__T5__flash_duration`: contribution `-0.003396`
- `lag_10__CT1__flash_duration`: contribution `-0.003118`
