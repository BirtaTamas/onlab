# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `6107`, seconds `75.50`, LSTM `0.7299`, delta `+0.2018`
- tick `2203`, seconds `14.50`, LSTM `0.6306`, delta `+0.1142`
- tick `6875`, seconds `87.50`, LSTM `0.9463`, delta `+0.0816`
- tick `6427`, seconds `80.50`, LSTM `0.9106`, delta `+0.0557`
- tick `6299`, seconds `78.50`, LSTM `0.7587`, delta `+0.0480`
- tick `6395`, seconds `80.00`, LSTM `0.8549`, delta `+0.0471`
- tick `2235`, seconds `15.00`, LSTM `0.6764`, delta `+0.0457`
- tick `2459`, seconds `18.50`, LSTM `0.6013`, delta `-0.0426`
- tick `5211`, seconds `61.50`, LSTM `0.6771`, delta `+0.0420`
- tick `6843`, seconds `87.00`, LSTM `0.8647`, delta `-0.0385`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002392`, |coef| `0.002392`
- `lag_02__CT_place_UNDERPASS`: coefficient `0.002290`, |coef| `0.002290`
- `lag_15__CT_place_TRUCK`: coefficient `-0.002035`, |coef| `0.002035`
- `lag_00__T1__duck_amount`: coefficient `-0.001987`, |coef| `0.001987`
- `lag_00__kill_diff_last_3s`: coefficient `0.001869`, |coef| `0.001869`
- `lag_13__T_place_PALACEINTERIOR`: coefficient `0.001853`, |coef| `0.001853`
- `lag_13__T_place_TRAMP`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_00__damage_diff_last_5s`: coefficient `0.001689`, |coef| `0.001689`
- `lag_00__CT_damage_last_5s`: coefficient `0.001683`, |coef| `0.001683`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001658`, |coef| `0.001658`
- `lag_00__CT3__duck_amount`: coefficient `0.001599`, |coef| `0.001599`
- `lag_07__CT_place_TRUCK`: coefficient `0.001525`, |coef| `0.001525`
- `lag_04__T3__flash_duration`: coefficient `0.001495`, |coef| `0.001495`
- `lag_14__CT1__duck_amount`: coefficient `-0.001462`, |coef| `0.001462`
- `lag_02__CT_place_CATWALK`: coefficient `-0.001435`, |coef| `0.001435`

## Top 10 utility ridge features

- `lag_04__T3__flash_duration`: coefficient `0.001495` (raises CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `0.001133` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000957` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000825` (raises CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000817` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000811` (lowers CT win probability)
- `lag_08__CT5__flash`: coefficient `-0.000792` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000773` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000704` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.000617` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002392` (raises CT win probability)
- `lag_02__CT_place_UNDERPASS`: coefficient `0.002290` (raises CT win probability)
- `lag_15__CT_place_TRUCK`: coefficient `-0.002035` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.001987` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001869` (raises CT win probability)
- `lag_13__T_place_PALACEINTERIOR`: coefficient `0.001853` (raises CT win probability)
- `lag_13__T_place_TRAMP`: coefficient `-0.001700` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001689` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001683` (raises CT win probability)
- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001658` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `6107`, seconds `75.50`, LSTM delta `+0.2018`

Top all feature movements:
- `lag_02__CT_place_UNDERPASS`: contribution `+0.013280`
- `lag_04__T3__flash_duration`: contribution `+0.008509`
- `lag_00__CT_kills_last_3s`: contribution `+0.006906`
- `lag_13__T_place_PALACEINTERIOR`: contribution `+0.006216`
- `lag_02__CT_place_CATWALK`: contribution `+0.005717`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.008509`

### tick `2203`, seconds `14.50`, LSTM delta `+0.1142`

Top all feature movements:
- `lag_07__CT_place_TRUCK`: contribution `+0.009834`
- `lag_00__CT_kills_last_3s`: contribution `+0.006906`
- `lag_06__CT_place_TRUCK`: contribution `-0.005406`
- `lag_00__kill_diff_last_3s`: contribution `+0.004498`
- `lag_09__T_place_HOUSE`: contribution `+0.004460`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.003285`

### tick `6875`, seconds `87.50`, LSTM delta `+0.0816`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.006906`
- `lag_00__T_duck_amount_mean`: contribution `+0.006596`
- `lag_00__CT3__duck_amount`: contribution `+0.005951`
- `lag_00__kill_diff_last_3s`: contribution `+0.004498`
- `lag_09__CT_place_JUNGLE`: contribution `+0.003676`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6427`, seconds `80.50`, LSTM delta `+0.0557`

Top all feature movements:
- `lag_04__T3__flash_duration`: contribution `-0.008509`
- `lag_00__CT_kills_last_3s`: contribution `+0.006906`
- `lag_00__CT3__duck_amount`: contribution `-0.005590`
- `lag_12__CT_place_UNDERPASS`: contribution `+0.005505`
- `lag_00__kill_diff_last_3s`: contribution `+0.004498`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `-0.008509`
- `lag_14__T3__flash_duration`: contribution `+0.004400`
- `lag_04__T_flash_duration_sum`: contribution `-0.001957`

### tick `6299`, seconds `78.50`, LSTM delta `+0.0480`

Top all feature movements:
- `lag_13__T_place_PALACEINTERIOR`: contribution `+0.006216`
- `lag_13__T_place_TRAMP`: contribution `+0.004975`
- `lag_00__kill_diff_last_3s`: contribution `-0.004498`
- `lag_00__CT_damage_last_5s`: contribution `+0.003670`
- `lag_12__CT_place_TRUCK`: contribution `+0.002935`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `+0.001609`
