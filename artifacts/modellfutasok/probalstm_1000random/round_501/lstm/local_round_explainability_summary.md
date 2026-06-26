# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `1`

## Largest probability jumps

- tick `4278`, seconds `47.00`, LSTM `0.7793`, delta `+0.1172`
- tick `7894`, seconds `103.50`, LSTM `0.9269`, delta `+0.0800`
- tick `4374`, seconds `48.50`, LSTM `0.8830`, delta `+0.0735`
- tick `3414`, seconds `33.50`, LSTM `0.5811`, delta `+0.0476`
- tick `7062`, seconds `90.50`, LSTM `0.8853`, delta `-0.0445`
- tick `3574`, seconds `36.00`, LSTM `0.6158`, delta `+0.0306`
- tick `7734`, seconds `101.00`, LSTM `0.8673`, delta `+0.0302`
- tick `3670`, seconds `37.50`, LSTM `0.6321`, delta `-0.0278`
- tick `3446`, seconds `34.00`, LSTM `0.6082`, delta `+0.0272`
- tick `3638`, seconds `37.00`, LSTM `0.6600`, delta `+0.0269`

## Top 15 local ridge features

- `lag_00__T_place_UPPERPARK`: coefficient `-0.001657`, |coef| `0.001657`
- `lag_04__CT5__flash_duration`: coefficient `0.001526`, |coef| `0.001526`
- `lag_00__CT_kills_last_3s`: coefficient `0.001270`, |coef| `0.001270`
- `lag_00__damage_diff_last_5s`: coefficient `0.001246`, |coef| `0.001246`
- `lag_01__T_place_UPPERPARK`: coefficient `-0.001191`, |coef| `0.001191`
- `lag_00__CT2__duck_amount`: coefficient `0.001182`, |coef| `0.001182`
- `lag_00__kill_diff_last_3s`: coefficient `0.001157`, |coef| `0.001157`
- `lag_00__T1__alive`: coefficient `-0.001112`, |coef| `0.001112`
- `lag_00__CT_damage_last_5s`: coefficient `0.001094`, |coef| `0.001094`
- `lag_04__CT_flashed_players`: coefficient `0.001041`, |coef| `0.001041`
- `lag_01__T_flashed_players`: coefficient `-0.001041`, |coef| `0.001041`
- `lag_00__T1__hp`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_00__T1__smoke`: coefficient `-0.000992`, |coef| `0.000992`
- `lag_04__CT_flash_duration_sum`: coefficient `0.000989`, |coef| `0.000989`
- `lag_04__T_flashed_players`: coefficient `0.000976`, |coef| `0.000976`

## Top 10 utility ridge features

- `lag_04__CT5__flash_duration`: coefficient `0.001526` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000992` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.000989` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000872` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.000866` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000750` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000665` (raises CT win probability)
- `lag_07__T2__flash`: coefficient `-0.000660` (lowers CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000650` (raises CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.000589` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_UPPERPARK`: coefficient `-0.001657` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001270` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001246` (raises CT win probability)
- `lag_01__T_place_UPPERPARK`: coefficient `-0.001191` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.001182` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001157` (raises CT win probability)
- `lag_00__T1__alive`: coefficient `-0.001112` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001094` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `0.001041` (raises CT win probability)
- `lag_01__T_flashed_players`: coefficient `-0.001041` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4278`, seconds `47.00`, LSTM delta `+0.1172`

Top all feature movements:
- `lag_00__T_place_UPPERPARK`: contribution `+0.008739`
- `lag_04__CT5__flash_duration`: contribution `+0.008355`
- `lag_06__T_place_PLAYGROUND`: contribution `+0.007351`
- `lag_04__T_flashed_players`: contribution `+0.005648`
- `lag_04__CT_flashed_players`: contribution `+0.004561`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.008355`
- `lag_04__CT_flash_duration_sum`: contribution `+0.003865`
- `lag_00__T1__smoke`: contribution `+0.002140`
- `lag_04__CT3__flash_duration`: contribution `+0.002005`

### tick `7894`, seconds `103.50`, LSTM delta `+0.0800`

Top all feature movements:
- `lag_14__CT_place_FOUNTAIN`: contribution `+0.009874`
- `lag_05__CT_place_FOUNTAIN`: contribution `+0.007931`
- `lag_03__CT_place_STAIRS`: contribution `+0.007358`
- `lag_01__T_place_UPPERPARK`: contribution `+0.006282`
- `lag_06__CT_place_STAIRS`: contribution `+0.005518`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4374`, seconds `48.50`, LSTM delta `+0.0735`

Top all feature movements:
- `lag_00__T_place_UPPERPARK`: contribution `+0.008739`
- `lag_09__T_place_PLAYGROUND`: contribution `+0.005788`
- `lag_07__CT5__flash_duration`: contribution `+0.004776`
- `lag_04__T_flashed_players`: contribution `-0.003766`
- `lag_00__CT_kills_last_3s`: contribution `+0.003666`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.004776`
- `lag_07__CT_flash_duration_sum`: contribution `+0.001782`

### tick `3414`, seconds `33.50`, LSTM delta `+0.0476`

Top all feature movements:
- `lag_01__T_place_UPPERPARK`: contribution `+0.006282`
- `lag_09__T_place_PLAYGROUND`: contribution `+0.005788`
- `lag_00__CT2__duck_amount`: contribution `+0.004503`
- `lag_01__T_flashed_players`: contribution `+0.004017`
- `lag_00__CT_place_STAIRS`: contribution `+0.003675`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7062`, seconds `90.50`, LSTM delta `-0.0445`

Top all feature movements:
- `lag_02__CT_place_TSTAIRS`: contribution `-0.018647`
- `lag_10__T_place_RESTROOM`: contribution `-0.010603`
- `lag_10__T_place_LOWERPARK`: contribution `-0.002819`
- `lag_00__kill_diff_last_3s`: contribution `-0.002784`
- `lag_08__CT2__duck_amount`: contribution `-0.002710`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.001433`
- `lag_02__CT5__flash_duration`: contribution `+0.001349`
- `lag_02__T2__flash_duration`: contribution `-0.001238`
