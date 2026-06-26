# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `10922`, seconds `1.00`, LSTM `0.2900`, delta `-0.0815`
- tick `11306`, seconds `7.00`, LSTM `0.1999`, delta `-0.0689`
- tick `12874`, seconds `31.50`, LSTM `0.0996`, delta `-0.0685`
- tick `13162`, seconds `36.00`, LSTM `0.0598`, delta `-0.0548`
- tick `10954`, seconds `1.50`, LSTM `0.2519`, delta `-0.0380`
- tick `11146`, seconds `4.50`, LSTM `0.2464`, delta `-0.0360`
- tick `11082`, seconds `3.50`, LSTM `0.2691`, delta `+0.0359`
- tick `12202`, seconds `21.00`, LSTM `0.2436`, delta `+0.0347`
- tick `12554`, seconds `26.50`, LSTM `0.2397`, delta `+0.0331`
- tick `12266`, seconds `22.00`, LSTM `0.1951`, delta `-0.0321`

## Top 15 local ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_02__CT_place_CTSPAWN`: coefficient `-0.000930`, |coef| `0.000930`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000899`, |coef| `0.000899`
- `lag_00__CT_place_UNDERA`: coefficient `-0.000795`, |coef| `0.000795`
- `lag_00__T_place_TSPAWN`: coefficient `0.000723`, |coef| `0.000723`
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.000720`, |coef| `0.000720`
- `lag_02__T_place_TSPAWN`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_01__T_flashes_last_5s`: coefficient `-0.000696`, |coef| `0.000696`
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000674`, |coef| `0.000674`
- `lag_02__T_place_LONGA`: coefficient `-0.000668`, |coef| `0.000668`
- `lag_15__CT2__is_scoped`: coefficient `-0.000654`, |coef| `0.000654`
- `lag_00__CT_place_BDOORS`: coefficient `-0.000604`, |coef| `0.000604`
- `lag_02__T2__has_bomb`: coefficient `-0.000579`, |coef| `0.000579`
- `lag_00__CT5__duck_amount`: coefficient `-0.000574`, |coef| `0.000574`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000569`, |coef| `0.000569`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.001490` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000899` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000696` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000569` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000556` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `-0.000525` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `0.000524` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000435` (lowers CT win probability)
- `lag_02__T_smoke_inv`: coefficient `-0.000418` (lowers CT win probability)
- `lag_02__T2__smoke`: coefficient `-0.000409` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_CTSPAWN`: coefficient `-0.000930` (lowers CT win probability)
- `lag_00__CT_place_UNDERA`: coefficient `-0.000795` (lowers CT win probability)
- `lag_00__T_place_TSPAWN`: coefficient `0.000723` (raises CT win probability)
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.000720` (lowers CT win probability)
- `lag_02__T_place_TSPAWN`: coefficient `-0.000717` (lowers CT win probability)
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000674` (raises CT win probability)
- `lag_02__T_place_LONGA`: coefficient `-0.000668` (lowers CT win probability)
- `lag_15__CT2__is_scoped`: coefficient `-0.000654` (lowers CT win probability)
- `lag_00__CT_place_BDOORS`: coefficient `-0.000604` (lowers CT win probability)
- `lag_02__T2__has_bomb`: coefficient `-0.000579` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `10922`, seconds `1.00`, LSTM delta `-0.0815`

Top all feature movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.013499`
- `lag_02__CT_place_CTSPAWN`: contribution `-0.004446`
- `lag_02__T_place_TSPAWN`: contribution `-0.003175`
- `lag_02__T2__has_bomb`: contribution `-0.001599`
- `lag_01__T_velocity_mean`: contribution `-0.001333`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.013499`
- `lag_00__T1__flash`: contribution `-0.001211`
- `lag_02__T_smoke_inv`: contribution `-0.000952`
- `lag_02__T4__flash`: contribution `-0.000714`
- `lag_02__T2__smoke`: contribution `-0.000635`

### tick `11306`, seconds `7.00`, LSTM delta `-0.0689`

Top all feature movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.003359`
- `lag_00__CT_place_BDOORS`: contribution `-0.002905`
- `lag_00__CT_flashed_players`: contribution `-0.002689`
- `lag_00__CT_place_UNDERA`: contribution `+0.002429`
- `lag_11__T_place_TOPOFMID`: contribution `-0.001888`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.003359`
- `lag_00__T1__flash_duration`: contribution `-0.001631`
- `lag_00__CT1__flash_duration`: contribution `-0.001455`
- `lag_00__CT5__flash_duration`: contribution `-0.000986`

### tick `12874`, seconds `31.50`, LSTM delta `-0.0685`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.006034`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.004044`
- `lag_06__T_place_PIT`: contribution `-0.003475`
- `lag_02__T_place_LONGA`: contribution `-0.002846`
- `lag_07__T5__is_scoped`: contribution `-0.002605`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.006034`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002413`

### tick `13162`, seconds `36.00`, LSTM delta `-0.0548`

Top all feature movements:
- `lag_15__CT2__is_scoped`: contribution `-0.004002`
- `lag_02__T_place_LONGA`: contribution `-0.002846`
- `lag_07__T5__is_scoped`: contribution `-0.002605`
- `lag_14__T_place_LONGA`: contribution `-0.002196`
- `lag_05__T4__flash_duration`: contribution `-0.001926`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.001926`
- `lag_05__CT2__flash_duration`: contribution `-0.001549`
- `lag_09__T_utility_damage_last_5s`: contribution `-0.001427`

### tick `10954`, seconds `1.50`, LSTM delta `-0.0380`

Top all feature movements:
- `lag_01__T_flashes_last_5s`: contribution `-0.006309`
- `lag_00__T_place_TOPOFMID`: contribution `-0.003108`
- `lag_03__CT_place_CTSPAWN`: contribution `-0.002530`
- `lag_00__CT_place_UNDERA`: contribution `-0.002429`
- `lag_00__T_place_TSPAWN`: contribution `-0.002059`

Top utility-only movements:
- `lag_01__T_flashes_last_5s`: contribution `-0.006309`
- `lag_01__T1__flash`: contribution `-0.000578`
- `lag_03__T_smoke_inv`: contribution `-0.000453`
- `lag_03__T2__smoke`: contribution `-0.000444`
- `lag_03__T4__flash`: contribution `-0.000407`
