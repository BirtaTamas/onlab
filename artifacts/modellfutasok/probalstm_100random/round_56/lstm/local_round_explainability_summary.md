# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-big-vs-furia-bo3-8LyYppfzx0M6KmNUlhRuUi/big-vs-furia-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `122426`, seconds `75.00`, LSTM `0.2577`, delta `-0.2583`
- tick `120826`, seconds `50.00`, LSTM `0.8041`, delta `+0.2251`
- tick `120762`, seconds `49.00`, LSTM `0.5424`, delta `+0.2212`
- tick `122266`, seconds `72.50`, LSTM `0.7040`, delta `-0.1982`
- tick `122298`, seconds `73.00`, LSTM `0.5428`, delta `-0.1612`
- tick `120858`, seconds `50.50`, LSTM `0.9392`, delta `+0.1351`
- tick `121434`, seconds `59.50`, LSTM `0.8485`, delta `-0.1020`
- tick `120698`, seconds `48.00`, LSTM `0.2946`, delta `+0.0492`
- tick `122458`, seconds `75.50`, LSTM `0.2161`, delta `-0.0416`
- tick `117658`, seconds `0.50`, LSTM `0.2805`, delta `-0.0405`

## Top 15 local ridge features

- `lag_00__CT_place_TOPOFMID`: coefficient `0.003518`, |coef| `0.003518`
- `lag_01__CT_place_TOPOFMID`: coefficient `0.003303`, |coef| `0.003303`
- `lag_03__CT_place_TOPOFMID`: coefficient `0.002939`, |coef| `0.002939`
- `lag_02__CT_place_TOPOFMID`: coefficient `0.002727`, |coef| `0.002727`
- `lag_04__CT_place_TOPOFMID`: coefficient `0.002725`, |coef| `0.002725`
- `lag_00__kill_diff_last_3s`: coefficient `0.002490`, |coef| `0.002490`
- `lag_03__CT_shots_fired_sum`: coefficient `0.002211`, |coef| `0.002211`
- `lag_00__damage_diff_last_5s`: coefficient `0.002182`, |coef| `0.002182`
- `lag_05__CT_place_TOPOFMID`: coefficient `0.002181`, |coef| `0.002181`
- `lag_00__T_kills_last_3s`: coefficient `-0.001812`, |coef| `0.001812`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001753`, |coef| `0.001753`
- `lag_00__CT_macro_MID`: coefficient `0.001679`, |coef| `0.001679`
- `lag_01__CT_macro_MID`: coefficient `0.001597`, |coef| `0.001597`
- `lag_10__CT_place_BALCONY`: coefficient `-0.001472`, |coef| `0.001472`
- `lag_01__T1__shots_fired`: coefficient `0.001463`, |coef| `0.001463`

## Top 10 utility ridge features

- `lag_02__CT_flash_duration_sum`: coefficient `-0.001313` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.001304` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001227` (lowers CT win probability)
- `lag_07__CT1__flash_duration`: coefficient `-0.001193` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001146` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001134` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.001123` (raises CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.001107` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001059` (lowers CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.001037` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TOPOFMID`: coefficient `0.003518` (raises CT win probability)
- `lag_01__CT_place_TOPOFMID`: coefficient `0.003303` (raises CT win probability)
- `lag_03__CT_place_TOPOFMID`: coefficient `0.002939` (raises CT win probability)
- `lag_02__CT_place_TOPOFMID`: coefficient `0.002727` (raises CT win probability)
- `lag_04__CT_place_TOPOFMID`: coefficient `0.002725` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002490` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.002211` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002182` (raises CT win probability)
- `lag_05__CT_place_TOPOFMID`: coefficient `0.002181` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001812` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `122426`, seconds `75.00`, LSTM delta `-0.2583`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `-0.011057`
- `lag_03__CT_place_TOPOFMID`: contribution `-0.010667`
- `lag_07__CT_flash_duration_sum`: contribution `-0.010647`
- `lag_04__CT_place_TOPOFMID`: contribution `-0.009890`
- `lag_07__CT1__flash_duration`: contribution `-0.009028`

Top utility-only movements:
- `lag_07__CT_flash_duration_sum`: contribution `-0.010647`
- `lag_07__CT1__flash_duration`: contribution `-0.009028`
- `lag_07__CT5__flash_duration`: contribution `-0.007081`
- `lag_04__CT1__flash_duration`: contribution `-0.005671`
- `lag_05__CT5__flash_duration`: contribution `-0.005661`

### tick `120826`, seconds `50.00`, LSTM delta `+0.2251`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `+0.010753`
- `lag_03__CT_place_TOPOFMID`: contribution `+0.010667`
- `lag_00__CT_place_QUAD`: contribution `+0.010243`
- `lag_04__CT_place_TOPOFMID`: contribution `+0.009890`
- `lag_04__T1__flash_duration`: contribution `+0.008886`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.008886`
- `lag_04__T3__flash_duration`: contribution `+0.006682`
- `lag_04__T_flash_duration_sum`: contribution `+0.005637`

### tick `120762`, seconds `49.00`, LSTM delta `+0.2212`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.014456`
- `lag_01__CT_place_TOPOFMID`: contribution `+0.011987`
- `lag_02__CT_place_TOPOFMID`: contribution `+0.009895`
- `lag_10__CT_place_BALCONY`: contribution `+0.009450`
- `lag_02__T3__flash_duration`: contribution `+0.007385`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.007385`
- `lag_00__T1__flash_duration`: contribution `+0.005293`
- `lag_02__T_flash_duration_sum`: contribution `+0.002414`

### tick `122266`, seconds `72.50`, LSTM delta `-0.1982`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.023042`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.012766`
- `lag_02__CT_flash_duration_sum`: contribution `-0.010725`
- `lag_02__CT1__flash_duration`: contribution `-0.009287`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007591`

Top utility-only movements:
- `lag_02__CT_flash_duration_sum`: contribution `-0.010725`
- `lag_02__CT1__flash_duration`: contribution `-0.009287`
- `lag_02__CT5__flash_duration`: contribution `-0.007511`
- `lag_00__CT5__flash_duration`: contribution `-0.005185`

### tick `122298`, seconds `73.00`, LSTM delta `-0.1612`

Top all feature movements:
- `lag_00__CT_place_TOPOFMID`: contribution `-0.012766`
- `lag_01__CT_place_TOPOFMID`: contribution `-0.011987`
- `lag_03__CT_flash_duration_sum`: contribution `-0.009259`
- `lag_03__CT1__flash_duration`: contribution `-0.008379`
- `lag_03__CT5__flash_duration`: contribution `-0.006233`

Top utility-only movements:
- `lag_03__CT_flash_duration_sum`: contribution `-0.009259`
- `lag_03__CT1__flash_duration`: contribution `-0.008379`
- `lag_03__CT5__flash_duration`: contribution `-0.006233`
- `lag_00__CT1__flash_duration`: contribution `-0.003656`
- `lag_01__CT5__flash_duration`: contribution `-0.002366`
