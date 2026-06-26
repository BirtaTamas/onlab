# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `3`

## Largest probability jumps

- tick `22391`, seconds `46.50`, LSTM `0.8331`, delta `+0.1441`
- tick `23159`, seconds `58.50`, LSTM `0.8736`, delta `+0.1199`
- tick `26199`, seconds `106.00`, LSTM `0.8320`, delta `-0.1127`
- tick `25399`, seconds `93.50`, LSTM `0.8787`, delta `+0.1040`
- tick `25943`, seconds `102.00`, LSTM `0.9386`, delta `+0.0966`
- tick `26455`, seconds `110.00`, LSTM `0.7265`, delta `-0.0842`
- tick `21975`, seconds `40.00`, LSTM `0.7549`, delta `+0.0786`
- tick `22327`, seconds `45.50`, LSTM `0.6897`, delta `+0.0764`
- tick `23255`, seconds `60.00`, LSTM `0.8140`, delta `-0.0749`
- tick `24599`, seconds `81.00`, LSTM `0.6289`, delta `-0.0735`

## Top 15 local ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.002010`, |coef| `0.002010`
- `lag_07__CT_place_RESTROOM`: coefficient `-0.002009`, |coef| `0.002009`
- `lag_15__CT2__flash_duration`: coefficient `0.001971`, |coef| `0.001971`
- `lag_00__CT_walking_count`: coefficient `-0.001880`, |coef| `0.001880`
- `lag_00__CT2__is_walking`: coefficient `-0.001671`, |coef| `0.001671`
- `lag_00__kill_diff_last_3s`: coefficient `0.001663`, |coef| `0.001663`
- `lag_00__CT_kills_last_3s`: coefficient `0.001656`, |coef| `0.001656`
- `lag_03__T_place_UPPERPARK`: coefficient `-0.001626`, |coef| `0.001626`
- `lag_02__CT2__flash_duration`: coefficient `0.001547`, |coef| `0.001547`
- `lag_00__CT_place_RESTROOM`: coefficient `0.001524`, |coef| `0.001524`
- `lag_00__damage_diff_last_5s`: coefficient `0.001522`, |coef| `0.001522`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001440`, |coef| `0.001440`
- `lag_11__CT_place_BACKOFA`: coefficient `-0.001435`, |coef| `0.001435`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001343`, |coef| `0.001343`
- `lag_05__CT_place_BACKOFA`: coefficient `-0.001343`, |coef| `0.001343`

## Top 10 utility ridge features

- `lag_15__CT2__flash_duration`: coefficient `0.001971` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.001547` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001176` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001063` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001048` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001024` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `-0.001023` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001014` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000974` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000920` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.002010` (lowers CT win probability)
- `lag_07__CT_place_RESTROOM`: coefficient `-0.002009` (lowers CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.001880` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001671` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001663` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001656` (raises CT win probability)
- `lag_03__T_place_UPPERPARK`: coefficient `-0.001626` (lowers CT win probability)
- `lag_00__CT_place_RESTROOM`: coefficient `0.001524` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001522` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.001440` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22391`, seconds `46.50`, LSTM delta `+0.1441`

Top all feature movements:
- `lag_13__CT_place_PIPE`: contribution `+0.019605`
- `lag_03__CT_place_PIPE`: contribution `+0.016806`
- `lag_02__T_flash_duration_sum`: contribution `+0.007235`
- `lag_02__T5__flash_duration`: contribution `+0.007150`
- `lag_02__T_flashed_players`: contribution `+0.007134`

Top utility-only movements:
- `lag_02__T_flash_duration_sum`: contribution `+0.007235`
- `lag_02__T5__flash_duration`: contribution `+0.007150`
- `lag_02__T4__flash_duration`: contribution `+0.006582`

### tick `23159`, seconds `58.50`, LSTM delta `+0.1199`

Top all feature movements:
- `lag_02__CT_place_TSTAIRS`: contribution `+0.029099`
- `lag_09__CT_place_STORAGEROOM`: contribution `+0.016776`
- `lag_04__CT_place_STORAGEROOM`: contribution `+0.014573`
- `lag_09__CT_place_BACKOFA`: contribution `+0.008325`
- `lag_13__CT_place_BACKOFA`: contribution `-0.007149`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.001925`

### tick `26199`, seconds `106.00`, LSTM delta `-0.1127`

Top all feature movements:
- `lag_15__CT2__flash_duration`: contribution `-0.013968`
- `lag_00__T_duck_amount_mean`: contribution `-0.007226`
- `lag_00__kill_diff_last_3s`: contribution `-0.004002`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.003699`
- `lag_00__damage_diff_last_5s`: contribution `-0.003398`

Top utility-only movements:
- `lag_15__CT2__flash_duration`: contribution `-0.013968`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.003699`
- `lag_15__CT_flash_duration_sum`: contribution `-0.002958`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.002736`

### tick `25399`, seconds `93.50`, LSTM delta `+0.1040`

Top all feature movements:
- `lag_05__CT_place_BACKOFA`: contribution `+0.012965`
- `lag_02__CT2__flash_duration`: contribution `+0.010963`
- `lag_03__T_place_UPPERPARK`: contribution `+0.008572`
- `lag_13__CT_place_BACKOFA`: contribution `+0.007149`
- `lag_02__CT_flash_duration_sum`: contribution `+0.005287`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `+0.010963`
- `lag_02__CT_flash_duration_sum`: contribution `+0.005287`
- `lag_02__T2__flash_duration`: contribution `+0.004153`
- `lag_02__T_flash_duration_sum`: contribution `+0.003360`
- `lag_02__CT4__flash_duration`: contribution `+0.003001`

### tick `25943`, seconds `102.00`, LSTM delta `+0.0966`

Top all feature movements:
- `lag_07__CT2__flash_duration`: contribution `+0.007251`
- `lag_09__CT_place_LOWERPARK`: contribution `+0.005802`
- `lag_00__T_place_LOWERPARK`: contribution `+0.004911`
- `lag_00__CT3__is_walking`: contribution `+0.004798`
- `lag_00__CT_kills_last_3s`: contribution `+0.004781`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `+0.007251`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.002937`
- `lag_13__CT4__flash_duration`: contribution `+0.002366`
- `lag_06__CT2__molly`: contribution `+0.001561`
