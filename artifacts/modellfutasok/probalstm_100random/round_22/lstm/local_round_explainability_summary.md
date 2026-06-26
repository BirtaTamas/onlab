# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `9`

## Largest probability jumps

- tick `72841`, seconds `110.00`, LSTM `0.5578`, delta `+0.3804`
- tick `71753`, seconds `93.00`, LSTM `0.2770`, delta `-0.3487`
- tick `71561`, seconds `90.00`, LSTM `0.4110`, delta `+0.3439`
- tick `71881`, seconds `95.00`, LSTM `0.0429`, delta `-0.2264`
- tick `73225`, seconds `116.00`, LSTM `0.8574`, delta `+0.1857`
- tick `66185`, seconds `6.00`, LSTM `0.1448`, delta `-0.0865`
- tick `65833`, seconds `0.50`, LSTM `0.1361`, delta `-0.0774`
- tick `72873`, seconds `110.50`, LSTM `0.4812`, delta `-0.0766`
- tick `71625`, seconds `91.00`, LSTM `0.5383`, delta `+0.0738`
- tick `68969`, seconds `49.50`, LSTM `0.0288`, delta `-0.0697`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005950`, |coef| `0.005950`
- `lag_00__CT_defusing_count`: coefficient `0.005718`, |coef| `0.005718`
- `lag_00__CT_kills_last_3s`: coefficient `0.005701`, |coef| `0.005701`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005306`, |coef| `0.005306`
- `lag_07__T1__flash_duration`: coefficient `0.004587`, |coef| `0.004587`
- `lag_00__damage_diff_last_5s`: coefficient `0.003982`, |coef| `0.003982`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003697`, |coef| `0.003697`
- `lag_12__T_flash_alpha_mean`: coefficient `-0.003580`, |coef| `0.003580`
- `lag_05__CT_kills_last_3s`: coefficient `0.003288`, |coef| `0.003288`
- `lag_00__CT_velocity_mean`: coefficient `-0.003076`, |coef| `0.003076`
- `lag_15__T_bomb_zone_count`: coefficient `-0.003075`, |coef| `0.003075`
- `lag_15__T5__duck_amount`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_00__CT_damage_last_5s`: coefficient `0.003040`, |coef| `0.003040`
- `lag_06__T4__flash_duration`: coefficient `0.002752`, |coef| `0.002752`
- `lag_05__kill_diff_last_3s`: coefficient `0.002696`, |coef| `0.002696`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005306` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.004587` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.003580` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.002752` (raises CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.002598` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.002432` (raises CT win probability)
- `lag_05__T5__flash`: coefficient `-0.002345` (lowers CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `0.002318` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.002307` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.002093` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005950` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.005718` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005701` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003982` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003697` (lowers CT win probability)
- `lag_05__CT_kills_last_3s`: coefficient `0.003288` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.003076` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `-0.003075` (lowers CT win probability)
- `lag_15__T5__duck_amount`: coefficient `-0.003066` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003040` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72841`, seconds `110.00`, LSTM delta `+0.3804`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.032191`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.018043`
- `lag_15__T_bomb_zone_count`: contribution `+0.017902`
- `lag_00__CT_kills_last_3s`: contribution `+0.016460`
- `lag_07__CT_duck_amount_mean`: contribution `+0.015754`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.032191`
- `lag_05__T5__flash`: contribution `+0.006656`

### tick `71753`, seconds `93.00`, LSTM delta `-0.3487`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.028642`
- `lag_07__T1__flash_duration`: contribution `-0.019537`
- `lag_00__CT_kills_last_3s`: contribution `-0.016460`
- `lag_06__T4__flash_duration`: contribution `-0.011279`
- `lag_00__T5__is_scoped`: contribution `-0.011199`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `-0.019537`
- `lag_06__T4__flash_duration`: contribution `-0.011279`
- `lag_13__T4__flash_duration`: contribution `-0.007899`
- `lag_07__T_flash_duration_sum`: contribution `-0.004604`

### tick `71561`, seconds `90.00`, LSTM delta `+0.3439`

Top all feature movements:
- `lag_07__T1__flash_duration`: contribution `+0.019537`
- `lag_00__CT_kills_last_3s`: contribution `+0.016460`
- `lag_00__kill_diff_last_3s`: contribution `+0.014321`
- `lag_04__T5__is_scoped`: contribution `+0.010496`
- `lag_07__T4__flash_duration`: contribution `+0.009967`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `+0.019537`
- `lag_07__T4__flash_duration`: contribution `+0.009967`
- `lag_01__T1__flash_duration`: contribution `+0.009827`
- `lag_07__T_flash_duration_sum`: contribution `+0.008934`
- `lag_00__T4__flash_duration`: contribution `+0.008577`

### tick `71881`, seconds `95.00`, LSTM delta `-0.2264`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.017968`
- `lag_00__kill_diff_last_3s`: contribution `-0.014321`
- `lag_04__T5__is_scoped`: contribution `-0.010496`
- `lag_05__CT_kills_last_3s`: contribution `-0.009493`
- `lag_00__CT_damage_last_5s`: contribution `-0.006627`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.006103`
- `lag_11__T1__flash_duration`: contribution `-0.004302`

### tick `73225`, seconds `116.00`, LSTM delta `+0.1857`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.055431`
- `lag_03__CT_flashes_last_5s`: contribution `+0.025490`
- `lag_12__T_flash_alpha_mean`: contribution `+0.021721`
- `lag_00__CT_velocity_mean`: contribution `+0.011165`
- `lag_12__T_place_SIDEENTRANCE`: contribution `+0.010450`

Top utility-only movements:
- `lag_03__CT_flashes_last_5s`: contribution `+0.025490`
- `lag_12__T_flash_alpha_mean`: contribution `+0.021721`
