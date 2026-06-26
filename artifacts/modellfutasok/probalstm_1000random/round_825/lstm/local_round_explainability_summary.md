# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `51279`, seconds `97.50`, LSTM `0.7045`, delta `+0.4346`
- tick `51311`, seconds `98.00`, LSTM `0.4082`, delta `-0.2962`
- tick `50383`, seconds `83.50`, LSTM `0.1438`, delta `-0.2936`
- tick `52271`, seconds `113.00`, LSTM `0.7718`, delta `+0.2228`
- tick `51151`, seconds `95.50`, LSTM `0.2539`, delta `-0.1900`
- tick `50767`, seconds `89.50`, LSTM `0.3603`, delta `+0.1671`
- tick `51055`, seconds `94.00`, LSTM `0.4638`, delta `+0.1490`
- tick `51343`, seconds `98.50`, LSTM `0.5189`, delta `+0.1106`
- tick `51567`, seconds `102.00`, LSTM `0.6258`, delta `+0.1062`
- tick `51599`, seconds `102.50`, LSTM `0.5216`, delta `-0.1042`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004659`, |coef| `0.004659`
- `lag_15__T_place_ARAMP`: coefficient `0.003098`, |coef| `0.003098`
- `lag_00__CT_kills_last_3s`: coefficient `0.003063`, |coef| `0.003063`
- `lag_00__T_place_UNDERA`: coefficient `-0.002781`, |coef| `0.002781`
- `lag_00__T_kills_last_3s`: coefficient `-0.002772`, |coef| `0.002772`
- `lag_15__CT_shots_fired_sum`: coefficient `-0.002754`, |coef| `0.002754`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002553`, |coef| `0.002553`
- `lag_09__T2__flash_duration`: coefficient `0.002498`, |coef| `0.002498`
- `lag_04__T2__duck_amount`: coefficient `0.002475`, |coef| `0.002475`
- `lag_01__CT_place_HOLE`: coefficient `-0.002425`, |coef| `0.002425`
- `lag_07__T3__flash_duration`: coefficient `0.002395`, |coef| `0.002395`
- `lag_00__CT_place_LONGDOORS`: coefficient `0.002394`, |coef| `0.002394`
- `lag_11__T_place_ARAMP`: coefficient `-0.002330`, |coef| `0.002330`
- `lag_02__T_place_EXTENDEDA`: coefficient `-0.002264`, |coef| `0.002264`
- `lag_00__T_place_LONGA`: coefficient `-0.002263`, |coef| `0.002263`

## Top 10 utility ridge features

- `lag_09__T2__flash_duration`: coefficient `0.002498` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.002395` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.002152` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001969` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001749` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.001663` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.001393` (lowers CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.001387` (lowers CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.001361` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001347` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004659` (raises CT win probability)
- `lag_15__T_place_ARAMP`: coefficient `0.003098` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003063` (raises CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.002781` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002772` (lowers CT win probability)
- `lag_15__CT_shots_fired_sum`: coefficient `-0.002754` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002553` (raises CT win probability)
- `lag_04__T2__duck_amount`: coefficient `0.002475` (raises CT win probability)
- `lag_01__CT_place_HOLE`: coefficient `-0.002425` (lowers CT win probability)
- `lag_00__CT_place_LONGDOORS`: coefficient `0.002394` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `51279`, seconds `97.50`, LSTM delta `+0.4346`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.027078`
- `lag_15__CT_shots_fired_sum`: contribution `+0.024871`
- `lag_03__CT_place_HOLE`: contribution `+0.023720`
- `lag_09__T2__flash_duration`: contribution `+0.017468`
- `lag_15__CT1__shots_fired`: contribution `+0.015430`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.017468`
- `lag_09__CT1__flash_duration`: contribution `+0.012228`
- `lag_07__T3__flash_duration`: contribution `+0.011658`

### tick `51311`, seconds `98.00`, LSTM delta `-0.2962`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `-0.022094`
- `lag_00__CT_shots_fired_sum`: contribution `-0.012416`
- `lag_00__kill_diff_last_3s`: contribution `-0.011214`
- `lag_04__CT_place_HOLE`: contribution `-0.011117`
- `lag_04__T2__duck_amount`: contribution `-0.009463`

Top utility-only movements:
- `lag_10__CT1__flash_duration`: contribution `-0.007458`
- `lag_10__T2__flash_duration`: contribution `-0.005522`
- `lag_13__T_A_site_active_infernos`: contribution `-0.004129`
- `lag_14__CT5__flash_duration`: contribution `-0.004008`

### tick `50383`, seconds `83.50`, LSTM delta `-0.2936`

Top all feature movements:
- `lag_02__T_place_EXTENDEDA`: contribution `-0.011225`
- `lag_00__kill_diff_last_3s`: contribution `-0.011214`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.010482`
- `lag_05__CT_place_BDOORS`: contribution `-0.010065`
- `lag_15__CT3__is_scoped`: contribution `-0.008882`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.005861`

### tick `52271`, seconds `113.00`, LSTM delta `+0.2228`

Top all feature movements:
- `lag_15__T_place_ARAMP`: contribution `+0.028034`
- `lag_11__T_place_ARAMP`: contribution `+0.021085`
- `lag_00__T_bomb_zone_count`: contribution `+0.011651`
- `lag_10__T_bomb_zone_count`: contribution `+0.011498`
- `lag_00__kill_diff_last_3s`: contribution `+0.011214`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.003668`

### tick `51151`, seconds `95.50`, LSTM delta `-0.1900`

Top all feature movements:
- `lag_12__T_place_UNDERA`: contribution `-0.028659`
- `lag_11__CT_shots_fired_sum`: contribution `-0.012548`
- `lag_00__kill_diff_last_3s`: contribution `-0.011214`
- `lag_00__T_kills_last_3s`: contribution `-0.008781`
- `lag_05__CT1__flash_duration`: contribution `-0.007913`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.007913`
- `lag_05__T2__flash_duration`: contribution `-0.006782`
