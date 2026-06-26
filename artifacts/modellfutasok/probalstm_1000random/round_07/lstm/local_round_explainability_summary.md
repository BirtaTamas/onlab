# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `144473`, seconds `104.00`, LSTM `0.8264`, delta `+0.2772`
- tick `144409`, seconds `103.00`, LSTM `0.5328`, delta `+0.2602`
- tick `144153`, seconds `99.00`, LSTM `0.2487`, delta `-0.2118`
- tick `143737`, seconds `92.50`, LSTM `0.5291`, delta `-0.1955`
- tick `143673`, seconds `91.50`, LSTM `0.6796`, delta `+0.0863`
- tick `144537`, seconds `105.00`, LSTM `0.9320`, delta `+0.0792`
- tick `143641`, seconds `91.00`, LSTM `0.5933`, delta `+0.0696`
- tick `144121`, seconds `98.50`, LSTM `0.4605`, delta `+0.0579`
- tick `144185`, seconds `99.50`, LSTM `0.1912`, delta `-0.0576`
- tick `144089`, seconds `98.00`, LSTM `0.4026`, delta `-0.0474`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002938`, |coef| `0.002938`
- `lag_00__damage_diff_last_5s`: coefficient `0.002242`, |coef| `0.002242`
- `lag_00__CT_kills_last_3s`: coefficient `0.002160`, |coef| `0.002160`
- `lag_06__CT_place_LONGDOG`: coefficient `-0.002148`, |coef| `0.002148`
- `lag_13__T_B_site_active_infernos`: coefficient `-0.002027`, |coef| `0.002027`
- `lag_15__CT5__flash_duration`: coefficient `-0.001926`, |coef| `0.001926`
- `lag_08__CT_place_LONGDOG`: coefficient `-0.001857`, |coef| `0.001857`
- `lag_14__T_bomb_zone_count`: coefficient `0.001781`, |coef| `0.001781`
- `lag_12__CT5__flash_duration`: coefficient `-0.001749`, |coef| `0.001749`
- `lag_07__CT_place_ENTRANCE`: coefficient `0.001713`, |coef| `0.001713`
- `lag_09__CT_place_ENTRANCE`: coefficient `0.001706`, |coef| `0.001706`
- `lag_02__kill_diff_last_3s`: coefficient `0.001672`, |coef| `0.001672`
- `lag_06__CT_place_BACKOFB`: coefficient `0.001662`, |coef| `0.001662`
- `lag_14__CT5__flash_duration`: coefficient `-0.001629`, |coef| `0.001629`
- `lag_13__T_active_infernos`: coefficient `-0.001516`, |coef| `0.001516`

## Top 10 utility ridge features

- `lag_13__T_B_site_active_infernos`: coefficient `-0.002027` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.001926` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.001749` (lowers CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.001629` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.001516` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.001440` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.001364` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001364` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.001285` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.001260` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002938` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002242` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002160` (raises CT win probability)
- `lag_06__CT_place_LONGDOG`: coefficient `-0.002148` (lowers CT win probability)
- `lag_08__CT_place_LONGDOG`: coefficient `-0.001857` (lowers CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `0.001781` (raises CT win probability)
- `lag_07__CT_place_ENTRANCE`: coefficient `0.001713` (raises CT win probability)
- `lag_09__CT_place_ENTRANCE`: coefficient `0.001706` (raises CT win probability)
- `lag_02__kill_diff_last_3s`: coefficient `0.001672` (raises CT win probability)
- `lag_06__CT_place_BACKOFB`: coefficient `0.001662` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `144473`, seconds `104.00`, LSTM delta `+0.2772`

Top all feature movements:
- `lag_09__CT_place_ENTRANCE`: contribution `+0.015135`
- `lag_08__CT_place_LONGDOG`: contribution `+0.012110`
- `lag_14__T_bomb_zone_count`: contribution `+0.010368`
- `lag_14__CT5__flash_duration`: contribution `+0.008610`
- `lag_00__T_bomb_zone_count`: contribution `+0.008358`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `+0.008610`
- `lag_13__T_B_site_active_infernos`: contribution `+0.005730`
- `lag_08__T_B_site_active_infernos`: contribution `+0.003337`
- `lag_13__T_active_infernos`: contribution `+0.003157`
- `lag_15__T_B_site_active_infernos`: contribution `+0.002758`

### tick `144409`, seconds `103.00`, LSTM delta `+0.2602`

Top all feature movements:
- `lag_07__CT_place_ENTRANCE`: contribution `+0.015202`
- `lag_06__CT_place_LONGDOG`: contribution `+0.014013`
- `lag_06__CT_place_BACKOFB`: contribution `+0.009491`
- `lag_12__CT5__flash_duration`: contribution `+0.009245`
- `lag_12__T_bomb_zone_count`: contribution `+0.008790`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.009245`
- `lag_13__T_B_site_active_infernos`: contribution `+0.005730`
- `lag_06__T_B_site_active_infernos`: contribution `+0.003857`
- `lag_13__T_active_infernos`: contribution `+0.003157`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002844`

### tick `144153`, seconds `99.00`, LSTM delta `-0.2118`

Top all feature movements:
- `lag_15__CT5__flash_duration`: contribution `-0.011684`
- `lag_15__T1__flash_duration`: contribution `-0.007909`
- `lag_04__CT5__flash_duration`: contribution `-0.007610`
- `lag_00__kill_diff_last_3s`: contribution `-0.007073`
- `lag_13__CT2__flash_duration`: contribution `-0.006939`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.011684`
- `lag_15__T1__flash_duration`: contribution `-0.007909`
- `lag_04__CT5__flash_duration`: contribution `-0.007610`
- `lag_13__CT2__flash_duration`: contribution `-0.006939`
- `lag_12__T_B_site_active_infernos`: contribution `-0.003634`

### tick `143737`, seconds `92.50`, LSTM delta `-0.1955`

Top all feature movements:
- `lag_02__T1__flash_duration`: contribution `-0.010165`
- `lag_05__CT_place_LONGDOG`: contribution `-0.009513`
- `lag_00__kill_diff_last_3s`: contribution `-0.007073`
- `lag_02__CT5__flash_duration`: contribution `-0.006950`
- `lag_06__CT2__flash_duration`: contribution `-0.005909`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.010165`
- `lag_02__CT5__flash_duration`: contribution `-0.006950`
- `lag_06__CT2__flash_duration`: contribution `-0.005909`
- `lag_00__CT2__flash_duration`: contribution `-0.005714`
- `lag_06__T_B_site_active_infernos`: contribution `-0.003857`

### tick `143673`, seconds `91.50`, LSTM delta `+0.0863`

Top all feature movements:
- `lag_00__T1__flash_duration`: contribution `+0.006578`
- `lag_01__kill_diff_last_3s`: contribution `+0.003452`
- `lag_00__CT5__flash_duration`: contribution `+0.003433`
- `lag_01__CT_kills_last_3s`: contribution `+0.003198`
- `lag_04__CT2__flash_duration`: contribution `+0.002653`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.006578`
- `lag_00__CT5__flash_duration`: contribution `+0.003433`
- `lag_04__CT2__flash_duration`: contribution `+0.002653`
- `lag_00__CT_flash_duration_sum`: contribution `+0.002443`
- `lag_04__CT_flash_duration_sum`: contribution `+0.002178`
