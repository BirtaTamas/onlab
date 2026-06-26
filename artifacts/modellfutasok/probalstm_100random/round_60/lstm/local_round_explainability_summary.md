# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `13`

## Largest probability jumps

- tick `97668`, seconds `18.00`, LSTM `0.4174`, delta `+0.3114`
- tick `98020`, seconds `23.50`, LSTM `0.5617`, delta `+0.1857`
- tick `97572`, seconds `16.50`, LSTM `0.1481`, delta `-0.1625`
- tick `98436`, seconds `30.00`, LSTM `0.0355`, delta `-0.1600`
- tick `97988`, seconds `23.00`, LSTM `0.3760`, delta `-0.1533`
- tick `98340`, seconds `28.50`, LSTM `0.1895`, delta `-0.1159`
- tick `98052`, seconds `24.00`, LSTM `0.4481`, delta `-0.1136`
- tick `97476`, seconds `15.00`, LSTM `0.3763`, delta `-0.0997`
- tick `98116`, seconds `25.00`, LSTM `0.3412`, delta `-0.0676`
- tick `97540`, seconds `16.00`, LSTM `0.3106`, delta `-0.0414`

## Top 15 local ridge features

- `lag_10__T3__flash_duration`: coefficient `0.003336`, |coef| `0.003336`
- `lag_00__T3__flash_duration`: coefficient `-0.003126`, |coef| `0.003126`
- `lag_00__kill_diff_last_3s`: coefficient `0.002544`, |coef| `0.002544`
- `lag_10__T1__flash_duration`: coefficient `0.002501`, |coef| `0.002501`
- `lag_10__T_flash_duration_sum`: coefficient `0.002435`, |coef| `0.002435`
- `lag_03__CT4__flash_duration`: coefficient `-0.002383`, |coef| `0.002383`
- `lag_00__damage_diff_last_5s`: coefficient `0.002278`, |coef| `0.002278`
- `lag_07__CT_flash_duration_sum`: coefficient `0.002044`, |coef| `0.002044`
- `lag_07__CT3__flash_duration`: coefficient `0.002006`, |coef| `0.002006`
- `lag_10__T_flashed_players`: coefficient `0.001939`, |coef| `0.001939`
- `lag_00__T_kills_last_3s`: coefficient `-0.001900`, |coef| `0.001900`
- `lag_07__T3__flash_duration`: coefficient `-0.001872`, |coef| `0.001872`
- `lag_15__T3__duck_amount`: coefficient `0.001779`, |coef| `0.001779`
- `lag_01__CT1__is_walking`: coefficient `0.001685`, |coef| `0.001685`
- `lag_01__CT3__flash_duration`: coefficient `-0.001645`, |coef| `0.001645`

## Top 10 utility ridge features

- `lag_10__T3__flash_duration`: coefficient `0.003336` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.003126` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.002501` (raises CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `0.002435` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.002383` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.002044` (raises CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `0.002006` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.001872` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.001645` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001507` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002544` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002278` (raises CT win probability)
- `lag_10__T_flashed_players`: coefficient `0.001939` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001900` (lowers CT win probability)
- `lag_15__T3__duck_amount`: coefficient `0.001779` (raises CT win probability)
- `lag_01__CT1__is_walking`: coefficient `0.001685` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001643` (raises CT win probability)
- `lag_07__T5__duck_amount`: coefficient `0.001640` (raises CT win probability)
- `lag_01__CT_place_RUINS`: coefficient `-0.001368` (lowers CT win probability)
- `lag_11__CT2__is_walking`: coefficient `-0.001353` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `97668`, seconds `18.00`, LSTM delta `+0.3114`

Top all feature movements:
- `lag_10__T3__flash_duration`: contribution `+0.026844`
- `lag_00__T3__flash_duration`: contribution `+0.025020`
- `lag_03__CT4__flash_duration`: contribution `+0.016550`
- `lag_07__CT_flash_duration_sum`: contribution `+0.014922`
- `lag_07__CT3__flash_duration`: contribution `+0.012512`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `+0.026844`
- `lag_00__T3__flash_duration`: contribution `+0.025020`
- `lag_03__CT4__flash_duration`: contribution `+0.016550`
- `lag_07__CT_flash_duration_sum`: contribution `+0.014922`
- `lag_07__CT3__flash_duration`: contribution `+0.012512`

### tick `98020`, seconds `23.50`, LSTM delta `+0.1857`

Top all feature movements:
- `lag_10__T1__flash_duration`: contribution `+0.017795`
- `lag_01__CT3__flash_duration`: contribution `+0.011702`
- `lag_11__T3__flash_duration`: contribution `+0.011280`
- `lag_10__T3__flash_duration`: contribution `+0.009997`
- `lag_10__T_flash_duration_sum`: contribution `+0.007210`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `+0.017795`
- `lag_01__CT3__flash_duration`: contribution `+0.011702`
- `lag_11__T3__flash_duration`: contribution `+0.011280`
- `lag_10__T3__flash_duration`: contribution `+0.009997`
- `lag_10__T_flash_duration_sum`: contribution `+0.007210`

### tick `97572`, seconds `16.50`, LSTM delta `-0.1625`

Top all feature movements:
- `lag_07__T3__flash_duration`: contribution `-0.015062`
- `lag_04__CT_flash_duration_sum`: contribution `-0.008537`
- `lag_00__CT4__flash_duration`: contribution `-0.008375`
- `lag_00__kill_diff_last_3s`: contribution `-0.006123`
- `lag_00__T_kills_last_3s`: contribution `-0.006019`

Top utility-only movements:
- `lag_07__T3__flash_duration`: contribution `-0.015062`
- `lag_04__CT_flash_duration_sum`: contribution `-0.008537`
- `lag_00__CT4__flash_duration`: contribution `-0.008375`
- `lag_04__CT4__flash_duration`: contribution `-0.005691`
- `lag_04__CT5__flash_duration`: contribution `-0.004705`

### tick `98436`, seconds `30.00`, LSTM delta `-0.1600`

Top all feature movements:
- `lag_10__T1__flash_duration`: contribution `-0.017795`
- `lag_14__CT3__flash_duration`: contribution `-0.009337`
- `lag_10__T_flash_duration_sum`: contribution `-0.007210`
- `lag_12__CT5__flash_duration`: contribution `-0.006905`
- `lag_00__kill_diff_last_3s`: contribution `-0.006123`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `-0.017795`
- `lag_14__CT3__flash_duration`: contribution `-0.009337`
- `lag_10__T_flash_duration_sum`: contribution `-0.007210`
- `lag_12__CT5__flash_duration`: contribution `-0.006905`
- `lag_12__CT_flash_duration_sum`: contribution `-0.002293`

### tick `97988`, seconds `23.00`, LSTM delta `-0.1533`

Top all feature movements:
- `lag_10__T3__flash_duration`: contribution `-0.026704`
- `lag_10__T_flash_duration_sum`: contribution `-0.008121`
- `lag_00__damage_diff_last_5s`: contribution `-0.007092`
- `lag_00__kill_diff_last_3s`: contribution `-0.006123`
- `lag_00__T_kills_last_3s`: contribution `-0.006019`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.026704`
- `lag_10__T_flash_duration_sum`: contribution `-0.008121`
- `lag_13__CT4__flash_duration`: contribution `-0.005257`
- `lag_00__CT3__flash_duration`: contribution `-0.004666`
- `lag_00__CT_flash_duration_sum`: contribution `-0.003365`
