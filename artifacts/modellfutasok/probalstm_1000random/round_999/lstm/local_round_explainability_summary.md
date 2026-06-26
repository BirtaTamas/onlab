# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `44787`, seconds `62.00`, LSTM `0.7248`, delta `+0.3678`
- tick `44435`, seconds `56.50`, LSTM `0.4358`, delta `-0.3422`
- tick `45235`, seconds `69.00`, LSTM `0.3997`, delta `-0.2627`
- tick `44115`, seconds `51.50`, LSTM `0.5893`, delta `+0.2048`
- tick `43891`, seconds `48.00`, LSTM `0.1054`, delta `-0.2011`
- tick `44083`, seconds `51.00`, LSTM `0.3845`, delta `+0.2009`
- tick `43923`, seconds `48.50`, LSTM `0.2930`, delta `+0.1876`
- tick `43859`, seconds `47.50`, LSTM `0.3065`, delta `-0.1709`
- tick `44723`, seconds `61.00`, LSTM `0.3926`, delta `+0.1665`
- tick `44179`, seconds `52.50`, LSTM `0.6667`, delta `+0.1138`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004081`, |coef| `0.004081`
- `lag_00__T_kills_last_3s`: coefficient `-0.003736`, |coef| `0.003736`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003129`, |coef| `0.003129`
- `lag_11__T_kills_last_3s`: coefficient `0.002730`, |coef| `0.002730`
- `lag_01__T5__duck_amount`: coefficient `-0.002729`, |coef| `0.002729`
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.002605`, |coef| `0.002605`
- `lag_01__T_duck_amount_mean`: coefficient `-0.002533`, |coef| `0.002533`
- `lag_00__CT_place_SHOP`: coefficient `-0.002435`, |coef| `0.002435`
- `lag_15__CT_kills_last_3s`: coefficient `-0.002364`, |coef| `0.002364`
- `lag_01__damage_diff_last_5s`: coefficient `0.002229`, |coef| `0.002229`
- `lag_05__T1__is_scoped`: coefficient `-0.002186`, |coef| `0.002186`
- `lag_03__T1__is_scoped`: coefficient `-0.002142`, |coef| `0.002142`
- `lag_11__CT1__duck_amount`: coefficient `0.002110`, |coef| `0.002110`
- `lag_14__T1__duck_amount`: coefficient `0.002071`, |coef| `0.002071`
- `lag_08__CT_kills_last_3s`: coefficient `0.002048`, |coef| `0.002048`

## Top 10 utility ridge features

- `lag_04__CT_A_site_active_infernos`: coefficient `-0.002605` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.001967` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.001681` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.001520` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `-0.001474` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.001392` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001342` (lowers CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.001190` (raises CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.001011` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000997` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004081` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003736` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003129` (lowers CT win probability)
- `lag_11__T_kills_last_3s`: coefficient `0.002730` (raises CT win probability)
- `lag_01__T5__duck_amount`: coefficient `-0.002729` (lowers CT win probability)
- `lag_01__T_duck_amount_mean`: coefficient `-0.002533` (lowers CT win probability)
- `lag_00__CT_place_SHOP`: coefficient `-0.002435` (lowers CT win probability)
- `lag_15__CT_kills_last_3s`: coefficient `-0.002364` (lowers CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002229` (raises CT win probability)
- `lag_05__T1__is_scoped`: coefficient `-0.002186` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44787`, seconds `62.00`, LSTM delta `+0.3678`

Top all feature movements:
- `lag_03__T1__is_scoped`: contribution `+0.012238`
- `lag_00__kill_diff_last_3s`: contribution `+0.009822`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.009194`
- `lag_11__T_kills_last_3s`: contribution `+0.008650`
- `lag_01__T_duck_amount_mean`: contribution `-0.007367`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `+0.009194`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.006943`

### tick `44435`, seconds `56.50`, LSTM delta `-0.3422`

Top all feature movements:
- `lag_05__T1__is_scoped`: contribution `-0.012487`
- `lag_00__T_kills_last_3s`: contribution `-0.011836`
- `lag_00__T_shots_fired_sum`: contribution `-0.011731`
- `lag_10__T3__flash_duration`: contribution `-0.011479`
- `lag_01__T5__duck_amount`: contribution `-0.010364`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `-0.011479`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.009194`

### tick `45235`, seconds `69.00`, LSTM delta `-0.2627`

Top all feature movements:
- `lag_01__T_duck_amount_mean`: contribution `-0.014734`
- `lag_00__CT_place_SHOP`: contribution `-0.012214`
- `lag_00__T_kills_last_3s`: contribution `-0.011836`
- `lag_00__T_shots_fired_sum`: contribution `-0.011731`
- `lag_01__T5__duck_amount`: contribution `-0.010364`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44115`, seconds `51.50`, LSTM delta `+0.2048`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.011731`
- `lag_06__T2__flash_duration`: contribution `+0.010189`
- `lag_01__T_duck_amount_mean`: contribution `+0.009408`
- `lag_14__T1__duck_amount`: contribution `+0.008108`
- `lag_11__CT1__duck_amount`: contribution `+0.008051`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `+0.010189`
- `lag_08__CT5__flash_duration`: contribution `+0.006984`
- `lag_00__T3__flash_duration`: contribution `+0.006115`
- `lag_13__T3__flash_duration`: contribution `+0.005648`
- `lag_10__CT5__flash_duration`: contribution `+0.004159`

### tick `43891`, seconds `48.00`, LSTM delta `-0.2011`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.011836`
- `lag_00__T_shots_fired_sum`: contribution `-0.011731`
- `lag_06__CT_flashed_players`: contribution `-0.011586`
- `lag_06__T2__flash_duration`: contribution `-0.010189`
- `lag_00__kill_diff_last_3s`: contribution `-0.009822`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.010189`
- `lag_06__T_flash_duration_sum`: contribution `-0.005848`
- `lag_01__CT5__flash_duration`: contribution `-0.005507`
- `lag_06__T3__flash_duration`: contribution `-0.004939`
