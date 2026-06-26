# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `18`

## Largest probability jumps

- tick `138716`, seconds `56.00`, LSTM `0.8168`, delta `+0.2445`
- tick `140604`, seconds `85.50`, LSTM `0.8880`, delta `+0.2125`
- tick `137212`, seconds `32.50`, LSTM `0.6021`, delta `-0.1288`
- tick `139228`, seconds `64.00`, LSTM `0.8222`, delta `-0.1236`
- tick `138780`, seconds `57.00`, LSTM `0.9290`, delta `+0.0987`
- tick `135772`, seconds `10.00`, LSTM `0.8001`, delta `+0.0825`
- tick `137916`, seconds `43.50`, LSTM `0.6795`, delta `+0.0750`
- tick `139516`, seconds `68.50`, LSTM `0.6338`, delta `-0.0708`
- tick `139676`, seconds `71.00`, LSTM `0.6585`, delta `-0.0652`
- tick `140316`, seconds `81.00`, LSTM `0.6702`, delta `-0.0630`

## Top 15 local ridge features

- `lag_02__T4__is_scoped`: coefficient `-0.003335`, |coef| `0.003335`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003322`, |coef| `0.003322`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003195`, |coef| `0.003195`
- `lag_00__kill_diff_last_3s`: coefficient `0.003002`, |coef| `0.003002`
- `lag_00__CT_kills_last_3s`: coefficient `0.002519`, |coef| `0.002519`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002513`, |coef| `0.002513`
- `lag_00__CT3__duck_amount`: coefficient `0.002353`, |coef| `0.002353`
- `lag_06__T_flashes_last_5s`: coefficient `-0.002236`, |coef| `0.002236`
- `lag_00__T4__flash_duration`: coefficient `0.002216`, |coef| `0.002216`
- `lag_05__T1__flash_duration`: coefficient `0.002158`, |coef| `0.002158`
- `lag_02__T_scoped_count`: coefficient `-0.002080`, |coef| `0.002080`
- `lag_11__T4__is_scoped`: coefficient `0.001973`, |coef| `0.001973`
- `lag_14__T_bomb_zone_count`: coefficient `-0.001969`, |coef| `0.001969`
- `lag_08__CT_place_LIBRARY`: coefficient `0.001882`, |coef| `0.001882`
- `lag_00__damage_diff_last_5s`: coefficient `0.001865`, |coef| `0.001865`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003195` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.002236` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.002216` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.002158` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001715` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.001698` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.001582` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001139` (lowers CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.001000` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000951` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T4__is_scoped`: coefficient `-0.003335` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003322` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003002` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002519` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002513` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.002353` (raises CT win probability)
- `lag_02__T_scoped_count`: coefficient `-0.002080` (lowers CT win probability)
- `lag_11__T4__is_scoped`: coefficient `0.001973` (raises CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.001969` (lowers CT win probability)
- `lag_08__CT_place_LIBRARY`: coefficient `0.001882` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `138716`, seconds `56.00`, LSTM delta `+0.2445`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `+0.017004`
- `lag_05__T1__flash_duration`: contribution `+0.015410`
- `lag_08__CT_place_LIBRARY`: contribution `+0.012065`
- `lag_04__CT_place_LIBRARY`: contribution `+0.009601`
- `lag_05__T2__flash_duration`: contribution `+0.009432`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.017004`
- `lag_05__T1__flash_duration`: contribution `+0.015410`
- `lag_05__T2__flash_duration`: contribution `+0.009432`
- `lag_05__CT5__flash_duration`: contribution `+0.009393`
- `lag_05__T_flash_duration_sum`: contribution `+0.008294`

### tick `140604`, seconds `85.50`, LSTM delta `+0.2125`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019383`
- `lag_02__T4__is_scoped`: contribution `+0.015489`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012220`
- `lag_14__T_bomb_zone_count`: contribution `+0.011463`
- `lag_11__T4__is_scoped`: contribution `+0.009165`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.019383`

### tick `137212`, seconds `32.50`, LSTM delta `-0.1288`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.020259`
- `lag_00__kill_diff_last_3s`: contribution `-0.007227`
- `lag_11__T_flashed_players`: contribution `-0.005616`
- `lag_00__damage_diff_last_5s`: contribution `-0.004208`
- `lag_01__T2__duck_amount`: contribution `-0.004130`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.020259`
- `lag_00__CT1__utility_total`: contribution `-0.003398`
- `lag_00__CT1__flash`: contribution `-0.003350`

### tick `139228`, seconds `64.00`, LSTM delta `-0.1236`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.007227`
- `lag_02__T4__flash_duration`: contribution `-0.006839`
- `lag_13__CT_shots_fired_sum`: contribution `-0.006668`
- `lag_05__T_bomb_zone_count`: contribution `-0.005797`
- `lag_09__T_duck_amount_mean`: contribution `-0.005434`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.006839`
- `lag_04__CT5__flash_duration`: contribution `-0.003973`

### tick `138780`, seconds `57.00`, LSTM delta `+0.0987`

Top all feature movements:
- `lag_00__CT3__duck_amount`: contribution `-0.008756`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008729`
- `lag_00__CT_kills_last_3s`: contribution `+0.007272`
- `lag_00__kill_diff_last_3s`: contribution `+0.007227`
- `lag_02__T4__flash_duration`: contribution `+0.006839`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.006839`
- `lag_05__T2__flash_duration`: contribution `-0.003268`
