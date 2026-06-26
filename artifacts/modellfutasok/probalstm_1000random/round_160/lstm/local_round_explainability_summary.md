# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `113245`, seconds `59.50`, LSTM `0.4455`, delta `-0.3866`
- tick `114429`, seconds `78.00`, LSTM `0.5856`, delta `+0.2132`
- tick `112957`, seconds `55.00`, LSTM `0.7357`, delta `+0.1866`
- tick `114493`, seconds `79.00`, LSTM `0.6825`, delta `+0.1406`
- tick `111421`, seconds `31.00`, LSTM `0.5095`, delta `-0.1222`
- tick `114269`, seconds `75.50`, LSTM `0.3277`, delta `+0.0834`
- tick `113629`, seconds `65.50`, LSTM `0.3220`, delta `-0.0748`
- tick `112989`, seconds `55.50`, LSTM `0.8051`, delta `+0.0695`
- tick `113277`, seconds `60.00`, LSTM `0.3852`, delta `-0.0603`
- tick `114525`, seconds `79.50`, LSTM `0.7380`, delta `+0.0555`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004965`, |coef| `0.004965`
- `lag_00__CT_kills_last_3s`: coefficient `0.003908`, |coef| `0.003908`
- `lag_05__T_bomb_zone_count`: coefficient `-0.003594`, |coef| `0.003594`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002856`, |coef| `0.002856`
- `lag_00__T_place_TSIDELOWER`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_14__CT1__duck_amount`: coefficient `0.002590`, |coef| `0.002590`
- `lag_03__kill_diff_last_3s`: coefficient `0.002503`, |coef| `0.002503`
- `lag_00__CT_place_RAMP`: coefficient `0.002498`, |coef| `0.002498`
- `lag_03__T_place_SIDEENTRANCE`: coefficient `0.002491`, |coef| `0.002491`
- `lag_03__CT_kills_last_3s`: coefficient `0.002463`, |coef| `0.002463`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002433`, |coef| `0.002433`
- `lag_00__T_kills_last_3s`: coefficient `-0.002247`, |coef| `0.002247`
- `lag_05__T3__flash_duration`: coefficient `-0.002188`, |coef| `0.002188`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002167`, |coef| `0.002167`
- `lag_09__T_place_TSIDELOWER`: coefficient `0.002166`, |coef| `0.002166`

## Top 10 utility ridge features

- `lag_05__T3__flash_duration`: coefficient `-0.002188` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `0.001343` (raises CT win probability)
- `lag_13__T5__molly`: coefficient `-0.001327` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.001214` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.001184` (lowers CT win probability)
- `lag_09__T_active_infernos`: coefficient `0.001002` (raises CT win probability)
- `lag_08__T5__molly`: coefficient `-0.000981` (lowers CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000980` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `-0.000962` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000928` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004965` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003908` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.003594` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002856` (lowers CT win probability)
- `lag_00__T_place_TSIDELOWER`: coefficient `-0.002643` (lowers CT win probability)
- `lag_14__CT1__duck_amount`: coefficient `0.002590` (raises CT win probability)
- `lag_03__kill_diff_last_3s`: coefficient `0.002503` (raises CT win probability)
- `lag_00__CT_place_RAMP`: coefficient `0.002498` (raises CT win probability)
- `lag_03__T_place_SIDEENTRANCE`: coefficient `0.002491` (raises CT win probability)
- `lag_03__CT_kills_last_3s`: coefficient `0.002463` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `113245`, seconds `59.50`, LSTM delta `-0.3866`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.023902`
- `lag_00__T_shots_fired_sum`: contribution `-0.021891`
- `lag_09__T_place_TSIDELOWER`: contribution `-0.016237`
- `lag_05__T3__flash_duration`: contribution `-0.014977`
- `lag_00__CT_place_RAMP`: contribution `-0.014924`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.014977`

### tick `114429`, seconds `78.00`, LSTM delta `+0.2132`

Top all feature movements:
- `lag_05__T_bomb_zone_count`: contribution `+0.020922`
- `lag_03__T_place_SIDEENTRANCE`: contribution `+0.012156`
- `lag_00__kill_diff_last_3s`: contribution `+0.011951`
- `lag_00__CT_kills_last_3s`: contribution `+0.011284`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.010484`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `+0.003796`
- `lag_13__T5__molly`: contribution `+0.002936`

### tick `112957`, seconds `55.00`, LSTM delta `+0.1866`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.023902`
- `lag_00__CT_kills_last_3s`: contribution `+0.022568`
- `lag_00__T_place_TSIDELOWER`: contribution `+0.019809`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007528`
- `lag_08__CT_place_HOUSE`: contribution `+0.006856`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114493`, seconds `79.00`, LSTM delta `+0.1406`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.018068`
- `lag_00__kill_diff_last_3s`: contribution `+0.011951`
- `lag_00__CT_kills_last_3s`: contribution `+0.011284`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.010484`
- `lag_06__T_place_SIDEENTRANCE`: contribution `+0.009474`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111421`, seconds `31.00`, LSTM delta `-0.1222`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011951`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.009786`
- `lag_00__T_shots_fired_sum`: contribution `-0.007297`
- `lag_00__T_kills_last_3s`: contribution `-0.007118`
- `lag_03__T_place_TSIDELOWER`: contribution `-0.006887`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.005890`
- `lag_04__CT5__flash_duration`: contribution `-0.003995`
- `lag_04__CT4__flash_duration`: contribution `-0.003676`
- `lag_04__CT_flash_duration_sum`: contribution `-0.003180`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.002390`
