# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `71157`, seconds `59.50`, LSTM `0.0356`, delta `-0.0913`
- tick `67381`, seconds `0.50`, LSTM `0.1617`, delta `-0.0715`
- tick `68437`, seconds `17.00`, LSTM `0.1668`, delta `-0.0420`
- tick `67413`, seconds `1.00`, LSTM `0.1267`, delta `-0.0350`
- tick `70965`, seconds `56.50`, LSTM `0.1516`, delta `-0.0297`
- tick `68181`, seconds `13.00`, LSTM `0.2071`, delta `+0.0270`
- tick `68053`, seconds `11.00`, LSTM `0.1623`, delta `-0.0263`
- tick `68565`, seconds `19.00`, LSTM `0.1953`, delta `+0.0259`
- tick `69621`, seconds `35.50`, LSTM `0.1747`, delta `-0.0230`
- tick `67445`, seconds `1.50`, LSTM `0.1038`, delta `-0.0229`

## Top 15 local ridge features

- `lag_09__T3__flash_duration`: coefficient `-0.001025`, |coef| `0.001025`
- `lag_00__CT5__is_scoped`: coefficient `0.001017`, |coef| `0.001017`
- `lag_00__CT_velocity_mean`: coefficient `-0.000776`, |coef| `0.000776`
- `lag_09__CT_place_EXTENDEDA`: coefficient `-0.000728`, |coef| `0.000728`
- `lag_00__CT_place_ARAMP`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000701`, |coef| `0.000701`
- `lag_00__T_velocity_mean`: coefficient `-0.000682`, |coef| `0.000682`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000651`, |coef| `0.000651`
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.000637`, |coef| `0.000637`
- `lag_08__T3__flash_duration`: coefficient `-0.000617`, |coef| `0.000617`
- `lag_00__CT_scoped_count`: coefficient `0.000609`, |coef| `0.000609`
- `lag_01__T5__is_walking`: coefficient `0.000584`, |coef| `0.000584`
- `lag_00__T_flashes_last_5s`: coefficient `-0.000560`, |coef| `0.000560`
- `lag_00__T_place_TUNNELSTAIRS`: coefficient `-0.000557`, |coef| `0.000557`
- `lag_09__T_place_CATWALK`: coefficient `-0.000554`, |coef| `0.000554`

## Top 10 utility ridge features

- `lag_09__T3__flash_duration`: coefficient `-0.001025` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000617` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000560` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000469` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000449` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000440` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.000437` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000425` (raises CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000415` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000407` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT5__is_scoped`: coefficient `0.001017` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000776` (lowers CT win probability)
- `lag_09__CT_place_EXTENDEDA`: coefficient `-0.000728` (lowers CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.000713` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000701` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000682` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000651` (lowers CT win probability)
- `lag_01__T_place_SHORTSTAIRS`: coefficient `-0.000637` (lowers CT win probability)
- `lag_00__CT_scoped_count`: coefficient `0.000609` (raises CT win probability)
- `lag_01__T5__is_walking`: coefficient `0.000584` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `71157`, seconds `59.50`, LSTM delta `-0.0913`

Top all feature movements:
- `lag_09__T3__flash_duration`: contribution `-0.007976`
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.004087`
- `lag_00__CT5__is_scoped`: contribution `-0.003636`
- `lag_13__CT_place_EXTENDEDA`: contribution `-0.002839`
- `lag_01__T_place_SHORTSTAIRS`: contribution `-0.002679`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `-0.007976`
- `lag_09__T_flash_duration_sum`: contribution `-0.001346`
- `lag_06__T_A_site_active_infernos`: contribution `-0.001187`

### tick `67381`, seconds `0.50`, LSTM delta `-0.0715`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003111`
- `lag_01__T_place_TSPAWN`: contribution `-0.003104`
- `lag_00__T_velocity_mean`: contribution `-0.002534`
- `lag_00__CT_velocity_mean`: contribution `-0.002457`
- `lag_01__molly_inv_diff`: contribution `-0.001308`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.001308`
- `lag_01__utility_inv_diff`: contribution `-0.001184`
- `lag_01__T_utility_inv`: contribution `-0.000967`
- `lag_01__T3__utility_total`: contribution `-0.000875`
- `lag_01__T_molly_inv`: contribution `-0.000835`

### tick `68437`, seconds `17.00`, LSTM delta `-0.0420`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.004440`
- `lag_00__CT5__is_scoped`: contribution `-0.003636`
- `lag_15__T_place_OUTSIDETUNNEL`: contribution `-0.001462`
- `lag_15__T1__is_scoped`: contribution `-0.001453`
- `lag_00__CT_scoped_count`: contribution `-0.001305`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67413`, seconds `1.00`, LSTM delta `-0.0350`

Top all feature movements:
- `lag_02__CT_place_CTSPAWN`: contribution `-0.002289`
- `lag_02__T_place_TSPAWN`: contribution `-0.002083`
- `lag_00__CT2__smoke`: contribution `-0.000922`
- `lag_02__molly_inv_diff`: contribution `-0.000915`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000906`

Top utility-only movements:
- `lag_00__CT2__smoke`: contribution `-0.000922`
- `lag_02__molly_inv_diff`: contribution `-0.000915`
- `lag_02__utility_inv_diff`: contribution `-0.000871`
- `lag_02__T_utility_inv`: contribution `-0.000682`
- `lag_02__flash_inv_diff`: contribution `-0.000581`

### tick `70965`, seconds `56.50`, LSTM delta `-0.0297`

Top all feature movements:
- `lag_03__T3__flash_duration`: contribution `-0.003165`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.001954`
- `lag_01__CT5__is_scoped`: contribution `-0.001492`
- `lag_03__T2__duck_amount`: contribution `-0.001393`
- `lag_00__CT_velocity_mean`: contribution `+0.001251`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `-0.003165`
