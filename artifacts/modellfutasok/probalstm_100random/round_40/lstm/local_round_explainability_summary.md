# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `25771`, seconds `55.50`, LSTM `0.3229`, delta `-0.2691`
- tick `25931`, seconds `58.00`, LSTM `0.0286`, delta `-0.1195`
- tick `25899`, seconds `57.50`, LSTM `0.1481`, delta `-0.0769`
- tick `25611`, seconds `53.00`, LSTM `0.6100`, delta `-0.0465`
- tick `25803`, seconds `56.00`, LSTM `0.2786`, delta `-0.0442`
- tick `25867`, seconds `57.00`, LSTM `0.2250`, delta `-0.0345`
- tick `22955`, seconds `11.50`, LSTM `0.5694`, delta `+0.0302`
- tick `23723`, seconds `23.50`, LSTM `0.5639`, delta `-0.0292`
- tick `22315`, seconds `1.50`, LSTM `0.5009`, delta `-0.0292`
- tick `23851`, seconds `25.50`, LSTM `0.6062`, delta `+0.0289`

## Top 15 local ridge features

- `lag_05__T_flashed_players`: coefficient `-0.003436`, |coef| `0.003436`
- `lag_01__T_flash_duration_sum`: coefficient `-0.002420`, |coef| `0.002420`
- `lag_05__T4__flash_duration`: coefficient `-0.002307`, |coef| `0.002307`
- `lag_00__T_kills_last_3s`: coefficient `-0.002254`, |coef| `0.002254`
- `lag_01__T4__flash_duration`: coefficient `-0.002254`, |coef| `0.002254`
- `lag_01__T5__flash_duration`: coefficient `-0.002202`, |coef| `0.002202`
- `lag_09__T3__duck_amount`: coefficient `-0.002154`, |coef| `0.002154`
- `lag_01__T_place_UNDERPASS`: coefficient `0.002134`, |coef| `0.002134`
- `lag_12__T_place_UNDERPASS`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_05__T_flash_duration_sum`: coefficient `-0.002113`, |coef| `0.002113`
- `lag_00__CT4__molly`: coefficient `0.002018`, |coef| `0.002018`
- `lag_00__CT4__alive`: coefficient `0.002009`, |coef| `0.002009`
- `lag_00__CT4__hp`: coefficient `0.001980`, |coef| `0.001980`
- `lag_05__CT_place_BANANA`: coefficient `0.001918`, |coef| `0.001918`
- `lag_03__T3__duck_amount`: coefficient `0.001880`, |coef| `0.001880`

## Top 10 utility ridge features

- `lag_01__T_flash_duration_sum`: coefficient `-0.002420` (lowers CT win probability)
- `lag_05__T4__flash_duration`: coefficient `-0.002307` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.002254` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `-0.002202` (lowers CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.002113` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.002018` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001759` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001653` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.001489` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001420` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_flashed_players`: coefficient `-0.003436` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002254` (lowers CT win probability)
- `lag_09__T3__duck_amount`: coefficient `-0.002154` (lowers CT win probability)
- `lag_01__T_place_UNDERPASS`: coefficient `0.002134` (raises CT win probability)
- `lag_12__T_place_UNDERPASS`: coefficient `-0.002122` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.002009` (raises CT win probability)
- `lag_00__CT4__hp`: coefficient `0.001980` (raises CT win probability)
- `lag_05__CT_place_BANANA`: coefficient `0.001918` (raises CT win probability)
- `lag_03__T3__duck_amount`: coefficient `0.001880` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001779` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `25771`, seconds `55.50`, LSTM delta `-0.2691`

Top all feature movements:
- `lag_05__T_flashed_players`: contribution `-0.019891`
- `lag_01__T_flash_duration_sum`: contribution `-0.011480`
- `lag_01__T4__flash_duration`: contribution `-0.009964`
- `lag_01__T5__flash_duration`: contribution `-0.009000`
- `lag_01__T_place_UNDERPASS`: contribution `-0.008361`

Top utility-only movements:
- `lag_01__T_flash_duration_sum`: contribution `-0.011480`
- `lag_01__T4__flash_duration`: contribution `-0.009964`
- `lag_01__T5__flash_duration`: contribution `-0.009000`
- `lag_05__T4__flash_duration`: contribution `-0.007069`
- `lag_05__T_flash_duration_sum`: contribution `-0.006146`

### tick `25931`, seconds `58.00`, LSTM delta `-0.1195`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007141`
- `lag_06__T4__flash_duration`: contribution `-0.004894`
- `lag_06__T_flash_duration_sum`: contribution `-0.004810`
- `lag_10__T_flashed_players`: contribution `-0.004379`
- `lag_00__kill_diff_last_3s`: contribution `-0.004122`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.004894`
- `lag_06__T_flash_duration_sum`: contribution `-0.004810`
- `lag_06__T5__flash_duration`: contribution `-0.003794`
- `lag_00__CT1__utility_total`: contribution `-0.002763`
- `lag_00__CT1__molly`: contribution `-0.002112`

### tick `25899`, seconds `57.50`, LSTM delta `-0.0769`

Top all feature movements:
- `lag_05__T4__flash_duration`: contribution `-0.010197`
- `lag_05__T_flash_duration_sum`: contribution `-0.010023`
- `lag_05__T5__flash_duration`: contribution `-0.007188`
- `lag_05__T_flashed_players`: contribution `-0.006630`
- `lag_05__T2__flash_duration`: contribution `-0.003672`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.010197`
- `lag_05__T_flash_duration_sum`: contribution `-0.010023`
- `lag_05__T5__flash_duration`: contribution `-0.007188`
- `lag_05__T2__flash_duration`: contribution `-0.003672`

### tick `25611`, seconds `53.00`, LSTM delta `-0.0465`

Top all feature movements:
- `lag_00__T2__duck_amount`: contribution `-0.004346`
- `lag_00__T_flashed_players`: contribution `-0.004077`
- `lag_07__T_place_SECONDMID`: contribution `-0.002666`
- `lag_00__T4__flash_duration`: contribution `-0.002513`
- `lag_00__CT_place_BANANA`: contribution `-0.002486`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.002513`
- `lag_14__T_A_site_active_infernos`: contribution `-0.002001`
- `lag_00__T_flash_duration_sum`: contribution `-0.001333`
- `lag_00__T5__flash_duration`: contribution `-0.001121`

### tick `25803`, seconds `56.00`, LSTM delta `-0.0442`

Top all feature movements:
- `lag_05__T2__duck_amount`: contribution `+0.005356`
- `lag_06__T_flashed_players`: contribution `-0.005145`
- `lag_06__T4__flash_duration`: contribution `-0.003393`
- `lag_06__T_flash_duration_sum`: contribution `-0.002950`
- `lag_02__T_flash_duration_sum`: contribution `-0.002679`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.003393`
- `lag_06__T_flash_duration_sum`: contribution `-0.002950`
- `lag_02__T_flash_duration_sum`: contribution `-0.002679`
- `lag_02__T5__flash_duration`: contribution `-0.002382`
- `lag_06__T5__flash_duration`: contribution `-0.002303`
