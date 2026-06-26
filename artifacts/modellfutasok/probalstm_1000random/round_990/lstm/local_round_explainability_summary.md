# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `64667`, seconds `25.50`, LSTM `0.2028`, delta `-0.3630`
- tick `65051`, seconds `31.50`, LSTM `0.0593`, delta `-0.1700`
- tick `64859`, seconds `28.50`, LSTM `0.2624`, delta `-0.0566`
- tick `64795`, seconds `27.50`, LSTM `0.2853`, delta `+0.0545`
- tick `65083`, seconds `32.00`, LSTM `0.0178`, delta `-0.0415`
- tick `64187`, seconds `18.00`, LSTM `0.6314`, delta `-0.0362`
- tick `64891`, seconds `29.00`, LSTM `0.2277`, delta `-0.0347`
- tick `64283`, seconds `19.50`, LSTM `0.5757`, delta `-0.0340`
- tick `64827`, seconds `28.00`, LSTM `0.3190`, delta `+0.0337`
- tick `64923`, seconds `29.50`, LSTM `0.1943`, delta `-0.0333`

## Top 15 local ridge features

- `lag_03__T_place_SCAFFOLDING`: coefficient `-0.002590`, |coef| `0.002590`
- `lag_01__T_place_SCAFFOLDING`: coefficient `0.002185`, |coef| `0.002185`
- `lag_10__CT5__flash_duration`: coefficient `-0.001199`, |coef| `0.001199`
- `lag_13__T_place_SCAFFOLDING`: coefficient `0.001126`, |coef| `0.001126`
- `lag_15__T_place_SCAFFOLDING`: coefficient `-0.000995`, |coef| `0.000995`
- `lag_05__T_place_SCAFFOLDING`: coefficient `-0.000922`, |coef| `0.000922`
- `lag_07__T_place_SCAFFOLDING`: coefficient `0.000885`, |coef| `0.000885`
- `lag_08__CT5__flash_duration`: coefficient `-0.000811`, |coef| `0.000811`
- `lag_06__T_burning_players`: coefficient `-0.000767`, |coef| `0.000767`
- `lag_10__CT3__flash_duration`: coefficient `-0.000750`, |coef| `0.000750`
- `lag_08__CT4__is_scoped`: coefficient `-0.000734`, |coef| `0.000734`
- `lag_12__CT5__flash_duration`: coefficient `-0.000727`, |coef| `0.000727`
- `lag_03__T_place_STAIRS`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_06__CT4__is_scoped`: coefficient `-0.000699`, |coef| `0.000699`
- `lag_12__CT_place_TRUCK`: coefficient `0.000693`, |coef| `0.000693`

## Top 10 utility ridge features

- `lag_10__CT5__flash_duration`: coefficient `-0.001199` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `-0.000811` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000750` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000727` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.000661` (lowers CT win probability)
- `lag_09__T_flashes_last_5s`: coefficient `-0.000659` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000617` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000615` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.000596` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `-0.000582` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_SCAFFOLDING`: coefficient `-0.002590` (lowers CT win probability)
- `lag_01__T_place_SCAFFOLDING`: coefficient `0.002185` (raises CT win probability)
- `lag_13__T_place_SCAFFOLDING`: coefficient `0.001126` (raises CT win probability)
- `lag_15__T_place_SCAFFOLDING`: coefficient `-0.000995` (lowers CT win probability)
- `lag_05__T_place_SCAFFOLDING`: coefficient `-0.000922` (lowers CT win probability)
- `lag_07__T_place_SCAFFOLDING`: coefficient `0.000885` (raises CT win probability)
- `lag_06__T_burning_players`: coefficient `-0.000767` (lowers CT win probability)
- `lag_08__CT4__is_scoped`: coefficient `-0.000734` (lowers CT win probability)
- `lag_03__T_place_STAIRS`: coefficient `-0.000717` (lowers CT win probability)
- `lag_06__CT4__is_scoped`: coefficient `-0.000699` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `64667`, seconds `25.50`, LSTM delta `-0.3630`

Top all feature movements:
- `lag_03__T_place_SCAFFOLDING`: contribution `-0.088199`
- `lag_01__T_place_SCAFFOLDING`: contribution `-0.074403`
- `lag_10__CT5__flash_duration`: contribution `-0.007203`
- `lag_09__T_flashes_last_5s`: contribution `-0.005973`
- `lag_10__CT3__flash_duration`: contribution `-0.005819`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.007203`
- `lag_09__T_flashes_last_5s`: contribution `-0.005973`
- `lag_10__CT3__flash_duration`: contribution `-0.005819`
- `lag_00__CT3__flash_duration`: contribution `-0.004471`
- `lag_07__CT4__flash_duration`: contribution `-0.003181`

### tick `65051`, seconds `31.50`, LSTM delta `-0.1700`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `-0.038348`
- `lag_15__T_place_SCAFFOLDING`: contribution `-0.033898`
- `lag_03__T_place_STAIRS`: contribution `-0.013723`
- `lag_00__T_place_STAIRS`: contribution `-0.008746`
- `lag_11__T_flashes_last_5s`: contribution `-0.003001`

Top utility-only movements:
- `lag_11__T_flashes_last_5s`: contribution `-0.003001`
- `lag_11__T5__flash_duration`: contribution `-0.001251`
- `lag_12__T1__flash_duration`: contribution `-0.001126`
- `lag_11__T_flash_duration_sum`: contribution `-0.001002`

### tick `64859`, seconds `28.50`, LSTM delta `-0.0566`

Top all feature movements:
- `lag_07__T_place_SCAFFOLDING`: contribution `-0.030130`
- `lag_09__T_place_SCAFFOLDING`: contribution `+0.003466`
- `lag_05__T_flashes_last_5s`: contribution `-0.002798`
- `lag_08__T5__is_scoped`: contribution `+0.001955`
- `lag_00__T_kills_last_3s`: contribution `+0.001782`

Top utility-only movements:
- `lag_05__T_flashes_last_5s`: contribution `-0.002798`
- `lag_15__T_flashes_last_5s`: contribution `-0.001542`

### tick `64795`, seconds `27.50`, LSTM delta `+0.0545`

Top all feature movements:
- `lag_05__T_place_SCAFFOLDING`: contribution `+0.031415`
- `lag_07__T_place_SCAFFOLDING`: contribution `+0.030130`
- `lag_13__CT_place_JUNGLE`: contribution `+0.003996`
- `lag_14__CT5__flash_duration`: contribution `-0.003333`
- `lag_13__T_flashes_last_5s`: contribution `+0.002825`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `-0.003333`
- `lag_13__T_flashes_last_5s`: contribution `+0.002825`

### tick `65083`, seconds `32.00`, LSTM delta `-0.0415`

Top all feature movements:
- `lag_14__T_place_SCAFFOLDING`: contribution `-0.012394`
- `lag_01__T_place_STAIRS`: contribution `+0.008809`
- `lag_04__T_place_STAIRS`: contribution `-0.005851`
- `lag_12__T_flashes_last_5s`: contribution `-0.002573`
- `lag_15__T5__is_scoped`: contribution `-0.002510`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.002573`
- `lag_10__CT3__flash_duration`: contribution `+0.001627`
- `lag_13__CT_flash_duration_sum`: contribution `-0.000896`
- `lag_07__CT4__flash_duration`: contribution `-0.000885`
