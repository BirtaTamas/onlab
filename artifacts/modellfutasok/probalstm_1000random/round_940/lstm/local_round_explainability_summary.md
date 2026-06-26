# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `10`

## Largest probability jumps

- tick `82709`, seconds `94.00`, LSTM `0.1443`, delta `-0.3023`
- tick `81301`, seconds `72.00`, LSTM `0.2290`, delta `-0.2673`
- tick `80789`, seconds `64.00`, LSTM `0.5075`, delta `-0.1558`
- tick `82325`, seconds `88.00`, LSTM `0.2921`, delta `+0.0858`
- tick `80597`, seconds `61.00`, LSTM `0.6098`, delta `+0.0782`
- tick `82741`, seconds `94.50`, LSTM `0.0711`, delta `-0.0731`
- tick `82357`, seconds `88.50`, LSTM `0.3498`, delta `+0.0577`
- tick `82549`, seconds `91.50`, LSTM `0.3579`, delta `+0.0575`
- tick `81653`, seconds `77.50`, LSTM `0.1229`, delta `-0.0555`
- tick `81333`, seconds `72.50`, LSTM `0.1739`, delta `-0.0551`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003215`, |coef| `0.003215`
- `lag_00__kill_diff_last_3s`: coefficient `0.002879`, |coef| `0.002879`
- `lag_00__T_kills_last_3s`: coefficient `-0.002544`, |coef| `0.002544`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.002450`, |coef| `0.002450`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.002250`, |coef| `0.002250`
- `lag_00__damage_diff_last_5s`: coefficient `0.001914`, |coef| `0.001914`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_00__T_damage_last_5s`: coefficient `-0.001863`, |coef| `0.001863`
- `lag_12__CT2__is_scoped`: coefficient `0.001830`, |coef| `0.001830`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001655`, |coef| `0.001655`
- `lag_00__CT2__flash`: coefficient `0.001630`, |coef| `0.001630`
- `lag_09__CT2__is_scoped`: coefficient `-0.001615`, |coef| `0.001615`
- `lag_03__T_place_RAMP`: coefficient `-0.001560`, |coef| `0.001560`
- `lag_10__CT_shots_fired_sum`: coefficient `0.001503`, |coef| `0.001503`

## Top 10 utility ridge features

- `lag_00__CT2__flash`: coefficient `0.001630` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001272` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.001196` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001154` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `-0.001141` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.001036` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.001011` (lowers CT win probability)
- `lag_06__T_active_infernos`: coefficient `0.000946` (raises CT win probability)
- `lag_09__T4__smoke`: coefficient `0.000933` (raises CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000888` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003215` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002879` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002544` (lowers CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.002450` (raises CT win probability)
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.002250` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001914` (raises CT win probability)
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001878` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001863` (lowers CT win probability)
- `lag_12__CT2__is_scoped`: coefficient `0.001830` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001816` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `82709`, seconds `94.00`, LSTM delta `-0.3023`

Top all feature movements:
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.018416`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.016914`
- `lag_12__CT2__is_scoped`: contribution `-0.011202`
- `lag_09__CT2__is_scoped`: contribution `-0.009887`
- `lag_00__T_kills_last_3s`: contribution `-0.008060`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.005897`
- `lag_00__CT2__utility_total`: contribution `-0.003597`

### tick `81301`, seconds `72.00`, LSTM delta `-0.2673`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008060`
- `lag_00__kill_diff_last_3s`: contribution `-0.006930`
- `lag_01__CT2__is_scoped`: contribution `-0.006629`
- `lag_00__CT_flashed_players`: contribution `-0.006240`
- `lag_05__T5__is_scoped`: contribution `-0.005824`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.003920`
- `lag_01__T5__flash_duration`: contribution `-0.003396`
- `lag_06__T_B_site_active_infernos`: contribution `-0.003382`

### tick `80789`, seconds `64.00`, LSTM delta `-0.1558`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013861`
- `lag_01__T_place_SIDEENTRANCE`: contribution `-0.009164`
- `lag_00__T_kills_last_3s`: contribution `-0.008060`
- `lag_06__T_place_SIDEENTRANCE`: contribution `-0.005770`
- `lag_00__T_damage_last_5s`: contribution `-0.004467`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82325`, seconds `88.00`, LSTM delta `+0.0858`

Top all feature movements:
- `lag_01__CT_place_TSIDEUPPER`: contribution `+0.007142`
- `lag_00__kill_diff_last_3s`: contribution `+0.006930`
- `lag_04__T_place_CTSPAWN`: contribution `+0.005609`
- `lag_03__T_place_SIDEENTRANCE`: contribution `+0.004751`
- `lag_00__CT_kills_last_3s`: contribution `+0.003278`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80597`, seconds `61.00`, LSTM delta `+0.0782`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.008865`
- `lag_00__kill_diff_last_3s`: contribution `+0.006930`
- `lag_05__T_place_TSIDELOWER`: contribution `+0.004421`
- `lag_00__damage_diff_last_5s`: contribution `+0.004188`
- `lag_11__CT4__duck_amount`: contribution `+0.003886`

Top utility-only movements:
- No utility movement among the top local contributors.
