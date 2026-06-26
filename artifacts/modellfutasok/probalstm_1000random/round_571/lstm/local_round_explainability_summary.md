# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `32`

## Largest probability jumps

- tick `293727`, seconds `39.00`, LSTM `0.6965`, delta `+0.1784`
- tick `293791`, seconds `40.00`, LSTM `0.8901`, delta `+0.1522`
- tick `293919`, seconds `42.00`, LSTM `0.9294`, delta `+0.0773`
- tick `293823`, seconds `40.50`, LSTM `0.8471`, delta `-0.0430`
- tick `293759`, seconds `39.50`, LSTM `0.7379`, delta `+0.0414`
- tick `293663`, seconds `38.00`, LSTM `0.5291`, delta `+0.0238`
- tick `294687`, seconds `54.00`, LSTM `0.9692`, delta `+0.0234`
- tick `293311`, seconds `32.50`, LSTM `0.5070`, delta `+0.0187`
- tick `292095`, seconds `13.50`, LSTM `0.5163`, delta `+0.0186`
- tick `291999`, seconds `12.00`, LSTM `0.4911`, delta `+0.0163`

## Top 15 local ridge features

- `lag_00__T_place_TROPHY`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001520`, |coef| `0.001520`
- `lag_14__T_place_ROOF`: coefficient `0.001485`, |coef| `0.001485`
- `lag_02__CT_flashed_players`: coefficient `0.001477`, |coef| `0.001477`
- `lag_00__CT_kills_last_3s`: coefficient `0.001401`, |coef| `0.001401`
- `lag_02__CT3__flash_duration`: coefficient `0.001269`, |coef| `0.001269`
- `lag_00__T_place_VENDING`: coefficient `0.001254`, |coef| `0.001254`
- `lag_02__T_place_TROPHY`: coefficient `-0.001249`, |coef| `0.001249`
- `lag_00__CT2__shots_fired`: coefficient `0.001196`, |coef| `0.001196`
- `lag_00__kill_diff_last_3s`: coefficient `0.001168`, |coef| `0.001168`
- `lag_00__CT_damage_last_5s`: coefficient `0.001113`, |coef| `0.001113`
- `lag_04__CT3__flash_duration`: coefficient `0.001064`, |coef| `0.001064`
- `lag_02__CT_flash_duration_sum`: coefficient `0.001048`, |coef| `0.001048`
- `lag_13__CT_place_GARAGE`: coefficient `-0.000946`, |coef| `0.000946`
- `lag_00__T4__flash`: coefficient `-0.000940`, |coef| `0.000940`

## Top 10 utility ridge features

- `lag_02__CT3__flash_duration`: coefficient `0.001269` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001064` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.001048` (raises CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000940` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000853` (lowers CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000841` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.000792` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000774` (raises CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `-0.000757` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000754` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_TROPHY`: coefficient `-0.001674` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001520` (raises CT win probability)
- `lag_14__T_place_ROOF`: coefficient `0.001485` (raises CT win probability)
- `lag_02__CT_flashed_players`: coefficient `0.001477` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001401` (raises CT win probability)
- `lag_00__T_place_VENDING`: coefficient `0.001254` (raises CT win probability)
- `lag_02__T_place_TROPHY`: coefficient `-0.001249` (lowers CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.001196` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001168` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001113` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `293727`, seconds `39.00`, LSTM delta `+0.1784`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.010616`
- `lag_14__T_place_ROOF`: contribution `+0.008407`
- `lag_02__CT3__flash_duration`: contribution `+0.006912`
- `lag_13__CT_place_GARAGE`: contribution `+0.006799`
- `lag_02__CT_flashed_players`: contribution `+0.006471`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `+0.006912`
- `lag_02__CT_flash_duration_sum`: contribution `+0.003305`
- `lag_00__T4__flash`: contribution `+0.002554`
- `lag_09__T_A_site_active_smokes`: contribution `+0.002154`

### tick `293791`, seconds `40.00`, LSTM delta `+0.1522`

Top all feature movements:
- `lag_02__T_place_TROPHY`: contribution `+0.007918`
- `lag_04__CT3__flash_duration`: contribution `+0.005798`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005281`
- `lag_02__T_place_VENDING`: contribution `+0.004583`
- `lag_15__CT_place_GARAGE`: contribution `+0.004269`

Top utility-only movements:
- `lag_04__CT3__flash_duration`: contribution `+0.005798`
- `lag_04__CT_flash_duration_sum`: contribution `+0.001865`
- `lag_02__T4__flash`: contribution `+0.001526`

### tick `293919`, seconds `42.00`, LSTM delta `+0.0773`

Top all feature movements:
- `lag_02__CT_flashed_players`: contribution `+0.006471`
- `lag_02__CT3__flash_duration`: contribution `-0.005740`
- `lag_02__T4__flash_duration`: contribution `+0.005596`
- `lag_02__T3__flash_duration`: contribution `+0.004958`
- `lag_02__CT2__flash_duration`: contribution `+0.003755`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `-0.005740`
- `lag_02__T4__flash_duration`: contribution `+0.005596`
- `lag_02__T3__flash_duration`: contribution `+0.004958`
- `lag_02__CT2__flash_duration`: contribution `+0.003755`
- `lag_02__CT_flash_duration_sum`: contribution `+0.003393`

### tick `293823`, seconds `40.50`, LSTM delta `-0.0430`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.013731`
- `lag_00__CT2__shots_fired`: contribution `-0.007727`
- `lag_02__CT_flashed_players`: contribution `-0.003236`
- `lag_04__CT3__is_scoped`: contribution `-0.003174`
- `lag_01__CT_shots_fired_sum`: contribution `+0.002930`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `-0.001225`

### tick `293759`, seconds `39.50`, LSTM delta `+0.0414`

Top all feature movements:
- `lag_01__T_place_TROPHY`: contribution `+0.005819`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005281`
- `lag_15__T_place_ROOF`: contribution `+0.004229`
- `lag_03__CT3__flash_duration`: contribution `+0.003800`
- `lag_01__T_place_VENDING`: contribution `+0.003108`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `+0.003800`
