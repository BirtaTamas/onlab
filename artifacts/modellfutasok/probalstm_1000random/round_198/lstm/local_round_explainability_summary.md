# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `55608`, seconds `102.50`, LSTM `0.7292`, delta `+0.3741`
- tick `54392`, seconds `83.50`, LSTM `0.5793`, delta `+0.2952`
- tick `55512`, seconds `101.00`, LSTM `0.5736`, delta `-0.2500`
- tick `54488`, seconds `85.00`, LSTM `0.8305`, delta `+0.1902`
- tick `55544`, seconds `101.50`, LSTM `0.4505`, delta `-0.1231`
- tick `55576`, seconds `102.00`, LSTM `0.3552`, delta `-0.0953`
- tick `54744`, seconds `89.00`, LSTM `0.8368`, delta `-0.0872`
- tick `53464`, seconds `69.00`, LSTM `0.3764`, delta `-0.0831`
- tick `51928`, seconds `45.00`, LSTM `0.4093`, delta `+0.0590`
- tick `54008`, seconds `77.50`, LSTM `0.2597`, delta `-0.0547`

## Top 15 local ridge features

- `lag_06__T_place_STAIRS`: coefficient `-0.006724`, |coef| `0.006724`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.005293`, |coef| `0.005293`
- `lag_09__T_place_STAIRS`: coefficient `0.002664`, |coef| `0.002664`
- `lag_00__kill_diff_last_3s`: coefficient `0.002621`, |coef| `0.002621`
- `lag_00__damage_diff_last_5s`: coefficient `0.002500`, |coef| `0.002500`
- `lag_00__CT_kills_last_3s`: coefficient `0.002393`, |coef| `0.002393`
- `lag_12__CT3__flash_duration`: coefficient `-0.002346`, |coef| `0.002346`
- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002167`, |coef| `0.002167`
- `lag_00__CT_damage_last_5s`: coefficient `0.001954`, |coef| `0.001954`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001922`, |coef| `0.001922`
- `lag_00__CT_place_TRUCK`: coefficient `0.001865`, |coef| `0.001865`
- `lag_15__T_place_CONNECTOR`: coefficient `0.001800`, |coef| `0.001800`
- `lag_03__T_place_STAIRS`: coefficient `0.001738`, |coef| `0.001738`
- `lag_11__CT3__flash_duration`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_03__T_A_site_active_infernos`: coefficient `0.001653`, |coef| `0.001653`

## Top 10 utility ridge features

- `lag_12__CT3__flash_duration`: coefficient `-0.002346` (lowers CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.001700` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `0.001653` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.001285` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.001252` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.001084` (lowers CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `-0.001060` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.001003` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000900` (lowers CT win probability)
- `lag_08__T5__molly`: coefficient `-0.000828` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__T_place_STAIRS`: coefficient `-0.006724` (lowers CT win probability)
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.005293` (lowers CT win probability)
- `lag_09__T_place_STAIRS`: coefficient `0.002664` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002621` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002500` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002393` (raises CT win probability)
- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002167` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001954` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001922` (raises CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001865` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `55608`, seconds `102.50`, LSTM delta `+0.3741`

Top all feature movements:
- `lag_06__T_place_STAIRS`: contribution `+0.128734`
- `lag_09__T_place_STAIRS`: contribution `+0.051002`
- `lag_03__T3__is_scoped`: contribution `+0.009010`
- `lag_04__T3__is_scoped`: contribution `+0.007398`
- `lag_00__CT_kills_last_3s`: contribution `+0.006910`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.004921`
- `lag_03__T_active_infernos`: contribution `+0.002089`

### tick `54392`, seconds `83.50`, LSTM delta `+0.2952`

Top all feature movements:
- `lag_00__T_place_SNIPERSNEST`: contribution `+0.094056`
- `lag_12__CT3__flash_duration`: contribution `+0.017345`
- `lag_12__T_place_CONNECTOR`: contribution `+0.007953`
- `lag_00__CT_kills_last_3s`: contribution `+0.006910`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006675`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `+0.017345`
- `lag_04__T_A_site_active_infernos`: contribution `+0.003728`
- `lag_12__CT_flash_duration_sum`: contribution `+0.003505`

### tick `55512`, seconds `101.00`, LSTM delta `-0.2500`

Top all feature movements:
- `lag_06__T_place_STAIRS`: contribution `-0.128734`
- `lag_03__T_place_STAIRS`: contribution `-0.033280`
- `lag_00__kill_diff_last_3s`: contribution `-0.006310`
- `lag_07__CT_place_UNDERPASS`: contribution `-0.006103`
- `lag_13__T_place_CONNECTOR`: contribution `-0.005592`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.004921`
- `lag_03__T_active_infernos`: contribution `-0.002089`

### tick `54488`, seconds `85.00`, LSTM delta `+0.1902`

Top all feature movements:
- `lag_03__T_place_SNIPERSNEST`: contribution `+0.038506`
- `lag_15__CT3__flash_duration`: contribution `+0.009499`
- `lag_15__T_place_CONNECTOR`: contribution `+0.008716`
- `lag_00__CT_kills_last_3s`: contribution `+0.006910`
- `lag_00__kill_diff_last_3s`: contribution `+0.006310`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `+0.009499`

### tick `55544`, seconds `101.50`, LSTM delta `-0.1231`

Top all feature movements:
- `lag_07__T_place_STAIRS`: contribution `-0.029538`
- `lag_04__T_place_STAIRS`: contribution `-0.024558`
- `lag_14__T_place_CONNECTOR`: contribution `-0.006820`
- `lag_11__T3__is_scoped`: contribution `-0.005117`
- `lag_08__CT_place_UNDERPASS`: contribution `-0.004916`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.003728`
