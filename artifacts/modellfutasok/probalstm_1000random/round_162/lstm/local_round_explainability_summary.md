# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-nrg-dust2-QDtqFlW1Z9UhZpBNOAavnd/heroic-vs-nrg-dust2.csv`
- round_num: `6`

## Largest probability jumps

- tick `35800`, seconds `33.50`, LSTM `0.5241`, delta `-0.2525`
- tick `36184`, seconds `39.50`, LSTM `0.2952`, delta `-0.2428`
- tick `35736`, seconds `32.50`, LSTM `0.8194`, delta `+0.1860`
- tick `34232`, seconds `9.00`, LSTM `0.7854`, delta `+0.0932`
- tick `35096`, seconds `22.50`, LSTM `0.6352`, delta `-0.0885`
- tick `36568`, seconds `45.50`, LSTM `0.2284`, delta `+0.0719`
- tick `36216`, seconds `40.00`, LSTM `0.2437`, delta `-0.0515`
- tick `35832`, seconds `34.00`, LSTM `0.5735`, delta `+0.0495`
- tick `33752`, seconds `1.50`, LSTM `0.7789`, delta `-0.0476`
- tick `37432`, seconds `59.00`, LSTM `0.2075`, delta `-0.0463`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002049`, |coef| `0.002049`
- `lag_00__T_kills_last_3s`: coefficient `-0.002041`, |coef| `0.002041`
- `lag_01__CT4__shots_fired`: coefficient `0.002036`, |coef| `0.002036`
- `lag_00__T_place_UNDERA`: coefficient `-0.001873`, |coef| `0.001873`
- `lag_01__T_smokes_last_5s`: coefficient `-0.001715`, |coef| `0.001715`
- `lag_00__CT2__is_scoped`: coefficient `0.001663`, |coef| `0.001663`
- `lag_02__CT4__duck_amount`: coefficient `0.001616`, |coef| `0.001616`
- `lag_00__CT3__duck_amount`: coefficient `0.001609`, |coef| `0.001609`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001608`, |coef| `0.001608`
- `lag_00__CT_place_LONGA`: coefficient `0.001584`, |coef| `0.001584`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001580`, |coef| `0.001580`
- `lag_14__CT2__duck_amount`: coefficient `0.001534`, |coef| `0.001534`
- `lag_04__T_place_LONGA`: coefficient `-0.001531`, |coef| `0.001531`
- `lag_08__T_place_LONGA`: coefficient `-0.001529`, |coef| `0.001529`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001498`, |coef| `0.001498`

## Top 10 utility ridge features

- `lag_01__T_smokes_last_5s`: coefficient `-0.001715` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.001402` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.001119` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.001111` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.001109` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001049` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.001032` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.000998` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000994` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `-0.000948` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002049` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002041` (lowers CT win probability)
- `lag_01__CT4__shots_fired`: coefficient `0.002036` (raises CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.001873` (lowers CT win probability)
- `lag_00__CT2__is_scoped`: coefficient `0.001663` (raises CT win probability)
- `lag_02__CT4__duck_amount`: coefficient `0.001616` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001609` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001608` (raises CT win probability)
- `lag_00__CT_place_LONGA`: coefficient `0.001584` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001580` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35800`, seconds `33.50`, LSTM delta `-0.2525`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.015369`
- `lag_01__CT4__shots_fired`: contribution `-0.015358`
- `lag_01__CT2__is_scoped`: contribution `-0.007833`
- `lag_11__CT_flashed_players`: contribution `-0.007140`
- `lag_04__T_place_LONGA`: contribution `-0.006523`

Top utility-only movements:
- `lag_11__T3__flash_duration`: contribution `-0.004974`
- `lag_02__CT3__flash_duration`: contribution `-0.004852`
- `lag_11__CT3__flash_duration`: contribution `-0.004133`

### tick `36184`, seconds `39.50`, LSTM delta `-0.2428`

Top all feature movements:
- `lag_13__CT_shots_fired_sum`: contribution `-0.014251`
- `lag_00__CT2__is_scoped`: contribution `-0.010178`
- `lag_13__CT4__shots_fired`: contribution `-0.010145`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.009160`
- `lag_14__CT_flashed_players`: contribution `-0.008984`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.004142`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.003922`

### tick `35736`, seconds `32.50`, LSTM delta `+0.1860`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `+0.010178`
- `lag_09__T3__flash_duration`: contribution `+0.007354`
- `lag_00__CT3__duck_amount`: contribution `+0.005985`
- `lag_02__CT4__duck_amount`: contribution `+0.005935`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005489`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `+0.007354`
- `lag_09__CT3__flash_duration`: contribution `+0.005440`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.003912`
- `lag_00__CT3__flash_duration`: contribution `+0.003472`

### tick `34232`, seconds `9.00`, LSTM delta `+0.0932`

Top all feature movements:
- `lag_01__T_smokes_last_5s`: contribution `+0.050292`
- `lag_00__CT2__is_scoped`: contribution `-0.010178`
- `lag_11__T_smokes_last_5s`: contribution `+0.009138`
- `lag_12__T_smokes_last_5s`: contribution `+0.005996`
- `lag_00__kill_diff_last_3s`: contribution `+0.004933`

Top utility-only movements:
- `lag_01__T_smokes_last_5s`: contribution `+0.050292`
- `lag_11__T_smokes_last_5s`: contribution `+0.009138`
- `lag_12__T_smokes_last_5s`: contribution `+0.005996`

### tick `35096`, seconds `22.50`, LSTM delta `-0.0885`

Top all feature movements:
- `lag_15__T_flashes_last_5s`: contribution `-0.006523`
- `lag_00__T_kills_last_3s`: contribution `-0.006466`
- `lag_00__kill_diff_last_3s`: contribution `-0.004933`
- `lag_00__CT_place_LONGA`: contribution `-0.004230`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003352`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `-0.006523`
- `lag_04__CT1__flash_duration`: contribution `-0.002912`
- `lag_00__CT1__flash_duration`: contribution `-0.002556`
