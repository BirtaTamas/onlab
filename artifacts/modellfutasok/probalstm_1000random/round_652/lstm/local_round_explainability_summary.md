# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `32858`, seconds `71.00`, LSTM `0.6098`, delta `-0.2794`
- tick `33178`, seconds `76.00`, LSTM `0.8149`, delta `+0.2547`
- tick `32506`, seconds `65.50`, LSTM `0.5928`, delta `+0.2518`
- tick `32538`, seconds `66.00`, LSTM `0.7861`, delta `+0.1933`
- tick `32378`, seconds `63.50`, LSTM `0.3501`, delta `-0.1810`
- tick `31194`, seconds `45.00`, LSTM `0.6771`, delta `+0.1222`
- tick `32026`, seconds `58.00`, LSTM `0.5656`, delta `-0.1181`
- tick `32410`, seconds `64.00`, LSTM `0.2887`, delta `-0.0615`
- tick `31290`, seconds `46.50`, LSTM `0.7269`, delta `+0.0409`
- tick `33370`, seconds `79.00`, LSTM `0.8363`, delta `-0.0402`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003659`, |coef| `0.003659`
- `lag_00__damage_diff_last_5s`: coefficient `0.003283`, |coef| `0.003283`
- `lag_09__T_shots_fired_sum`: coefficient `-0.003053`, |coef| `0.003053`
- `lag_00__CT_kills_last_3s`: coefficient `0.002430`, |coef| `0.002430`
- `lag_00__T_kills_last_3s`: coefficient `-0.002150`, |coef| `0.002150`
- `lag_00__T_place_LONGA`: coefficient `-0.002088`, |coef| `0.002088`
- `lag_08__T_shots_fired_sum`: coefficient `-0.002070`, |coef| `0.002070`
- `lag_00__T_damage_last_5s`: coefficient `-0.002038`, |coef| `0.002038`
- `lag_08__CT_place_BDOORS`: coefficient `-0.001956`, |coef| `0.001956`
- `lag_09__T_duck_amount_mean`: coefficient `-0.001937`, |coef| `0.001937`
- `lag_03__T_place_PIT`: coefficient `-0.001908`, |coef| `0.001908`
- `lag_09__T2__shots_fired`: coefficient `-0.001899`, |coef| `0.001899`
- `lag_10__T_place_PIT`: coefficient `0.001827`, |coef| `0.001827`
- `lag_07__T_shots_fired_sum`: coefficient `-0.001754`, |coef| `0.001754`
- `lag_13__T_place_LONGA`: coefficient `-0.001686`, |coef| `0.001686`

## Top 10 utility ridge features

- `lag_13__T2__flash_duration`: coefficient `0.001270` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.000900` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.000788` (raises CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `0.000700` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `-0.000658` (lowers CT win probability)
- `lag_15__CT1__molly`: coefficient `0.000650` (raises CT win probability)
- `lag_01__CT_active_smokes`: coefficient `-0.000640` (lowers CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `-0.000637` (lowers CT win probability)
- `lag_11__T4__flash_duration`: coefficient `0.000625` (raises CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.000557` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003659` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003283` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `-0.003053` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002430` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002150` (lowers CT win probability)
- `lag_00__T_place_LONGA`: coefficient `-0.002088` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `-0.002070` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002038` (lowers CT win probability)
- `lag_08__CT_place_BDOORS`: coefficient `-0.001956` (lowers CT win probability)
- `lag_09__T_duck_amount_mean`: coefficient `-0.001937` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `32858`, seconds `71.00`, LSTM delta `-0.2794`

Top all feature movements:
- `lag_13__T_place_LONGA`: contribution `-0.014365`
- `lag_10__T_place_PIT`: contribution `-0.011532`
- `lag_13__T_place_PIT`: contribution `-0.009385`
- `lag_00__damage_diff_last_5s`: contribution `-0.009035`
- `lag_00__kill_diff_last_3s`: contribution `-0.008808`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `33178`, seconds `76.00`, LSTM delta `+0.2547`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `+0.012590`
- `lag_09__T_duck_amount_mean`: contribution `+0.011087`
- `lag_08__CT_place_BDOORS`: contribution `+0.009411`
- `lag_09__T_shots_fired_sum`: contribution `+0.009155`
- `lag_00__T_place_LONGA`: contribution `+0.008897`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32506`, seconds `65.50`, LSTM delta `+0.2518`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.034139`
- `lag_08__T2__shots_fired`: contribution `+0.018902`
- `lag_07__T_place_PIT`: contribution `+0.014787`
- `lag_09__T_shots_fired_sum`: contribution `+0.011444`
- `lag_11__T_shots_fired_sum`: contribution `+0.011127`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32538`, seconds `66.00`, LSTM delta `+0.1933`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.050354`
- `lag_09__T2__shots_fired`: contribution `+0.024586`
- `lag_08__T_place_PIT`: contribution `+0.015140`
- `lag_03__T_place_PIT`: contribution `+0.012039`
- `lag_11__T_shots_fired_sum`: contribution `+0.011127`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32378`, seconds `63.50`, LSTM delta `-0.1810`

Top all feature movements:
- `lag_03__T_place_PIT`: contribution `-0.024078`
- `lag_07__T_shots_fired_sum`: contribution `-0.013151`
- `lag_04__T_shots_fired_sum`: contribution `-0.011799`
- `lag_00__kill_diff_last_3s`: contribution `-0.008808`
- `lag_06__T_shots_fired_sum`: contribution `-0.008261`

Top utility-only movements:
- No utility movement among the top local contributors.
