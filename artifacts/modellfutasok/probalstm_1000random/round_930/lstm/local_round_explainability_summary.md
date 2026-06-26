# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `150957`, seconds `54.00`, LSTM `0.2352`, delta `-0.2586`
- tick `152269`, seconds `74.50`, LSTM `0.0400`, delta `-0.1124`
- tick `153037`, seconds `86.50`, LSTM `0.0437`, delta `-0.0942`
- tick `150061`, seconds `40.00`, LSTM `0.4347`, delta `-0.0856`
- tick `152109`, seconds `72.00`, LSTM `0.1250`, delta `+0.0835`
- tick `150093`, seconds `40.50`, LSTM `0.3513`, delta `-0.0834`
- tick `150989`, seconds `54.50`, LSTM `0.1594`, delta `-0.0757`
- tick `152781`, seconds `82.50`, LSTM `0.0739`, delta `+0.0600`
- tick `149869`, seconds `37.00`, LSTM `0.5252`, delta `-0.0586`
- tick `151085`, seconds `56.00`, LSTM `0.0674`, delta `-0.0562`

## Top 15 local ridge features

- `lag_03__T_place_ARCH`: coefficient `0.003114`, |coef| `0.003114`
- `lag_13__T_place_ARCH`: coefficient `-0.002702`, |coef| `0.002702`
- `lag_05__T_place_LIBRARY`: coefficient `-0.002532`, |coef| `0.002532`
- `lag_00__T_kills_last_3s`: coefficient `-0.002254`, |coef| `0.002254`
- `lag_03__T_place_CTSPAWN`: coefficient `-0.002024`, |coef| `0.002024`
- `lag_00__kill_diff_last_3s`: coefficient `0.001849`, |coef| `0.001849`
- `lag_04__T_place_ARCH`: coefficient `0.001471`, |coef| `0.001471`
- `lag_00__damage_diff_last_5s`: coefficient `0.001470`, |coef| `0.001470`
- `lag_00__CT1__alive`: coefficient `0.001468`, |coef| `0.001468`
- `lag_01__damage_diff_last_5s`: coefficient `0.001462`, |coef| `0.001462`
- `lag_14__CT3__duck_amount`: coefficient `-0.001443`, |coef| `0.001443`
- `lag_15__T3__duck_amount`: coefficient `0.001431`, |coef| `0.001431`
- `lag_03__CT_place_RUINS`: coefficient `0.001416`, |coef| `0.001416`
- `lag_02__CT1__is_scoped`: coefficient `0.001366`, |coef| `0.001366`
- `lag_15__CT3__is_walking`: coefficient `0.001306`, |coef| `0.001306`

## Top 10 utility ridge features

- `lag_00__CT1__smoke`: coefficient `0.001303` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001230` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001076` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.001056` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.000963` (lowers CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000891` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000878` (raises CT win probability)
- `lag_15__CT_A_site_active_smokes`: coefficient `0.000841` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000774` (raises CT win probability)
- `lag_08__T_mollies_last_5s`: coefficient `-0.000762` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_ARCH`: coefficient `0.003114` (raises CT win probability)
- `lag_13__T_place_ARCH`: coefficient `-0.002702` (lowers CT win probability)
- `lag_05__T_place_LIBRARY`: coefficient `-0.002532` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002254` (lowers CT win probability)
- `lag_03__T_place_CTSPAWN`: coefficient `-0.002024` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001849` (raises CT win probability)
- `lag_04__T_place_ARCH`: coefficient `0.001471` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001470` (raises CT win probability)
- `lag_00__CT1__alive`: coefficient `0.001468` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001462` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `150957`, seconds `54.00`, LSTM delta `-0.2586`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `-0.028973`
- `lag_13__T_place_ARCH`: contribution `-0.025140`
- `lag_03__T_place_CTSPAWN`: contribution `-0.009656`
- `lag_00__T_kills_last_3s`: contribution `-0.007141`
- `lag_02__CT1__is_scoped`: contribution `-0.005852`

Top utility-only movements:
- `lag_00__CT1__smoke`: contribution `-0.002824`

### tick `152269`, seconds `74.50`, LSTM delta `-0.1124`

Top all feature movements:
- `lag_10__T_place_LIBRARY`: contribution `-0.010153`
- `lag_00__T_kills_last_3s`: contribution `-0.007141`
- `lag_14__CT3__duck_amount`: contribution `-0.005370`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.005302`
- `lag_04__CT_shots_fired_sum`: contribution `-0.005205`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.005302`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.002277`
- `lag_00__CT2__utility_total`: contribution `-0.001497`

### tick `153037`, seconds `86.50`, LSTM delta `-0.0942`

Top all feature movements:
- `lag_09__CT_place_BALCONY`: contribution `-0.007528`
- `lag_00__T_kills_last_3s`: contribution `-0.007141`
- `lag_10__CT5__flash_duration`: contribution `-0.005777`
- `lag_02__CT_place_BALCONY`: contribution `-0.004757`
- `lag_00__kill_diff_last_3s`: contribution `-0.004450`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `-0.005777`
- `lag_15__CT3__flash_duration`: contribution `-0.003987`
- `lag_00__CT5__flash_duration`: contribution `-0.003409`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.001543`
- `lag_10__CT3__flash_duration`: contribution `-0.001485`

### tick `150061`, seconds `40.00`, LSTM delta `-0.0856`

Top all feature movements:
- `lag_06__T_place_QUAD`: contribution `-0.012668`
- `lag_00__T_kills_last_3s`: contribution `-0.007141`
- `lag_08__T_place_QUAD`: contribution `-0.005790`
- `lag_06__T_utility_damage_last_5s`: contribution `-0.004845`
- `lag_00__kill_diff_last_3s`: contribution `-0.004450`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `-0.004845`
- `lag_08__CT_flash_duration_sum`: contribution `-0.001880`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.001649`
- `lag_08__CT4__flash_duration`: contribution `-0.001516`
- `lag_00__CT4__utility_total`: contribution `-0.001443`

### tick `152109`, seconds `72.00`, LSTM delta `+0.0835`

Top all feature movements:
- `lag_05__T_place_LIBRARY`: contribution `+0.055796`
- `lag_00__kill_diff_last_3s`: contribution `+0.004450`
- `lag_02__T_bomb_zone_count`: contribution `+0.003653`
- `lag_15__CT3__duck_amount`: contribution `-0.002611`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002069`

Top utility-only movements:
- No utility movement among the top local contributors.
