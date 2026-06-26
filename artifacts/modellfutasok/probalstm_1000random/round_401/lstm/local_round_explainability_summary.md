# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `1`

## Largest probability jumps

- tick `11856`, seconds `35.00`, LSTM `0.7844`, delta `+0.2175`
- tick `12016`, seconds `37.50`, LSTM `0.8698`, delta `+0.1521`
- tick `14832`, seconds `81.50`, LSTM `0.8529`, delta `+0.1428`
- tick `11760`, seconds `33.50`, LSTM `0.5488`, delta `+0.0761`
- tick `11920`, seconds `36.00`, LSTM `0.7440`, delta `-0.0718`
- tick `13872`, seconds `66.50`, LSTM `0.8415`, delta `-0.0644`
- tick `14768`, seconds `80.50`, LSTM `0.7152`, delta `-0.0393`
- tick `14864`, seconds `82.00`, LSTM `0.8894`, delta `+0.0366`
- tick `12272`, seconds `41.50`, LSTM `0.9098`, delta `+0.0365`
- tick `14320`, seconds `73.50`, LSTM `0.7790`, delta `-0.0358`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004383`, |coef| `0.004383`
- `lag_00__CT_kills_last_3s`: coefficient `0.003811`, |coef| `0.003811`
- `lag_00__damage_diff_last_5s`: coefficient `0.003704`, |coef| `0.003704`
- `lag_00__CT_damage_last_5s`: coefficient `0.002940`, |coef| `0.002940`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_00__T_macro_B`: coefficient `-0.002643`, |coef| `0.002643`
- `lag_00__CT4__duck_amount`: coefficient `0.002631`, |coef| `0.002631`
- `lag_09__CT_place_PIT`: coefficient `0.002053`, |coef| `0.002053`
- `lag_07__T4__duck_amount`: coefficient `-0.001967`, |coef| `0.001967`
- `lag_07__T_duck_amount_mean`: coefficient `-0.001901`, |coef| `0.001901`
- `lag_10__T1__is_walking`: coefficient `-0.001864`, |coef| `0.001864`
- `lag_02__T_place_BANANA`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_01__CT4__is_walking`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_00__alive_diff`: coefficient `0.001651`, |coef| `0.001651`
- `lag_02__CT4__duck_amount`: coefficient `-0.001619`, |coef| `0.001619`

## Top 10 utility ridge features

- `lag_04__T5__flash_duration`: coefficient `0.001393` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.001251` (raises CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `0.001098` (raises CT win probability)
- `lag_04__T1__molly`: coefficient `-0.001071` (lowers CT win probability)
- `lag_04__T3__smoke`: coefficient `-0.001054` (lowers CT win probability)
- `lag_07__CT3__smoke`: coefficient `0.000990` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.000953` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.000931` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `0.000922` (raises CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `0.000915` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004383` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003811` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003704` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002940` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002643` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002643` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.002631` (raises CT win probability)
- `lag_09__CT_place_PIT`: coefficient `0.002053` (raises CT win probability)
- `lag_07__T4__duck_amount`: coefficient `-0.001967` (lowers CT win probability)
- `lag_07__T_duck_amount_mean`: coefficient `-0.001901` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `11856`, seconds `35.00`, LSTM delta `+0.2175`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011004`
- `lag_00__kill_diff_last_3s`: contribution `+0.010549`
- `lag_00__CT4__duck_amount`: contribution `+0.009664`
- `lag_09__CT_place_PIT`: contribution `+0.008839`
- `lag_00__damage_diff_last_5s`: contribution `+0.007354`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `+0.003538`

### tick `12016`, seconds `37.50`, LSTM delta `+0.1521`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011004`
- `lag_00__kill_diff_last_3s`: contribution `+0.010549`
- `lag_04__T5__flash_duration`: contribution `+0.009082`
- `lag_00__damage_diff_last_5s`: contribution `+0.007688`
- `lag_00__CT_damage_last_5s`: contribution `+0.006408`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `+0.009082`
- `lag_04__T3__flash_duration`: contribution `+0.005561`
- `lag_04__CT3__flash_duration`: contribution `+0.004946`
- `lag_04__CT_flash_duration_sum`: contribution `+0.004512`
- `lag_01__CT4__flash_duration`: contribution `+0.004285`

### tick `14832`, seconds `81.50`, LSTM delta `+0.1428`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011004`
- `lag_00__kill_diff_last_3s`: contribution `+0.010549`
- `lag_00__damage_diff_last_5s`: contribution `+0.008357`
- `lag_07__T4__duck_amount`: contribution `+0.007272`
- `lag_00__CT_damage_last_5s`: contribution `+0.006408`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11760`, seconds `33.50`, LSTM delta `+0.0761`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011004`
- `lag_00__kill_diff_last_3s`: contribution `+0.010549`
- `lag_00__damage_diff_last_5s`: contribution `+0.006853`
- `lag_00__CT_damage_last_5s`: contribution `+0.005255`
- `lag_00__T_place_BOMBSITEB`: contribution `+0.004125`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11920`, seconds `36.00`, LSTM delta `-0.0718`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.008524`
- `lag_02__CT4__duck_amount`: contribution `-0.005948`
- `lag_01__T3__flash_duration`: contribution `-0.004398`
- `lag_01__CT3__flash_duration`: contribution `-0.004186`
- `lag_01__CT5__flash_duration`: contribution `-0.003362`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.004398`
- `lag_01__CT3__flash_duration`: contribution `-0.004186`
- `lag_01__CT5__flash_duration`: contribution `-0.003362`
- `lag_01__CT4__flash_duration`: contribution `+0.002289`
