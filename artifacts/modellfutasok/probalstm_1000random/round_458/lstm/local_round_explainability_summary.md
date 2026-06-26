# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `85556`, seconds `100.00`, LSTM `0.4160`, delta `-0.2601`
- tick `85524`, seconds `99.50`, LSTM `0.6761`, delta `-0.2190`
- tick `85268`, seconds `95.50`, LSTM `0.8241`, delta `+0.2072`
- tick `83828`, seconds `73.00`, LSTM `0.7442`, delta `-0.1471`
- tick `83732`, seconds `71.50`, LSTM `0.8311`, delta `+0.1451`
- tick `83572`, seconds `69.00`, LSTM `0.6433`, delta `-0.1383`
- tick `80788`, seconds `25.50`, LSTM `0.7721`, delta `+0.1180`
- tick `84340`, seconds `81.00`, LSTM `0.7905`, delta `+0.1119`
- tick `84372`, seconds `81.50`, LSTM `0.7207`, delta `-0.0698`
- tick `85780`, seconds `103.50`, LSTM `0.3883`, delta `-0.0580`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005552`, |coef| `0.005552`
- `lag_00__CT_kills_last_3s`: coefficient `0.003649`, |coef| `0.003649`
- `lag_00__T_kills_last_3s`: coefficient `-0.003304`, |coef| `0.003304`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003239`, |coef| `0.003239`
- `lag_01__kill_diff_last_3s`: coefficient `0.003005`, |coef| `0.003005`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002845`, |coef| `0.002845`
- `lag_00__damage_diff_last_5s`: coefficient `0.002741`, |coef| `0.002741`
- `lag_00__T_damage_last_5s`: coefficient `-0.002507`, |coef| `0.002507`
- `lag_06__CT_place_BANANA`: coefficient `0.002502`, |coef| `0.002502`
- `lag_01__CT_kills_last_3s`: coefficient `0.002404`, |coef| `0.002404`
- `lag_14__T1__is_walking`: coefficient `0.002281`, |coef| `0.002281`
- `lag_08__CT4__is_scoped`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_01__T_duck_amount_mean`: coefficient `-0.002142`, |coef| `0.002142`
- `lag_07__CT_place_BANANA`: coefficient `0.001926`, |coef| `0.001926`
- `lag_11__T_velocity_mean`: coefficient `-0.001832`, |coef| `0.001832`

## Top 10 utility ridge features

- `lag_13__T_B_site_active_smokes`: coefficient `-0.001321` (lowers CT win probability)
- `lag_04__CT4__smoke`: coefficient `-0.001254` (lowers CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.001165` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `-0.001160` (lowers CT win probability)
- `lag_12__CT4__smoke`: coefficient `0.001022` (raises CT win probability)
- `lag_13__CT4__smoke`: coefficient `0.000996` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000963` (lowers CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `-0.000955` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.000925` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.000875` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005552` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003649` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003304` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003239` (lowers CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.003005` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002845` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002741` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002507` (lowers CT win probability)
- `lag_06__CT_place_BANANA`: coefficient `0.002502` (raises CT win probability)
- `lag_01__CT_kills_last_3s`: coefficient `0.002404` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `85556`, seconds `100.00`, LSTM delta `-0.2601`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.013364`
- `lag_01__T_duck_amount_mean`: contribution `-0.012457`
- `lag_00__T_shots_fired_sum`: contribution `-0.012141`
- `lag_00__T_kills_last_3s`: contribution `-0.010467`
- `lag_08__CT4__is_scoped`: contribution `-0.007729`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85524`, seconds `99.50`, LSTM delta `-0.2190`

Top all feature movements:
- `lag_00__T_duck_amount_mean`: contribution `-0.016547`
- `lag_00__kill_diff_last_3s`: contribution `-0.013364`
- `lag_00__T_shots_fired_sum`: contribution `-0.012141`
- `lag_00__T_kills_last_3s`: contribution `-0.010467`
- `lag_06__CT_place_BANANA`: contribution `-0.007408`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85268`, seconds `95.50`, LSTM delta `+0.2072`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013364`
- `lag_00__CT_kills_last_3s`: contribution `+0.010536`
- `lag_08__CT4__is_scoped`: contribution `+0.007729`
- `lag_06__CT_place_BANANA`: contribution `+0.007408`
- `lag_14__T1__is_walking`: contribution `+0.005205`

Top utility-only movements:
- `lag_04__CT4__smoke`: contribution `+0.002736`

### tick `83828`, seconds `73.00`, LSTM delta `-0.1471`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.016665`
- `lag_00__kill_diff_last_3s`: contribution `-0.013364`
- `lag_00__T_kills_last_3s`: contribution `-0.010467`
- `lag_06__T3__shots_fired`: contribution `-0.010374`
- `lag_01__T_shots_fired_sum`: contribution `+0.008994`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.002975`

### tick `83732`, seconds `71.50`, LSTM delta `+0.1451`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `+0.014632`
- `lag_00__kill_diff_last_3s`: contribution `+0.013364`
- `lag_00__CT_place_QUAD`: contribution `+0.013082`
- `lag_00__CT_kills_last_3s`: contribution `+0.010536`
- `lag_00__T_shots_fired_sum`: contribution `-0.007284`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `+0.006320`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.002735`
