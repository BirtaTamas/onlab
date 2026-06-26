# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `85982`, seconds `72.00`, LSTM `0.0996`, delta `-0.3977`
- tick `85662`, seconds `67.00`, LSTM `0.0942`, delta `-0.3160`
- tick `85854`, seconds `70.00`, LSTM `0.3264`, delta `+0.2838`
- tick `85950`, seconds `71.50`, LSTM `0.4973`, delta `+0.2256`
- tick `83710`, seconds `36.50`, LSTM `0.5506`, delta `-0.1882`
- tick `83518`, seconds `33.50`, LSTM `0.5801`, delta `-0.1714`
- tick `83486`, seconds `33.00`, LSTM `0.7516`, delta `+0.1144`
- tick `83646`, seconds `35.50`, LSTM `0.6975`, delta `+0.1001`
- tick `83934`, seconds `40.00`, LSTM `0.4725`, delta `-0.0984`
- tick `83838`, seconds `38.50`, LSTM `0.4844`, delta `-0.0936`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005336`, |coef| `0.005336`
- `lag_00__T_kills_last_3s`: coefficient `-0.004983`, |coef| `0.004983`
- `lag_00__CT_defusing_count`: coefficient `0.004550`, |coef| `0.004550`
- `lag_06__CT3__is_scoped`: coefficient `-0.003470`, |coef| `0.003470`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003166`, |coef| `0.003166`
- `lag_00__damage_diff_last_5s`: coefficient `0.002968`, |coef| `0.002968`
- `lag_06__T_shots_fired_sum`: coefficient `0.002809`, |coef| `0.002809`
- `lag_06__CT_A_site_active_infernos`: coefficient `0.002624`, |coef| `0.002624`
- `lag_09__T5__duck_amount`: coefficient `0.002580`, |coef| `0.002580`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002574`, |coef| `0.002574`
- `lag_10__CT4__duck_amount`: coefficient `-0.002431`, |coef| `0.002431`
- `lag_04__T5__duck_amount`: coefficient `-0.002302`, |coef| `0.002302`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.002251`, |coef| `0.002251`
- `lag_03__CT4__duck_amount`: coefficient `0.002189`, |coef| `0.002189`
- `lag_00__CT4__duck_amount`: coefficient `0.002159`, |coef| `0.002159`

## Top 10 utility ridge features

- `lag_06__CT_A_site_active_infernos`: coefficient `0.002624` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.002010` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.001703` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.001602` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001511` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.001480` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001407` (raises CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `-0.001205` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.001082` (lowers CT win probability)
- `lag_04__T1__flash`: coefficient `0.000928` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005336` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004983` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004550` (raises CT win probability)
- `lag_06__CT3__is_scoped`: coefficient `-0.003470` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003166` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002968` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.002809` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.002580` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002574` (raises CT win probability)
- `lag_10__CT4__duck_amount`: coefficient `-0.002431` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `85982`, seconds `72.00`, LSTM delta `-0.3977`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.044109`
- `lag_01__CT_defusing_count`: contribution `-0.019387`
- `lag_00__T_kills_last_3s`: contribution `-0.015788`
- `lag_00__CT4__flash_duration`: contribution `-0.014353`
- `lag_00__kill_diff_last_3s`: contribution `-0.012843`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.014353`
- `lag_00__CT_flash_duration_sum`: contribution `-0.004826`

### tick `85662`, seconds `67.00`, LSTM delta `-0.3160`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.015788`
- `lag_06__CT3__is_scoped`: contribution `-0.015784`
- `lag_00__kill_diff_last_3s`: contribution `-0.012843`
- `lag_09__T5__duck_amount`: contribution `-0.009796`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.009262`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `-0.009262`
- `lag_00__CT5__flash_duration`: contribution `-0.003876`

### tick `85854`, seconds `70.00`, LSTM delta `+0.2838`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.025686`
- `lag_00__T_kills_last_3s`: contribution `+0.015788`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010996`
- `lag_06__T_shots_fired_sum`: contribution `+0.010530`
- `lag_10__CT4__duck_amount`: contribution `+0.008927`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `85950`, seconds `71.50`, LSTM delta `+0.2256`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.044109`
- `lag_09__T5__duck_amount`: contribution `+0.009796`
- `lag_03__CT4__duck_amount`: contribution `+0.008040`
- `lag_00__CT_duck_amount_mean`: contribution `+0.006831`
- `lag_09__T_shots_fired_sum`: contribution `+0.006627`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `+0.003097`

### tick `83710`, seconds `36.50`, LSTM delta `-0.1882`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.013195`
- `lag_06__T_shots_fired_sum`: contribution `-0.012636`
- `lag_08__T_burning_players`: contribution `-0.006332`
- `lag_07__T_shots_fired_sum`: contribution `-0.004728`
- `lag_06__CT_shots_fired_sum`: contribution `-0.003891`

Top utility-only movements:
- No utility movement among the top local contributors.
