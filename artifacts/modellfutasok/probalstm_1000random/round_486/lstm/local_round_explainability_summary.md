# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `176395`, seconds `93.50`, LSTM `0.5527`, delta `+0.4683`
- tick `176651`, seconds `97.50`, LSTM `0.8466`, delta `+0.2976`
- tick `176267`, seconds `91.50`, LSTM `0.1466`, delta `-0.2889`
- tick `176555`, seconds `96.00`, LSTM `0.6047`, delta `+0.2205`
- tick `176587`, seconds `96.50`, LSTM `0.3930`, delta `-0.2117`
- tick `171979`, seconds `24.50`, LSTM `0.3611`, delta `+0.2050`
- tick `170987`, seconds `9.00`, LSTM `0.2297`, delta `-0.2001`
- tick `173451`, seconds `47.50`, LSTM `0.3010`, delta `+0.1943`
- tick `172651`, seconds `35.00`, LSTM `0.5270`, delta `-0.1838`
- tick `176619`, seconds `97.00`, LSTM `0.5490`, delta `+0.1560`

## Top 15 local ridge features

- `lag_15__CT_place_HOLE`: coefficient `-0.005969`, |coef| `0.005969`
- `lag_00__kill_diff_last_3s`: coefficient `0.005111`, |coef| `0.005111`
- `lag_00__T_place_BDOORS`: coefficient `-0.004731`, |coef| `0.004731`
- `lag_00__CT4__duck_amount`: coefficient `0.003837`, |coef| `0.003837`
- `lag_00__CT_kills_last_3s`: coefficient `0.003781`, |coef| `0.003781`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003434`, |coef| `0.003434`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003305`, |coef| `0.003305`
- `lag_00__CT_defusing_count`: coefficient `0.003059`, |coef| `0.003059`
- `lag_00__damage_diff_last_5s`: coefficient `0.002918`, |coef| `0.002918`
- `lag_00__CT_damage_last_5s`: coefficient `0.002901`, |coef| `0.002901`
- `lag_14__CT_place_HOLE`: coefficient `-0.002631`, |coef| `0.002631`
- `lag_00__T_kills_last_3s`: coefficient `-0.002578`, |coef| `0.002578`
- `lag_00__T_place_PIT`: coefficient `-0.002488`, |coef| `0.002488`
- `lag_06__T_place_BDOORS`: coefficient `0.002454`, |coef| `0.002454`
- `lag_00__CT_place_BDOORS`: coefficient `0.002409`, |coef| `0.002409`

## Top 10 utility ridge features

- `lag_06__CT_flashes_last_5s`: coefficient `0.001900` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.001877` (raises CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `-0.001851` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001726` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.001670` (raises CT win probability)
- `lag_15__T_flashes_last_5s`: coefficient `-0.001557` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.001549` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001505` (raises CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.001343` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001287` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_HOLE`: coefficient `-0.005969` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005111` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.004731` (lowers CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.003837` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003781` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003434` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003305` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003059` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002918` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002901` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `176395`, seconds `93.50`, LSTM delta `+0.4683`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `+0.066638`
- `lag_11__CT_place_HOLE`: contribution `+0.019412`
- `lag_03__T_duck_amount_mean`: contribution `+0.013422`
- `lag_00__CT_duck_amount_mean`: contribution `+0.012491`
- `lag_00__kill_diff_last_3s`: contribution `+0.012302`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `176651`, seconds `97.50`, LSTM delta `+0.2976`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.059171`
- `lag_06__T_place_BDOORS`: contribution `+0.030696`
- `lag_03__CT_defusing_count`: contribution `+0.020664`
- `lag_00__kill_diff_last_3s`: contribution `+0.012302`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011927`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010470`

### tick `176267`, seconds `91.50`, LSTM delta `-0.2889`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `-0.066638`
- `lag_11__CT_place_HOLE`: contribution `+0.019412`
- `lag_07__CT_place_HOLE`: contribution `-0.013948`
- `lag_00__kill_diff_last_3s`: contribution `-0.012302`
- `lag_00__CT_place_BDOORS`: contribution `-0.011587`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `176555`, seconds `96.00`, LSTM delta `+0.2205`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.029658`
- `lag_03__T_place_BDOORS`: contribution `+0.019050`
- `lag_08__T_duck_amount_mean`: contribution `+0.006546`
- `lag_05__CT_duck_amount_mean`: contribution `+0.005538`
- `lag_04__CT_duck_amount_mean`: contribution `+0.004844`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `176587`, seconds `96.50`, LSTM delta `-0.2117`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.029658`
- `lag_01__CT_defusing_count`: contribution `-0.022174`
- `lag_04__T_place_BDOORS`: contribution `-0.021394`
- `lag_00__kill_diff_last_3s`: contribution `-0.012302`
- `lag_00__CT_kills_last_3s`: contribution `-0.010917`

Top utility-only movements:
- No utility movement among the top local contributors.
