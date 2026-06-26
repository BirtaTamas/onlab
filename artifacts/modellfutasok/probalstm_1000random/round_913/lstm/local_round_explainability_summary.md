# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-nrg-bo3-GH6ZBFOA9sfdeCxgnhHN9f/og-vs-nrg-m2-nuke.csv`
- round_num: `11`

## Largest probability jumps

- tick `89210`, seconds `130.00`, LSTM `0.4118`, delta `+0.3471`
- tick `89562`, seconds `135.50`, LSTM `0.2820`, delta `-0.2873`
- tick `87066`, seconds `96.50`, LSTM `0.3195`, delta `+0.2200`
- tick `88026`, seconds `111.50`, LSTM `0.4179`, delta `+0.2051`
- tick `89338`, seconds `132.00`, LSTM `0.6822`, delta `+0.2040`
- tick `84154`, seconds `51.00`, LSTM `0.2210`, delta `-0.1734`
- tick `88282`, seconds `115.50`, LSTM `0.2601`, delta `-0.1338`
- tick `87930`, seconds `110.00`, LSTM `0.2481`, delta `+0.1286`
- tick `89370`, seconds `132.50`, LSTM `0.5674`, delta `-0.1148`
- tick `87258`, seconds `99.50`, LSTM `0.2875`, delta `-0.0962`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005517`, |coef| `0.005517`
- `lag_05__T_place_OBSERVATION`: coefficient `-0.005021`, |coef| `0.005021`
- `lag_00__CT_defusing_count`: coefficient `0.004466`, |coef| `0.004466`
- `lag_00__damage_diff_last_5s`: coefficient `0.004435`, |coef| `0.004435`
- `lag_01__T_place_DECON`: coefficient `0.004193`, |coef| `0.004193`
- `lag_00__CT_place_GARAGE`: coefficient `0.003593`, |coef| `0.003593`
- `lag_00__T_kills_last_3s`: coefficient `-0.003585`, |coef| `0.003585`
- `lag_08__CT_place_VENTS`: coefficient `-0.003467`, |coef| `0.003467`
- `lag_00__CT_kills_last_3s`: coefficient `0.003351`, |coef| `0.003351`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003342`, |coef| `0.003342`
- `lag_09__T_place_OBSERVATION`: coefficient `-0.003000`, |coef| `0.003000`
- `lag_07__CT_defusing_count`: coefficient `-0.003000`, |coef| `0.003000`
- `lag_06__CT_defusing_count`: coefficient `0.002918`, |coef| `0.002918`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.002882`, |coef| `0.002882`
- `lag_08__CT_place_HUT`: coefficient `-0.002689`, |coef| `0.002689`

## Top 10 utility ridge features

- `lag_00__T2__flash`: coefficient `-0.002310` (lowers CT win probability)
- `lag_12__CT5__smoke`: coefficient `-0.001553` (lowers CT win probability)
- `lag_11__T2__flash`: coefficient `0.001259` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001064` (lowers CT win probability)
- `lag_03__CT4__smoke`: coefficient `0.000924` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000917` (lowers CT win probability)
- `lag_02__CT_smokes_last_5s`: coefficient `0.000893` (raises CT win probability)
- `lag_04__T2__flash`: coefficient `-0.000890` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000883` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000813` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005517` (raises CT win probability)
- `lag_05__T_place_OBSERVATION`: coefficient `-0.005021` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004466` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004435` (raises CT win probability)
- `lag_01__T_place_DECON`: coefficient `0.004193` (raises CT win probability)
- `lag_00__CT_place_GARAGE`: coefficient `0.003593` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003585` (lowers CT win probability)
- `lag_08__CT_place_VENTS`: coefficient `-0.003467` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003351` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003342` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `89210`, seconds `130.00`, LSTM delta `+0.3471`

Top all feature movements:
- `lag_05__T_place_OBSERVATION`: contribution `+0.085030`
- `lag_08__CT_place_VENTS`: contribution `+0.029088`
- `lag_00__kill_diff_last_3s`: contribution `+0.013279`
- `lag_00__CT_kills_last_3s`: contribution `+0.009674`
- `lag_11__T2__shots_fired`: contribution `+0.009492`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.006802`

### tick `89562`, seconds `135.50`, LSTM delta `-0.2873`

Top all feature movements:
- `lag_01__T_place_DECON`: contribution `-0.067362`
- `lag_07__CT_defusing_count`: contribution `-0.029077`
- `lag_06__CT_defusing_count`: contribution `-0.028287`
- `lag_00__kill_diff_last_3s`: contribution `-0.013279`
- `lag_00__T_kills_last_3s`: contribution `-0.011358`

Top utility-only movements:
- `lag_11__T2__flash`: contribution `-0.003705`

### tick `87066`, seconds `96.50`, LSTM delta `+0.2200`

Top all feature movements:
- `lag_08__CT_place_HUT`: contribution `+0.026224`
- `lag_08__CT_place_LOBBY`: contribution `+0.019870`
- `lag_00__kill_diff_last_3s`: contribution `+0.013279`
- `lag_05__CT_shots_fired_sum`: contribution `+0.012888`
- `lag_01__CT_place_MINI`: contribution `+0.012105`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88026`, seconds `111.50`, LSTM delta `+0.2051`

Top all feature movements:
- `lag_02__CT_place_SECRET`: contribution `+0.021821`
- `lag_06__CT_place_VENTS`: contribution `+0.020772`
- `lag_00__T_shots_fired_sum`: contribution `+0.016529`
- `lag_07__CT_place_SECRET`: contribution `+0.014941`
- `lag_00__kill_diff_last_3s`: contribution `+0.013279`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89338`, seconds `132.00`, LSTM delta `+0.2040`

Top all feature movements:
- `lag_09__T_place_OBSERVATION`: contribution `+0.050804`
- `lag_00__CT_defusing_count`: contribution `+0.043290`
- `lag_12__CT_place_VENTS`: contribution `+0.011147`
- `lag_03__CT_place_TUNNELS`: contribution `+0.006628`
- `lag_04__CT4__duck_amount`: contribution `+0.006366`

Top utility-only movements:
- `lag_04__T2__flash`: contribution `+0.002621`
