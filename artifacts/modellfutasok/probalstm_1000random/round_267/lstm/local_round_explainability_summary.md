# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `9`

## Largest probability jumps

- tick `89174`, seconds `92.50`, LSTM `0.1825`, delta `-0.1342`
- tick `89206`, seconds `93.00`, LSTM `0.0506`, delta `-0.1319`
- tick `88918`, seconds `88.50`, LSTM `0.2606`, delta `-0.0727`
- tick `89110`, seconds `91.50`, LSTM `0.3639`, delta `+0.0548`
- tick `89142`, seconds `92.00`, LSTM `0.3167`, delta `-0.0472`
- tick `89046`, seconds `90.50`, LSTM `0.2889`, delta `+0.0435`
- tick `88470`, seconds `81.50`, LSTM `0.3425`, delta `+0.0345`
- tick `88438`, seconds `81.00`, LSTM `0.3080`, delta `-0.0344`
- tick `85366`, seconds `33.00`, LSTM `0.4117`, delta `+0.0321`
- tick `88694`, seconds `85.00`, LSTM `0.2937`, delta `+0.0313`

## Top 15 local ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002192`, |coef| `0.002192`
- `lag_01__T_place_HOLE`: coefficient `-0.001913`, |coef| `0.001913`
- `lag_00__T_place_BDOORS`: coefficient `-0.001299`, |coef| `0.001299`
- `lag_00__CT_place_ARAMP`: coefficient `-0.000998`, |coef| `0.000998`
- `lag_08__T_place_MIDDOORS`: coefficient `0.000972`, |coef| `0.000972`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000943`, |coef| `0.000943`
- `lag_00__CT5__duck_amount`: coefficient `-0.000883`, |coef| `0.000883`
- `lag_09__T_place_MIDDOORS`: coefficient `0.000762`, |coef| `0.000762`
- `lag_14__T_place_BDOORS`: coefficient `-0.000716`, |coef| `0.000716`
- `lag_11__T5__duck_amount`: coefficient `0.000691`, |coef| `0.000691`
- `lag_10__T5__duck_amount`: coefficient `0.000676`, |coef| `0.000676`
- `lag_15__T5__is_scoped`: coefficient `-0.000673`, |coef| `0.000673`
- `lag_02__T_place_HOLE`: coefficient `-0.000666`, |coef| `0.000666`
- `lag_02__T3__duck_amount`: coefficient `-0.000665`, |coef| `0.000665`
- `lag_00__T3__duck_amount`: coefficient `0.000653`, |coef| `0.000653`

## Top 10 utility ridge features

- `lag_15__T_A_site_active_infernos`: coefficient `0.000557` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000555` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000542` (raises CT win probability)
- `lag_08__CT2__smoke`: coefficient `0.000498` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000487` (raises CT win probability)
- `lag_00__T_flash_inv`: coefficient `0.000460` (raises CT win probability)
- `lag_00__T4__utility_total`: coefficient `0.000444` (raises CT win probability)
- `lag_00__T_utility_inv`: coefficient `0.000430` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000418` (lowers CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000416` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002192` (lowers CT win probability)
- `lag_01__T_place_HOLE`: coefficient `-0.001913` (lowers CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001299` (lowers CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.000998` (lowers CT win probability)
- `lag_08__T_place_MIDDOORS`: coefficient `0.000972` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000943` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `-0.000883` (lowers CT win probability)
- `lag_09__T_place_MIDDOORS`: coefficient `0.000762` (raises CT win probability)
- `lag_14__T_place_BDOORS`: coefficient `-0.000716` (lowers CT win probability)
- `lag_11__T5__duck_amount`: coefficient `0.000691` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `89174`, seconds `92.50`, LSTM delta `-0.1342`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `-0.056500`
- `lag_08__T_place_BDOORS`: contribution `-0.006859`
- `lag_02__T_place_BDOORS`: contribution `-0.004501`
- `lag_08__T_place_MIDDOORS`: contribution `-0.004133`
- `lag_00__T_shots_fired_sum`: contribution `-0.003536`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `-0.001659`

### tick `89206`, seconds `93.00`, LSTM delta `-0.1319`

Top all feature movements:
- `lag_01__T_place_HOLE`: contribution `-0.049306`
- `lag_09__T_place_BDOORS`: contribution `-0.004570`
- `lag_00__T_shots_fired_sum`: contribution `-0.004243`
- `lag_06__T_place_BDOORS`: contribution `-0.003312`
- `lag_09__T_place_MIDDOORS`: contribution `-0.003239`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.003208`

### tick `88918`, seconds `88.50`, LSTM delta `-0.0727`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.016247`
- `lag_09__T_place_MIDDOORS`: contribution `+0.003239`
- `lag_04__T_place_MIDDOORS`: contribution `-0.002365`
- `lag_00__CT5__duck_amount`: contribution `+0.002052`
- `lag_12__CT4__is_scoped`: contribution `-0.001899`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `-0.001069`
- `lag_07__T_A_site_active_infernos`: contribution `-0.000965`

### tick `89110`, seconds `91.50`, LSTM delta `+0.0548`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.016247`
- `lag_06__T_place_BDOORS`: contribution `-0.003312`
- `lag_03__T_place_BDOORS`: contribution `+0.002834`
- `lag_00__T3__duck_amount`: contribution `+0.002462`
- `lag_09__CT2__duck_amount`: contribution `+0.001951`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.001274`
- `lag_01__CT1__flash_duration`: contribution `+0.001047`

### tick `89142`, seconds `92.00`, LSTM delta `-0.0472`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `-0.003762`
- `lag_00__T_shots_fired_sum`: contribution `-0.003536`
- `lag_00__CT5__duck_amount`: contribution `-0.003333`
- `lag_03__CT_place_ARAMP`: contribution `-0.003035`
- `lag_04__T_place_MIDDOORS`: contribution `+0.002365`

Top utility-only movements:
- `lag_08__CT2__smoke`: contribution `-0.001081`
