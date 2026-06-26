# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `15451`, seconds `51.00`, LSTM `0.4245`, delta `+0.2172`
- tick `15419`, seconds `50.50`, LSTM `0.2073`, delta `-0.1813`
- tick `16955`, seconds `74.50`, LSTM `0.6698`, delta `-0.1481`
- tick `15547`, seconds `52.50`, LSTM `0.6615`, delta `+0.1415`
- tick `15739`, seconds `55.50`, LSTM `0.8660`, delta `+0.1412`
- tick `16347`, seconds `65.00`, LSTM `0.8495`, delta `-0.1097`
- tick `15483`, seconds `51.50`, LSTM `0.5210`, delta `+0.0965`
- tick `15195`, seconds `47.00`, LSTM `0.3377`, delta `-0.0902`
- tick `15835`, seconds `57.00`, LSTM `0.9449`, delta `+0.0590`
- tick `16475`, seconds `67.00`, LSTM `0.8445`, delta `+0.0452`

## Top 15 local ridge features

- `lag_01__T_place_BDOORS`: coefficient `0.003887`, |coef| `0.003887`
- `lag_00__T_place_BDOORS`: coefficient `-0.002980`, |coef| `0.002980`
- `lag_04__T_place_BDOORS`: coefficient `0.002272`, |coef| `0.002272`
- `lag_01__CT_place_ARAMP`: coefficient `-0.001820`, |coef| `0.001820`
- `lag_00__kill_diff_last_3s`: coefficient `0.001807`, |coef| `0.001807`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001799`, |coef| `0.001799`
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.001720`, |coef| `0.001720`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001688`, |coef| `0.001688`
- `lag_00__CT3__duck_amount`: coefficient `-0.001670`, |coef| `0.001670`
- `lag_01__CT_place_UPPERTUNNEL`: coefficient `0.001635`, |coef| `0.001635`
- `lag_10__T_place_BDOORS`: coefficient `0.001612`, |coef| `0.001612`
- `lag_05__CT_place_ARAMP`: coefficient `0.001536`, |coef| `0.001536`
- `lag_12__T_place_MIDDOORS`: coefficient `-0.001501`, |coef| `0.001501`
- `lag_05__T_place_MIDDOORS`: coefficient `-0.001410`, |coef| `0.001410`
- `lag_07__CT_place_ARAMP`: coefficient `-0.001402`, |coef| `0.001402`

## Top 10 utility ridge features

- `lag_12__CT4__flash_duration`: coefficient `0.001107` (raises CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.000784` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.000727` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.000709` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.000658` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `0.000632` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000599` (raises CT win probability)
- `lag_06__CT2__smoke`: coefficient `-0.000559` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `0.000552` (raises CT win probability)
- `lag_04__CT_active_smokes`: coefficient `0.000548` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_BDOORS`: coefficient `0.003887` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.002980` (lowers CT win probability)
- `lag_04__T_place_BDOORS`: coefficient `0.002272` (raises CT win probability)
- `lag_01__CT_place_ARAMP`: coefficient `-0.001820` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001807` (raises CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001799` (lowers CT win probability)
- `lag_02__CT_place_UPPERTUNNEL`: coefficient `0.001720` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.001688` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `-0.001670` (lowers CT win probability)
- `lag_01__CT_place_UPPERTUNNEL`: coefficient `0.001635` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `15451`, seconds `51.00`, LSTM delta `+0.2172`

Top all feature movements:
- `lag_01__T_place_BDOORS`: contribution `+0.097238`
- `lag_01__T_place_MIDDOORS`: contribution `+0.015290`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `+0.012537`
- `lag_10__CT_place_BDOORS`: contribution `+0.006265`
- `lag_00__CT3__duck_amount`: contribution `+0.004364`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15419`, seconds `50.50`, LSTM delta `-0.1813`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.074542`
- `lag_05__CT_place_ARAMP`: contribution `-0.009569`
- `lag_07__CT_place_ARAMP`: contribution `-0.008733`
- `lag_12__T_place_MIDDOORS`: contribution `-0.006381`
- `lag_00__CT3__duck_amount`: contribution `-0.006214`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16955`, seconds `74.50`, LSTM delta `-0.1481`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `-0.015439`
- `lag_06__CT_place_HOLE`: contribution `-0.009762`
- `lag_01__T_duck_amount_mean`: contribution `-0.006911`
- `lag_10__CT_place_BDOORS`: contribution `-0.006265`
- `lag_02__T_duck_amount_mean`: contribution `-0.006170`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15547`, seconds `52.50`, LSTM delta `+0.1415`

Top all feature movements:
- `lag_04__T_place_BDOORS`: contribution `+0.056831`
- `lag_04__T_place_MIDDOORS`: contribution `+0.010744`
- `lag_00__CT_place_ARAMP`: contribution `-0.010516`
- `lag_04__CT_place_UPPERTUNNEL`: contribution `+0.009520`
- `lag_00__kill_diff_last_3s`: contribution `+0.004349`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.004120`
- `lag_02__T_flash_duration_sum`: contribution `+0.001676`

### tick `15739`, seconds `55.50`, LSTM delta `+0.1412`

Top all feature movements:
- `lag_10__T_place_BDOORS`: contribution `+0.040332`
- `lag_01__CT_place_ARAMP`: contribution `+0.011338`
- `lag_07__CT_place_HOLE`: contribution `+0.011285`
- `lag_10__T_place_MIDDOORS`: contribution `+0.011000`
- `lag_02__T_place_BDOORS`: contribution `-0.009263`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.004634`
- `lag_08__T1__flash_duration`: contribution `+0.003823`
