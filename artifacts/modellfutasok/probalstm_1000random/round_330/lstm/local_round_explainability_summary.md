# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `17292`, seconds `39.00`, LSTM `0.1075`, delta `-0.3119`
- tick `16844`, seconds `32.00`, LSTM `0.2075`, delta `-0.2875`
- tick `19084`, seconds `67.00`, LSTM `0.4876`, delta `+0.2699`
- tick `16812`, seconds `31.50`, LSTM `0.4950`, delta `+0.2559`
- tick `17260`, seconds `38.50`, LSTM `0.4194`, delta `+0.2109`
- tick `20012`, seconds `81.50`, LSTM `0.5763`, delta `-0.1477`
- tick `16012`, seconds `19.00`, LSTM `0.2481`, delta `+0.1264`
- tick `20044`, seconds `82.00`, LSTM `0.6942`, delta `+0.1180`
- tick `22060`, seconds `113.50`, LSTM `0.0602`, delta `-0.1030`
- tick `17164`, seconds `37.00`, LSTM `0.2225`, delta `-0.1016`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.004971`, |coef| `0.004971`
- `lag_00__kill_diff_last_3s`: coefficient `0.004519`, |coef| `0.004519`
- `lag_14__CT_place_LOBBY`: coefficient `0.004139`, |coef| `0.004139`
- `lag_02__CT_place_VENTS`: coefficient `0.003097`, |coef| `0.003097`
- `lag_00__damage_diff_last_5s`: coefficient `0.003092`, |coef| `0.003092`
- `lag_00__T_kills_last_3s`: coefficient `-0.002918`, |coef| `0.002918`
- `lag_00__T_place_DECON`: coefficient `-0.002875`, |coef| `0.002875`
- `lag_00__CT_kills_last_3s`: coefficient `0.002761`, |coef| `0.002761`
- `lag_02__CT_place_OBSERVATION`: coefficient `-0.002412`, |coef| `0.002412`
- `lag_06__CT_place_OBSERVATION`: coefficient `-0.002359`, |coef| `0.002359`
- `lag_00__T_place_CONTROL`: coefficient `-0.002355`, |coef| `0.002355`
- `lag_12__T_place_TROPHY`: coefficient `0.002348`, |coef| `0.002348`
- `lag_00__CT_velocity_mean`: coefficient `-0.002063`, |coef| `0.002063`
- `lag_09__CT_place_ROOF`: coefficient `0.002049`, |coef| `0.002049`
- `lag_09__CT_place_HELL`: coefficient `0.001979`, |coef| `0.001979`

## Top 10 utility ridge features

- `lag_09__CT2__flash_duration`: coefficient `-0.000898` (lowers CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.000888` (raises CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `0.000724` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.000643` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000605` (raises CT win probability)
- `lag_13__T_A_site_active_smokes`: coefficient `0.000589` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000559` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `0.000539` (raises CT win probability)
- `lag_02__T_A_site_active_smokes`: coefficient `0.000523` (raises CT win probability)
- `lag_15__T_active_smokes`: coefficient `0.000514` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.004971` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004519` (raises CT win probability)
- `lag_14__CT_place_LOBBY`: coefficient `0.004139` (raises CT win probability)
- `lag_02__CT_place_VENTS`: coefficient `0.003097` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003092` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002918` (lowers CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.002875` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002761` (raises CT win probability)
- `lag_02__CT_place_OBSERVATION`: coefficient `-0.002412` (lowers CT win probability)
- `lag_06__CT_place_OBSERVATION`: coefficient `-0.002359` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `17292`, seconds `39.00`, LSTM delta `-0.3119`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.055259`
- `lag_13__CT_place_SQUEAKY`: contribution `-0.017607`
- `lag_12__T_place_TROPHY`: contribution `-0.014893`
- `lag_04__CT_place_SQUEAKY`: contribution `-0.013874`
- `lag_12__T_place_CONTROL`: contribution `-0.013768`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16844`, seconds `32.00`, LSTM delta `-0.2875`

Top all feature movements:
- `lag_06__CT_place_OBSERVATION`: contribution `-0.041086`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.030864`
- `lag_00__CT_shots_fired_sum`: contribution `-0.027629`
- `lag_00__kill_diff_last_3s`: contribution `-0.010878`
- `lag_09__CT_place_HELL`: contribution `-0.010733`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19084`, seconds `67.00`, LSTM delta `+0.2699`

Top all feature movements:
- `lag_14__CT_place_LOBBY`: contribution `+0.033882`
- `lag_12__T_place_TROPHY`: contribution `+0.014893`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013815`
- `lag_12__T_place_CONTROL`: contribution `+0.013768`
- `lag_08__T_place_TROPHY`: contribution `+0.012537`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16812`, seconds `31.50`, LSTM delta `+0.2559`

Top all feature movements:
- `lag_02__CT_place_OBSERVATION`: contribution `+0.042002`
- `lag_00__CT_shots_fired_sum`: contribution `+0.024176`
- `lag_00__T_place_CONTROL`: contribution `+0.016733`
- `lag_13__T_place_TROPHY`: contribution `+0.012068`
- `lag_05__CT_place_OBSERVATION`: contribution `+0.011635`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17260`, seconds `38.50`, LSTM delta `+0.2109`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.024176`
- `lag_03__CT_place_SQUEAKY`: contribution `+0.021003`
- `lag_12__CT_place_SQUEAKY`: contribution `+0.016912`
- `lag_00__kill_diff_last_3s`: contribution `+0.010878`
- `lag_13__CT_shots_fired_sum`: contribution `+0.008334`

Top utility-only movements:
- No utility movement among the top local contributors.
