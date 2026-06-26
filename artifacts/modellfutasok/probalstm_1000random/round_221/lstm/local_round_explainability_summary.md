# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `28470`, seconds `0.50`, LSTM `0.0706`, delta `-0.0775`
- tick `31158`, seconds `42.50`, LSTM `0.0772`, delta `-0.0677`
- tick `29238`, seconds `12.50`, LSTM `0.1868`, delta `+0.0663`
- tick `29910`, seconds `23.00`, LSTM `0.2309`, delta `+0.0603`
- tick `30006`, seconds `24.50`, LSTM `0.2236`, delta `-0.0460`
- tick `30070`, seconds `25.50`, LSTM `0.1484`, delta `-0.0422`
- tick `28598`, seconds `2.50`, LSTM `0.1577`, delta `+0.0415`
- tick `31510`, seconds `48.00`, LSTM `0.0148`, delta `-0.0410`
- tick `29590`, seconds `18.00`, LSTM `0.2467`, delta `-0.0397`
- tick `29718`, seconds `20.00`, LSTM `0.2011`, delta `-0.0356`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000971`, |coef| `0.000971`
- `lag_05__CT_smokes_last_5s`: coefficient `0.000901`, |coef| `0.000901`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000842`, |coef| `0.000842`
- `lag_00__T_velocity_mean`: coefficient `-0.000759`, |coef| `0.000759`
- `lag_10__CT_he_last_5s`: coefficient `-0.000658`, |coef| `0.000658`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000647`, |coef| `0.000647`
- `lag_15__CT_he_last_5s`: coefficient `0.000642`, |coef| `0.000642`
- `lag_04__CT_he_last_5s`: coefficient `0.000591`, |coef| `0.000591`
- `lag_13__CT_he_last_5s`: coefficient `0.000561`, |coef| `0.000561`
- `lag_06__CT_place_MIDDOORS`: coefficient `-0.000559`, |coef| `0.000559`
- `lag_01__CT_smokes_last_5s`: coefficient `0.000540`, |coef| `0.000540`
- `lag_00__CT_velocity_mean`: coefficient `-0.000538`, |coef| `0.000538`
- `lag_05__CT_place_MIDDOORS`: coefficient `-0.000521`, |coef| `0.000521`
- `lag_12__CT_smokes_last_5s`: coefficient `-0.000517`, |coef| `0.000517`
- `lag_01__CT_walking_count`: coefficient `0.000510`, |coef| `0.000510`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000971` (raises CT win probability)
- `lag_05__CT_smokes_last_5s`: coefficient `0.000901` (raises CT win probability)
- `lag_10__CT_he_last_5s`: coefficient `-0.000658` (lowers CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `0.000642` (raises CT win probability)
- `lag_04__CT_he_last_5s`: coefficient `0.000591` (raises CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `0.000561` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000540` (raises CT win probability)
- `lag_12__CT_smokes_last_5s`: coefficient `-0.000517` (lowers CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `0.000474` (raises CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `0.000449` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000842` (lowers CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000759` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000647` (lowers CT win probability)
- `lag_06__CT_place_MIDDOORS`: coefficient `-0.000559` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000538` (lowers CT win probability)
- `lag_05__CT_place_MIDDOORS`: coefficient `-0.000521` (lowers CT win probability)
- `lag_01__CT_walking_count`: coefficient `0.000510` (raises CT win probability)
- `lag_02__T_place_LOWERTUNNEL`: coefficient `0.000510` (raises CT win probability)
- `lag_15__T_place_MIDDOORS`: coefficient `-0.000505` (lowers CT win probability)
- `lag_01__T5__is_walking`: coefficient `0.000493` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28470`, seconds `0.50`, LSTM delta `-0.0775`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.004027`
- `lag_00__T_velocity_mean`: contribution `-0.002925`
- `lag_01__T_place_TSPAWN`: contribution `-0.002863`
- `lag_00__CT_velocity_mean`: contribution `-0.001769`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001301`

Top utility-only movements:
- `lag_01__T5__utility_total`: contribution `-0.000806`
- `lag_01__T5__flash`: contribution `-0.000682`
- `lag_01__molly_inv_diff`: contribution `-0.000647`
- `lag_01__utility_inv_diff`: contribution `-0.000505`
- `lag_01__T2__smoke`: contribution `-0.000450`

### tick `31158`, seconds `42.50`, LSTM delta `-0.0677`

Top all feature movements:
- `lag_09__CT_place_UNDERA`: contribution `-0.002548`
- `lag_06__T_place_EXTENDEDA`: contribution `-0.002161`
- `lag_15__T_place_MIDDOORS`: contribution `-0.002147`
- `lag_04__T_place_LOWERTUNNEL`: contribution `-0.001718`
- `lag_00__T_kills_last_3s`: contribution `-0.001524`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29238`, seconds `12.50`, LSTM delta `+0.0663`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.016785`
- `lag_10__CT_he_last_5s`: contribution `+0.012075`
- `lag_02__T_place_LOWERTUNNEL`: contribution `+0.002205`
- `lag_13__CT_smokes_last_5s`: contribution `+0.002026`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.001721`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.016785`
- `lag_10__CT_he_last_5s`: contribution `+0.012075`
- `lag_13__CT_smokes_last_5s`: contribution `+0.002026`

### tick `29910`, seconds `23.00`, LSTM delta `+0.0603`

Top all feature movements:
- `lag_10__CT_he_last_5s`: contribution `+0.012075`
- `lag_11__CT_smokes_last_5s`: contribution `+0.007623`
- `lag_00__CT_place_ARAMP`: contribution `+0.002741`
- `lag_14__CT3__flash_duration`: contribution `+0.002645`
- `lag_04__CT_place_ARAMP`: contribution `+0.002548`

Top utility-only movements:
- `lag_10__CT_he_last_5s`: contribution `+0.012075`
- `lag_11__CT_smokes_last_5s`: contribution `+0.007623`
- `lag_14__CT3__flash_duration`: contribution `+0.002645`
- `lag_06__T2__flash_duration`: contribution `+0.000983`

### tick `30006`, seconds `24.50`, LSTM delta `-0.0460`

Top all feature movements:
- `lag_13__CT_he_last_5s`: contribution `-0.010287`
- `lag_14__CT_smokes_last_5s`: contribution `-0.007194`
- `lag_02__T_shots_fired_sum`: contribution `-0.002170`
- `lag_02__T3__shots_fired`: contribution `-0.001837`
- `lag_01__T_place_LOWERTUNNEL`: contribution `-0.001637`

Top utility-only movements:
- `lag_13__CT_he_last_5s`: contribution `-0.010287`
- `lag_14__CT_smokes_last_5s`: contribution `-0.007194`
- `lag_06__CT3__flash_duration`: contribution `-0.001452`
- `lag_01__T2__flash_duration`: contribution `-0.000930`
