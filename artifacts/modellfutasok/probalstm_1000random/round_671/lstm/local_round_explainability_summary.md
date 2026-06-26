# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `86777`, seconds `12.00`, LSTM `0.1473`, delta `-0.2942`
- tick `86841`, seconds `13.00`, LSTM `0.3722`, delta `+0.2303`
- tick `86873`, seconds `13.50`, LSTM `0.2111`, delta `-0.1611`
- tick `86969`, seconds `15.00`, LSTM `0.3218`, delta `+0.1494`
- tick `86713`, seconds `11.00`, LSTM `0.4376`, delta `+0.0708`
- tick `88153`, seconds `33.50`, LSTM `0.2589`, delta `-0.0627`
- tick `87417`, seconds `22.00`, LSTM `0.3453`, delta `-0.0605`
- tick `87161`, seconds `18.00`, LSTM `0.3049`, delta `-0.0515`
- tick `86041`, seconds `0.50`, LSTM `0.3095`, delta `-0.0514`
- tick `88601`, seconds `40.50`, LSTM `0.0687`, delta `-0.0512`

## Top 15 local ridge features

- `lag_13__CT_place_HELL`: coefficient `-0.003323`, |coef| `0.003323`
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.003201`, |coef| `0.003201`
- `lag_11__CT_place_HELL`: coefficient `0.002082`, |coef| `0.002082`
- `lag_01__CT_place_HUT`: coefficient `-0.001914`, |coef| `0.001914`
- `lag_15__CT_place_HELL`: coefficient `0.001579`, |coef| `0.001579`
- `lag_02__CT_place_SQUEAKY`: coefficient `0.001552`, |coef| `0.001552`
- `lag_03__CT_place_HUT`: coefficient `0.001489`, |coef| `0.001489`
- `lag_03__CT_place_SQUEAKY`: coefficient `-0.001416`, |coef| `0.001416`
- `lag_08__CT_place_ADMIN`: coefficient `0.001378`, |coef| `0.001378`
- `lag_00__CT_place_DECON`: coefficient `-0.001292`, |coef| `0.001292`
- `lag_03__CT_place_MINI`: coefficient `0.001216`, |coef| `0.001216`
- `lag_01__T5__duck_amount`: coefficient `-0.001150`, |coef| `0.001150`
- `lag_04__T_place_ROOF`: coefficient `0.001126`, |coef| `0.001126`
- `lag_12__CT_place_HELL`: coefficient `0.001091`, |coef| `0.001091`
- `lag_09__CT_place_MINI`: coefficient `-0.001080`, |coef| `0.001080`

## Top 10 utility ridge features

- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000751` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000730` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000558` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000529` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000482` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `-0.000478` (lowers CT win probability)
- `lag_07__T1__molly`: coefficient `0.000469` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.000469` (raises CT win probability)
- `lag_08__CT3__smoke`: coefficient `0.000463` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.000457` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_HELL`: coefficient `-0.003323` (lowers CT win probability)
- `lag_00__CT_place_SQUEAKY`: coefficient `-0.003201` (lowers CT win probability)
- `lag_11__CT_place_HELL`: coefficient `0.002082` (raises CT win probability)
- `lag_01__CT_place_HUT`: coefficient `-0.001914` (lowers CT win probability)
- `lag_15__CT_place_HELL`: coefficient `0.001579` (raises CT win probability)
- `lag_02__CT_place_SQUEAKY`: coefficient `0.001552` (raises CT win probability)
- `lag_03__CT_place_HUT`: coefficient `0.001489` (raises CT win probability)
- `lag_03__CT_place_SQUEAKY`: coefficient `-0.001416` (lowers CT win probability)
- `lag_08__CT_place_ADMIN`: coefficient `0.001378` (raises CT win probability)
- `lag_00__CT_place_DECON`: coefficient `-0.001292` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `86777`, seconds `12.00`, LSTM delta `-0.2942`

Top all feature movements:
- `lag_00__CT_place_SQUEAKY`: contribution `-0.042569`
- `lag_13__CT_place_HELL`: contribution `-0.036035`
- `lag_11__CT_place_HELL`: contribution `-0.022584`
- `lag_01__CT_place_HUT`: contribution `-0.018669`
- `lag_08__CT_place_ADMIN`: contribution `-0.009575`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `-0.002652`

### tick `86841`, seconds `13.00`, LSTM delta `+0.2303`

Top all feature movements:
- `lag_13__CT_place_HELL`: contribution `+0.036035`
- `lag_02__CT_place_SQUEAKY`: contribution `+0.020643`
- `lag_15__CT_place_HELL`: contribution `+0.017127`
- `lag_03__CT_place_HUT`: contribution `+0.014520`
- `lag_11__CT_place_HELL`: contribution `-0.011292`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86873`, seconds `13.50`, LSTM delta `-0.1611`

Top all feature movements:
- `lag_03__CT_place_SQUEAKY`: contribution `-0.018838`
- `lag_00__CT_place_HUT`: contribution `-0.009129`
- `lag_14__CT_place_ADMIN`: contribution `-0.007026`
- `lag_09__CT_place_MINI`: contribution `-0.006625`
- `lag_12__CT_place_HELL`: contribution `-0.005918`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002273`

### tick `86969`, seconds `15.00`, LSTM delta `+0.1494`

Top all feature movements:
- `lag_00__CT_place_SQUEAKY`: contribution `+0.042569`
- `lag_03__CT_place_HUT`: contribution `+0.014520`
- `lag_06__CT_place_SQUEAKY`: contribution `+0.014153`
- `lag_00__CT_place_HUT`: contribution `+0.009129`
- `lag_15__CT_place_HELL`: contribution `-0.008563`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `86713`, seconds `11.00`, LSTM delta `+0.0708`

Top all feature movements:
- `lag_11__CT_place_HELL`: contribution `+0.022584`
- `lag_12__CT_place_HELL`: contribution `+0.005918`
- `lag_09__CT_place_HELL`: contribution `-0.005451`
- `lag_07__CT_place_HELL`: contribution `+0.003977`
- `lag_06__CT_place_HEAVEN`: contribution `-0.003913`

Top utility-only movements:
- No utility movement among the top local contributors.
