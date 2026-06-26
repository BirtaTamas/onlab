# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `23`

## Largest probability jumps

- tick `193391`, seconds `97.00`, LSTM `0.1722`, delta `-0.2553`
- tick `193103`, seconds `92.50`, LSTM `0.5821`, delta `-0.2121`
- tick `193327`, seconds `96.00`, LSTM `0.4505`, delta `-0.1814`
- tick `193071`, seconds `92.00`, LSTM `0.7943`, delta `+0.1551`
- tick `191823`, seconds `72.50`, LSTM `0.5621`, delta `+0.1195`
- tick `193039`, seconds `91.50`, LSTM `0.6392`, delta `+0.0645`
- tick `193295`, seconds `95.50`, LSTM `0.6319`, delta `+0.0526`
- tick `194703`, seconds `117.50`, LSTM `0.0624`, delta `+0.0502`
- tick `193551`, seconds `99.50`, LSTM `0.0297`, delta `-0.0472`
- tick `193455`, seconds `98.00`, LSTM `0.0941`, delta `-0.0411`

## Top 15 local ridge features

- `lag_02__T_place_RAFTERS`: coefficient `-0.002397`, |coef| `0.002397`
- `lag_08__T_place_HUT`: coefficient `0.001986`, |coef| `0.001986`
- `lag_08__T_place_HEAVEN`: coefficient `-0.001747`, |coef| `0.001747`
- `lag_00__T_place_HUT`: coefficient `-0.001740`, |coef| `0.001740`
- `lag_00__T_place_CONTROL`: coefficient `-0.001679`, |coef| `0.001679`
- `lag_06__T_place_HUT`: coefficient `0.001674`, |coef| `0.001674`
- `lag_00__kill_diff_last_3s`: coefficient `0.001644`, |coef| `0.001644`
- `lag_01__T_place_HEAVEN`: coefficient `-0.001582`, |coef| `0.001582`
- `lag_00__T_kills_last_3s`: coefficient `-0.001515`, |coef| `0.001515`
- `lag_00__T_place_RAFTERS`: coefficient `-0.001487`, |coef| `0.001487`
- `lag_00__damage_diff_last_5s`: coefficient `0.001473`, |coef| `0.001473`
- `lag_00__CT_place_TROPHY`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001449`, |coef| `0.001449`
- `lag_00__CT_walking_count`: coefficient `-0.001283`, |coef| `0.001283`
- `lag_15__T1__is_walking`: coefficient `-0.001251`, |coef| `0.001251`

## Top 10 utility ridge features

- `lag_10__CT5__flash_duration`: coefficient `0.001202` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.001149` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001008` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000974` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.000838` (raises CT win probability)
- `lag_11__CT_flash_duration_sum`: coefficient `0.000703` (raises CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.000699` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000694` (raises CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `0.000690` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `-0.000655` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_RAFTERS`: coefficient `-0.002397` (lowers CT win probability)
- `lag_08__T_place_HUT`: coefficient `0.001986` (raises CT win probability)
- `lag_08__T_place_HEAVEN`: coefficient `-0.001747` (lowers CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.001740` (lowers CT win probability)
- `lag_00__T_place_CONTROL`: coefficient `-0.001679` (lowers CT win probability)
- `lag_06__T_place_HUT`: coefficient `0.001674` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001644` (raises CT win probability)
- `lag_01__T_place_HEAVEN`: coefficient `-0.001582` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001515` (lowers CT win probability)
- `lag_00__T_place_RAFTERS`: coefficient `-0.001487` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `193391`, seconds `97.00`, LSTM delta `-0.2553`

Top all feature movements:
- `lag_02__T_place_RAFTERS`: contribution `-0.062736`
- `lag_00__CT_place_TROPHY`: contribution `-0.021538`
- `lag_10__T_place_HEAVEN`: contribution `-0.014856`
- `lag_10__T_place_HUT`: contribution `-0.008779`
- `lag_00__CT_place_CONTROL`: contribution `-0.007552`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `-0.004876`
- `lag_09__CT4__flash_duration`: contribution `-0.003602`

### tick `193103`, seconds `92.50`, LSTM delta `-0.2121`

Top all feature movements:
- `lag_01__T_place_HEAVEN`: contribution `-0.019413`
- `lag_09__T_place_HUT`: contribution `-0.011510`
- `lag_07__T_place_HUT`: contribution `-0.010131`
- `lag_12__CT5__flash_duration`: contribution `-0.008013`
- `lag_12__T_place_SQUEAKY`: contribution `-0.006977`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `-0.008013`
- `lag_00__CT4__flash_duration`: contribution `-0.005653`
- `lag_12__CT_flash_duration_sum`: contribution `-0.002916`

### tick `193327`, seconds `96.00`, LSTM delta `-0.1814`

Top all feature movements:
- `lag_00__T_place_RAFTERS`: contribution `-0.038919`
- `lag_08__T_place_HEAVEN`: contribution `-0.021441`
- `lag_08__T_place_HUT`: contribution `-0.018509`
- `lag_00__T_place_HEAVEN`: contribution `-0.009250`
- `lag_00__T_shots_fired_sum`: contribution `-0.006520`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.004700`
- `lag_06__CT5__flash_duration`: contribution `-0.003789`

### tick `193071`, seconds `92.00`, LSTM delta `+0.1551`

Top all feature movements:
- `lag_08__T_place_HUT`: contribution `+0.018509`
- `lag_00__T_place_HUT`: contribution `+0.016220`
- `lag_06__T_place_HUT`: contribution `+0.015605`
- `lag_00__T_place_HEAVEN`: contribution `+0.009250`
- `lag_11__CT5__flash_duration`: contribution `+0.006791`

Top utility-only movements:
- `lag_11__CT5__flash_duration`: contribution `+0.006791`
- `lag_11__CT_flash_duration_sum`: contribution `+0.003134`
- `lag_07__CT4__flash_duration`: contribution `+0.002377`

### tick `191823`, seconds `72.50`, LSTM delta `+0.1195`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.023862`
- `lag_07__T_place_CONTROL`: contribution `+0.007120`
- `lag_09__CT_place_HEAVEN`: contribution `+0.006346`
- `lag_01__CT_place_RAFTERS`: contribution `+0.003958`
- `lag_00__kill_diff_last_3s`: contribution `+0.003956`

Top utility-only movements:
- No utility movement among the top local contributors.
