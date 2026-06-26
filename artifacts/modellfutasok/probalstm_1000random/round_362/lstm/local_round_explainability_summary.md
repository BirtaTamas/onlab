# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `151429`, seconds `104.50`, LSTM `0.7901`, delta `+0.4011`
- tick `150533`, seconds `90.50`, LSTM `0.4729`, delta `+0.2534`
- tick `150469`, seconds `89.50`, LSTM `0.2924`, delta `-0.2462`
- tick `150245`, seconds `86.00`, LSTM `0.5498`, delta `-0.2181`
- tick `151461`, seconds `105.00`, LSTM `0.8825`, delta `+0.0924`
- tick `150181`, seconds `85.00`, LSTM `0.7384`, delta `-0.0755`
- tick `146277`, seconds `24.00`, LSTM `0.9147`, delta `+0.0751`
- tick `150501`, seconds `90.00`, LSTM `0.2196`, delta `-0.0729`
- tick `150085`, seconds `83.50`, LSTM `0.8548`, delta `-0.0687`
- tick `150693`, seconds `93.00`, LSTM `0.3981`, delta `-0.0608`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.008751`, |coef| `0.008751`
- `lag_00__CT_defusing_count`: coefficient `0.007037`, |coef| `0.007037`
- `lag_05__T_duck_amount_mean`: coefficient `-0.004557`, |coef| `0.004557`
- `lag_00__damage_diff_last_5s`: coefficient `0.004538`, |coef| `0.004538`
- `lag_00__kill_diff_last_3s`: coefficient `0.004406`, |coef| `0.004406`
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004403`, |coef| `0.004403`
- `lag_00__CT_kills_last_3s`: coefficient `0.004184`, |coef| `0.004184`
- `lag_03__T_duck_amount_mean`: coefficient `-0.003739`, |coef| `0.003739`
- `lag_12__T_duck_amount_mean`: coefficient `-0.003569`, |coef| `0.003569`
- `lag_00__T1__alive`: coefficient `-0.003493`, |coef| `0.003493`
- `lag_00__T1__hp`: coefficient `-0.003437`, |coef| `0.003437`
- `lag_00__CT3__is_walking`: coefficient `-0.003419`, |coef| `0.003419`
- `lag_00__CT_damage_last_5s`: coefficient `0.003228`, |coef| `0.003228`
- `lag_05__T1__duck_amount`: coefficient `-0.002808`, |coef| `0.002808`
- `lag_13__T_duck_amount_mean`: coefficient `-0.002805`, |coef| `0.002805`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.008751` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.004403` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001506` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001366` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001099` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.001052` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000917` (lowers CT win probability)
- `lag_12__CT5__flash`: coefficient `0.000687` (raises CT win probability)
- `lag_07__CT1__flash`: coefficient `0.000682` (raises CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000632` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007037` (raises CT win probability)
- `lag_05__T_duck_amount_mean`: coefficient `-0.004557` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004538` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004406` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004184` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.003739` (lowers CT win probability)
- `lag_12__T_duck_amount_mean`: coefficient `-0.003569` (lowers CT win probability)
- `lag_00__T1__alive`: coefficient `-0.003493` (lowers CT win probability)
- `lag_00__T1__hp`: coefficient `-0.003437` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.003419` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `151429`, seconds `104.50`, LSTM delta `+0.4011`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.053097`
- `lag_05__T_duck_amount_mean`: contribution `+0.026504`
- `lag_03__T_duck_amount_mean`: contribution `+0.021745`
- `lag_07__T_duck_amount_mean`: contribution `+0.015562`
- `lag_00__CT_kills_last_3s`: contribution `+0.012080`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.053097`

### tick `150533`, seconds `90.50`, LSTM delta `+0.2534`

Top all feature movements:
- `lag_15__CT_kills_last_3s`: contribution `+0.013656`
- `lag_10__CT_place_CONNECTOR`: contribution `+0.012265`
- `lag_00__CT_kills_last_3s`: contribution `+0.012080`
- `lag_00__kill_diff_last_3s`: contribution `+0.010606`
- `lag_11__T_bomb_zone_count`: contribution `+0.010221`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `150469`, seconds `89.50`, LSTM delta `-0.2462`

Top all feature movements:
- `lag_08__CT_place_CONNECTOR`: contribution `-0.014545`
- `lag_00__kill_diff_last_3s`: contribution `-0.010606`
- `lag_00__damage_diff_last_5s`: contribution `-0.010237`
- `lag_02__T_bomb_zone_count`: contribution `-0.006863`
- `lag_13__T_damage_last_5s`: contribution `-0.005304`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `150245`, seconds `86.00`, LSTM delta `-0.2181`

Top all feature movements:
- `lag_15__CT_place_TMAIN`: contribution `-0.016852`
- `lag_00__kill_diff_last_3s`: contribution `-0.010606`
- `lag_00__damage_diff_last_5s`: contribution `-0.010237`
- `lag_02__T_bomb_zone_count`: contribution `+0.006863`
- `lag_10__CT_place_CONNECTOR`: contribution `+0.006133`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `-0.006107`
- `lag_12__T3__flash_duration`: contribution `-0.003666`

### tick `151461`, seconds `105.00`, LSTM delta `+0.0924`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.068217`
- `lag_01__T_flash_alpha_mean`: contribution `+0.026712`
- `lag_05__T_duck_amount_mean`: contribution `-0.026504`
- `lag_05__T1__duck_amount`: contribution `-0.010995`
- `lag_13__T_duck_amount_mean`: contribution `+0.008813`

Top utility-only movements:
- `lag_01__T_flash_alpha_mean`: contribution `+0.026712`
