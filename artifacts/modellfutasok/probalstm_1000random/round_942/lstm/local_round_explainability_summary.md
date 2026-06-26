# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `12634`, seconds `72.00`, LSTM `0.5503`, delta `-0.2702`
- tick `11546`, seconds `55.00`, LSTM `0.1654`, delta `-0.2696`
- tick `14394`, seconds `99.50`, LSTM `0.7732`, delta `+0.2681`
- tick `14234`, seconds `97.00`, LSTM `0.5132`, delta `-0.2577`
- tick `11770`, seconds `58.50`, LSTM `0.4448`, delta `+0.2517`
- tick `12506`, seconds `70.00`, LSTM `0.7156`, delta `+0.1948`
- tick `10330`, seconds `36.00`, LSTM `0.4061`, delta `-0.0838`
- tick `12122`, seconds `64.00`, LSTM `0.4051`, delta `-0.0759`
- tick `13530`, seconds `86.00`, LSTM `0.7962`, delta `+0.0745`
- tick `12602`, seconds `71.50`, LSTM `0.8204`, delta `+0.0745`

## Top 15 local ridge features

- `lag_11__T_place_LADDER`: coefficient `-0.003615`, |coef| `0.003615`
- `lag_05__T_place_JUNGLE`: coefficient `-0.003511`, |coef| `0.003511`
- `lag_00__kill_diff_last_3s`: coefficient `0.003178`, |coef| `0.003178`
- `lag_10__T_place_JUNGLE`: coefficient `0.003012`, |coef| `0.003012`
- `lag_00__damage_diff_last_5s`: coefficient `0.002999`, |coef| `0.002999`
- `lag_00__CT_place_STAIRS`: coefficient `0.002857`, |coef| `0.002857`
- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002663`, |coef| `0.002663`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002330`, |coef| `0.002330`
- `lag_03__CT1__duck_amount`: coefficient `0.002238`, |coef| `0.002238`
- `lag_00__T_kills_last_3s`: coefficient `-0.002193`, |coef| `0.002193`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.002168`, |coef| `0.002168`
- `lag_12__T_place_CONNECTOR`: coefficient `-0.002033`, |coef| `0.002033`
- `lag_01__CT_place_STAIRS`: coefficient `0.001925`, |coef| `0.001925`
- `lag_00__CT_kills_last_3s`: coefficient `0.001813`, |coef| `0.001813`
- `lag_10__CT_place_UNDERPASS`: coefficient `0.001808`, |coef| `0.001808`

## Top 10 utility ridge features

- `lag_13__CT5__flash_duration`: coefficient `0.001566` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.001360` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001293` (lowers CT win probability)
- `lag_13__CT_flash_duration_sum`: coefficient `0.000981` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000908` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000893` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.000709` (raises CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000706` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.000648` (raises CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000617` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_LADDER`: coefficient `-0.003615` (lowers CT win probability)
- `lag_05__T_place_JUNGLE`: coefficient `-0.003511` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003178` (raises CT win probability)
- `lag_10__T_place_JUNGLE`: coefficient `0.003012` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002999` (raises CT win probability)
- `lag_00__CT_place_STAIRS`: coefficient `0.002857` (raises CT win probability)
- `lag_03__T_place_SNIPERSNEST`: coefficient `-0.002663` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002330` (raises CT win probability)
- `lag_03__CT1__duck_amount`: coefficient `0.002238` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002193` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `12634`, seconds `72.00`, LSTM delta `-0.2702`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `-0.022234`
- `lag_05__CT_place_STAIRS`: contribution `-0.012843`
- `lag_03__CT1__duck_amount`: contribution `-0.008540`
- `lag_00__kill_diff_last_3s`: contribution `-0.007649`
- `lag_00__T_kills_last_3s`: contribution `-0.006949`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11546`, seconds `55.00`, LSTM delta `-0.2696`

Top all feature movements:
- `lag_11__T_place_LADDER`: contribution `-0.081729`
- `lag_12__T_place_CONNECTOR`: contribution `-0.009843`
- `lag_03__CT1__duck_amount`: contribution `-0.008540`
- `lag_06__CT5__flash_duration`: contribution `-0.007882`
- `lag_00__kill_diff_last_3s`: contribution `-0.007649`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.007882`
- `lag_06__CT_flash_duration_sum`: contribution `-0.003510`
- `lag_13__T_B_site_active_infernos`: contribution `-0.002525`

### tick `14394`, seconds `99.50`, LSTM delta `+0.2681`

Top all feature movements:
- `lag_10__T_place_JUNGLE`: contribution `+0.078040`
- `lag_06__T_place_JUNGLE`: contribution `+0.018109`
- `lag_08__T_place_JUNGLE`: contribution `+0.015342`
- `lag_12__T_place_CONNECTOR`: contribution `+0.009843`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008092`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14234`, seconds `97.00`, LSTM delta `-0.2577`

Top all feature movements:
- `lag_05__T_place_JUNGLE`: contribution `-0.090960`
- `lag_03__T_place_JUNGLE`: contribution `-0.013831`
- `lag_01__T_place_JUNGLE`: contribution `-0.012600`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.010485`
- `lag_00__kill_diff_last_3s`: contribution `-0.007649`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11770`, seconds `58.50`, LSTM delta `+0.2517`

Top all feature movements:
- `lag_03__T_place_SNIPERSNEST`: contribution `+0.047324`
- `lag_13__CT5__flash_duration`: contribution `+0.009072`
- `lag_00__kill_diff_last_3s`: contribution `+0.007649`
- `lag_02__CT5__flash_duration`: contribution `+0.007494`
- `lag_00__damage_diff_last_5s`: contribution `+0.006767`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.009072`
- `lag_02__CT5__flash_duration`: contribution `+0.007494`
- `lag_13__CT_flash_duration_sum`: contribution `+0.003794`
