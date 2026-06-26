# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `35946`, seconds `95.00`, LSTM `0.6931`, delta `-0.2031`
- tick `32906`, seconds `47.50`, LSTM `0.8684`, delta `+0.1729`
- tick `34186`, seconds `67.50`, LSTM `0.8892`, delta `+0.1606`
- tick `34058`, seconds `65.50`, LSTM `0.7370`, delta `-0.0985`
- tick `33706`, seconds `60.00`, LSTM `0.8483`, delta `-0.0641`
- tick `34602`, seconds `74.00`, LSTM `0.9212`, delta `-0.0580`
- tick `33802`, seconds `61.50`, LSTM `0.8824`, delta `+0.0462`
- tick `33674`, seconds `59.50`, LSTM `0.9124`, delta `+0.0449`
- tick `34794`, seconds `77.00`, LSTM `0.8742`, delta `+0.0394`
- tick `34410`, seconds `71.00`, LSTM `0.9654`, delta `+0.0389`

## Top 15 local ridge features

- `lag_07__T_place_JUNGLE`: coefficient `-0.003066`, |coef| `0.003066`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002823`, |coef| `0.002823`
- `lag_13__T_place_SNIPERSNEST`: coefficient `0.002694`, |coef| `0.002694`
- `lag_00__kill_diff_last_3s`: coefficient `0.002537`, |coef| `0.002537`
- `lag_12__T_place_STAIRS`: coefficient `-0.002152`, |coef| `0.002152`
- `lag_00__CT_kills_last_3s`: coefficient `0.001598`, |coef| `0.001598`
- `lag_00__T_kills_last_3s`: coefficient `-0.001586`, |coef| `0.001586`
- `lag_00__damage_diff_last_5s`: coefficient `0.001481`, |coef| `0.001481`
- `lag_13__T3__duck_amount`: coefficient `-0.001309`, |coef| `0.001309`
- `lag_00__CT2__shots_fired`: coefficient `0.001226`, |coef| `0.001226`
- `lag_13__T_place_CTSPAWN`: coefficient `-0.001132`, |coef| `0.001132`
- `lag_04__T_place_JUNGLE`: coefficient `-0.001102`, |coef| `0.001102`
- `lag_13__CT1__duck_amount`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_01__CT_duck_amount_mean`: coefficient `0.001044`, |coef| `0.001044`
- `lag_07__T_place_CTSPAWN`: coefficient `0.001036`, |coef| `0.001036`

## Top 10 utility ridge features

- `lag_00__T4__molly`: coefficient `-0.000838` (lowers CT win probability)
- `lag_07__CT2__smoke`: coefficient `-0.000825` (lowers CT win probability)
- `lag_02__CT4__smoke`: coefficient `-0.000768` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000732` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000566` (lowers CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `-0.000563` (lowers CT win probability)
- `lag_09__T1__flash`: coefficient `-0.000519` (lowers CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `-0.000491` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000490` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000490` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_JUNGLE`: coefficient `-0.003066` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002823` (raises CT win probability)
- `lag_13__T_place_SNIPERSNEST`: coefficient `0.002694` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002537` (raises CT win probability)
- `lag_12__T_place_STAIRS`: coefficient `-0.002152` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001598` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001586` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001481` (raises CT win probability)
- `lag_13__T3__duck_amount`: coefficient `-0.001309` (lowers CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.001226` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35946`, seconds `95.00`, LSTM delta `-0.2031`

Top all feature movements:
- `lag_13__T_place_SNIPERSNEST`: contribution `-0.047866`
- `lag_07__T_place_JUNGLE`: contribution `-0.039711`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013731`
- `lag_00__kill_diff_last_3s`: contribution `-0.006106`
- `lag_13__T_place_CTSPAWN`: contribution `-0.005399`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32906`, seconds `47.50`, LSTM delta `+0.1729`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.009808`
- `lag_00__kill_diff_last_3s`: contribution `+0.006106`
- `lag_13__T3__duck_amount`: contribution `+0.004935`
- `lag_00__CT_kills_last_3s`: contribution `+0.004614`
- `lag_13__CT1__duck_amount`: contribution `+0.004042`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.002068`
- `lag_00__T4__molly`: contribution `+0.001826`

### tick `34186`, seconds `67.50`, LSTM delta `+0.1606`

Top all feature movements:
- `lag_12__T_place_STAIRS`: contribution `+0.041207`
- `lag_00__kill_diff_last_3s`: contribution `+0.006106`
- `lag_00__CT_kills_last_3s`: contribution `+0.004614`
- `lag_15__CT_shots_fired_sum`: contribution `+0.003997`
- `lag_12__T_place_CONNECTOR`: contribution `+0.003912`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `+0.001685`

### tick `34058`, seconds `65.50`, LSTM delta `-0.0985`

Top all feature movements:
- `lag_08__T_place_STAIRS`: contribution `-0.016545`
- `lag_00__kill_diff_last_3s`: contribution `-0.006106`
- `lag_00__T_kills_last_3s`: contribution `-0.005023`
- `lag_13__T3__duck_amount`: contribution `-0.004458`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003923`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `-0.001394`
- `lag_15__T_active_infernos`: contribution `-0.001257`

### tick `33706`, seconds `60.00`, LSTM delta `-0.0641`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.019615`
- `lag_06__T_place_STAIRS`: contribution `-0.009093`
- `lag_00__kill_diff_last_3s`: contribution `-0.006106`
- `lag_00__CT2__shots_fired`: contribution `-0.006092`
- `lag_00__T_kills_last_3s`: contribution `-0.005023`

Top utility-only movements:
- `lag_04__T_active_infernos`: contribution `-0.002039`
- `lag_04__T_B_site_active_infernos`: contribution `-0.001386`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001185`
