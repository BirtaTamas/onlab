# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `22352`, seconds `12.50`, LSTM `0.2829`, delta `-0.3659`
- tick `25840`, seconds `67.00`, LSTM `0.0357`, delta `-0.3244`
- tick `25552`, seconds `62.50`, LSTM `0.0969`, delta `-0.2981`
- tick `25776`, seconds `66.00`, LSTM `0.2820`, delta `+0.1471`
- tick `22864`, seconds `20.50`, LSTM `0.3486`, delta `+0.0784`
- tick `25808`, seconds `66.50`, LSTM `0.3601`, delta `+0.0781`
- tick `22384`, seconds `13.00`, LSTM `0.2144`, delta `-0.0685`
- tick `22800`, seconds `19.50`, LSTM `0.2587`, delta `-0.0613`
- tick `22672`, seconds `17.50`, LSTM `0.2868`, delta `+0.0598`
- tick `22992`, seconds `22.50`, LSTM `0.3088`, delta `-0.0579`

## Top 15 local ridge features

- `lag_10__CT_shots_fired_sum`: coefficient `0.003385`, |coef| `0.003385`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003304`, |coef| `0.003304`
- `lag_00__T_kills_last_3s`: coefficient `-0.003207`, |coef| `0.003207`
- `lag_00__kill_diff_last_3s`: coefficient `0.002827`, |coef| `0.002827`
- `lag_12__CT_place_TOPOFMID`: coefficient `-0.002799`, |coef| `0.002799`
- `lag_09__T5__flash_duration`: coefficient `-0.002705`, |coef| `0.002705`
- `lag_00__T_damage_last_5s`: coefficient `-0.002503`, |coef| `0.002503`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.002374`, |coef| `0.002374`
- `lag_01__T3__flash_duration`: coefficient `-0.002205`, |coef| `0.002205`
- `lag_01__T2__flash_duration`: coefficient `-0.002185`, |coef| `0.002185`
- `lag_12__CT_place_ARCH`: coefficient `0.002177`, |coef| `0.002177`
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.002143`, |coef| `0.002143`
- `lag_00__damage_diff_last_5s`: coefficient `0.002128`, |coef| `0.002128`
- `lag_00__CT_place_QUAD`: coefficient `0.002127`, |coef| `0.002127`
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.002077`, |coef| `0.002077`

## Top 10 utility ridge features

- `lag_09__T5__flash_duration`: coefficient `-0.002705` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.002205` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.002185` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.002143` (lowers CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `-0.002077` (lowers CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.001571` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.001516` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `0.001505` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001399` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001385` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_shots_fired_sum`: coefficient `0.003385` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003304` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003207` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002827` (raises CT win probability)
- `lag_12__CT_place_TOPOFMID`: coefficient `-0.002799` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002503` (lowers CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.002374` (raises CT win probability)
- `lag_12__CT_place_ARCH`: coefficient `0.002177` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002128` (raises CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.002127` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `22352`, seconds `12.50`, LSTM delta `-0.3659`

Top all feature movements:
- `lag_12__CT_place_TOPOFMID`: contribution `-0.020314`
- `lag_12__CT_place_ARCH`: contribution `-0.017767`
- `lag_01__T3__flash_duration`: contribution `-0.016246`
- `lag_01__T2__flash_duration`: contribution `-0.015686`
- `lag_00__T_shots_fired_sum`: contribution `-0.012387`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.016246`
- `lag_01__T2__flash_duration`: contribution `-0.015686`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.010616`
- `lag_01__T_flash_duration_sum`: contribution `-0.010078`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.008440`

### tick `25840`, seconds `67.00`, LSTM delta `-0.3244`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `-0.021167`
- `lag_00__T_shots_fired_sum`: contribution `-0.012387`
- `lag_12__CT_shots_fired_sum`: contribution `-0.010670`
- `lag_00__T_kills_last_3s`: contribution `-0.010160`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.010157`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `-0.008320`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.004501`
- `lag_05__T_A_site_active_infernos`: contribution `-0.003592`

### tick `25552`, seconds `62.50`, LSTM delta `-0.2981`

Top all feature movements:
- `lag_09__T_flashed_players`: contribution `-0.014678`
- `lag_09__T5__flash_duration`: contribution `-0.014327`
- `lag_00__T_shots_fired_sum`: contribution `-0.012387`
- `lag_01__CT_shots_fired_sum`: contribution `-0.010909`
- `lag_00__T_kills_last_3s`: contribution `-0.010160`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.014327`
- `lag_00__T4__flash_duration`: contribution `-0.007833`
- `lag_02__T5__flash_duration`: contribution `-0.007335`
- `lag_09__T_flash_duration_sum`: contribution `-0.005682`

### tick `25776`, seconds `66.00`, LSTM delta `+0.1471`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `+0.018815`
- `lag_09__T5__flash_duration`: contribution `+0.014327`
- `lag_08__CT_shots_fired_sum`: contribution `+0.007163`
- `lag_00__kill_diff_last_3s`: contribution `+0.006804`
- `lag_07__T4__flash_duration`: contribution `+0.005694`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `+0.014327`
- `lag_07__T4__flash_duration`: contribution `+0.005694`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.002996`

### tick `22864`, seconds `20.50`, LSTM delta `+0.0784`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.027252`
- `lag_00__T5__shots_fired`: contribution `+0.010300`
- `lag_00__T5__duck_amount`: contribution `+0.005605`
- `lag_12__CT_place_RUINS`: contribution `+0.005049`
- `lag_04__T3__flash_duration`: contribution `+0.004377`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.004377`
- `lag_04__T2__flash_duration`: contribution `+0.004217`
- `lag_04__T_flash_duration_sum`: contribution `+0.003072`
- `lag_00__CT2__molly`: contribution `+0.002955`
