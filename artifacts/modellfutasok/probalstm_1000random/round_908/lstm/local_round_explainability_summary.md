# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `56771`, seconds `90.50`, LSTM `0.8899`, delta `+0.1731`
- tick `56323`, seconds `83.50`, LSTM `0.7882`, delta `+0.1447`
- tick `55491`, seconds `70.50`, LSTM `0.5193`, delta `-0.1096`
- tick `55715`, seconds `74.00`, LSTM `0.5804`, delta `+0.0746`
- tick `56419`, seconds `85.00`, LSTM `0.7592`, delta `-0.0388`
- tick `55843`, seconds `76.00`, LSTM `0.6072`, delta `+0.0317`
- tick `56611`, seconds `88.00`, LSTM `0.7370`, delta `-0.0290`
- tick `56291`, seconds `83.00`, LSTM `0.6435`, delta `+0.0288`
- tick `53571`, seconds `40.50`, LSTM `0.6823`, delta `+0.0265`
- tick `52131`, seconds `18.00`, LSTM `0.5978`, delta `+0.0245`

## Top 15 local ridge features

- `lag_13__CT_place_OBSERVATION`: coefficient `-0.001644`, |coef| `0.001644`
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001362`, |coef| `0.001362`
- `lag_11__CT_place_HUT`: coefficient `0.001358`, |coef| `0.001358`
- `lag_00__kill_diff_last_3s`: coefficient `0.001191`, |coef| `0.001191`
- `lag_14__CT_place_VENTS`: coefficient `0.001144`, |coef| `0.001144`
- `lag_06__CT_place_VENTS`: coefficient `0.001140`, |coef| `0.001140`
- `lag_05__CT4__flash_duration`: coefficient `-0.000993`, |coef| `0.000993`
- `lag_00__CT_kills_last_3s`: coefficient `0.000960`, |coef| `0.000960`
- `lag_14__CT_place_HEAVEN`: coefficient `0.000951`, |coef| `0.000951`
- `lag_00__damage_diff_last_5s`: coefficient `0.000950`, |coef| `0.000950`
- `lag_00__T_place_SQUEAKY`: coefficient `-0.000936`, |coef| `0.000936`
- `lag_05__CT_place_HEAVEN`: coefficient `0.000917`, |coef| `0.000917`
- `lag_03__CT_place_VENTS`: coefficient `-0.000904`, |coef| `0.000904`
- `lag_08__CT2__flash_duration`: coefficient `-0.000886`, |coef| `0.000886`
- `lag_12__T_place_SQUEAKY`: coefficient `-0.000873`, |coef| `0.000873`

## Top 10 utility ridge features

- `lag_05__CT4__flash_duration`: coefficient `-0.000993` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.000886` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.000529` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000519` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000465` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `0.000410` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.000407` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000400` (lowers CT win probability)
- `lag_06__CT2__molly`: coefficient `-0.000388` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.000370` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_OBSERVATION`: coefficient `-0.001644` (lowers CT win probability)
- `lag_04__CT_place_OBSERVATION`: coefficient `0.001362` (raises CT win probability)
- `lag_11__CT_place_HUT`: coefficient `0.001358` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001191` (raises CT win probability)
- `lag_14__CT_place_VENTS`: coefficient `0.001144` (raises CT win probability)
- `lag_06__CT_place_VENTS`: coefficient `0.001140` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000960` (raises CT win probability)
- `lag_14__CT_place_HEAVEN`: coefficient `0.000951` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000950` (raises CT win probability)
- `lag_00__T_place_SQUEAKY`: coefficient `-0.000936` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `56771`, seconds `90.50`, LSTM delta `+0.1731`

Top all feature movements:
- `lag_13__CT_place_OBSERVATION`: contribution `+0.028628`
- `lag_11__CT_place_HUT`: contribution `+0.013244`
- `lag_06__CT_place_VENTS`: contribution `+0.009565`
- `lag_03__CT_place_VENTS`: contribution `+0.007585`
- `lag_11__CT_place_HUTROOF`: contribution `+0.005910`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.001866`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.001783`

### tick `56323`, seconds `83.50`, LSTM delta `+0.1447`

Top all feature movements:
- `lag_04__CT_place_OBSERVATION`: contribution `+0.023714`
- `lag_14__CT_place_VENTS`: contribution `+0.009603`
- `lag_05__CT4__flash_duration`: contribution `+0.007754`
- `lag_12__T_place_SQUEAKY`: contribution `+0.005438`
- `lag_15__T_place_TROPHY`: contribution `+0.005035`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `+0.007754`
- `lag_05__CT_flash_duration_sum`: contribution `+0.001421`

### tick `55491`, seconds `70.50`, LSTM delta `-0.1096`

Top all feature movements:
- `lag_02__T_place_CONTROL`: contribution `-0.005489`
- `lag_14__CT_place_HEAVEN`: contribution `-0.005133`
- `lag_05__CT_place_HEAVEN`: contribution `-0.004954`
- `lag_08__CT2__flash_duration`: contribution `-0.004835`
- `lag_13__T_place_TROPHY`: contribution `-0.003808`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.004835`

### tick `55715`, seconds `74.00`, LSTM delta `+0.0746`

Top all feature movements:
- `lag_04__T_place_CONTROL`: contribution `+0.004750`
- `lag_00__T_place_CONTROL`: contribution `+0.003259`
- `lag_01__CT_place_HEAVEN`: contribution `+0.002890`
- `lag_00__kill_diff_last_3s`: contribution `+0.002868`
- `lag_00__CT_kills_last_3s`: contribution `+0.002771`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.001926`

### tick `56419`, seconds `85.00`, LSTM delta `-0.0388`

Top all feature movements:
- `lag_02__CT_place_OBSERVATION`: contribution `-0.010790`
- `lag_07__CT_place_OBSERVATION`: contribution `-0.003531`
- `lag_08__CT4__flash_duration`: contribution `-0.003206`
- `lag_00__CT_place_HUT`: contribution `-0.003089`
- `lag_01__CT_place_HEAVEN`: contribution `+0.002890`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `-0.003206`
- `lag_08__CT_flash_duration_sum`: contribution `+0.000855`
