# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `2`

## Largest probability jumps

- tick `5697`, seconds `0.50`, LSTM `0.0207`, delta `-0.0320`
- tick `10017`, seconds `68.00`, LSTM `0.0162`, delta `-0.0203`
- tick `9537`, seconds `60.50`, LSTM `0.0240`, delta `+0.0056`
- tick `10049`, seconds `68.50`, LSTM `0.0111`, delta `-0.0052`
- tick `5793`, seconds `2.00`, LSTM `0.0163`, delta `-0.0047`
- tick `9633`, seconds `62.00`, LSTM `0.0340`, delta `+0.0046`
- tick `9985`, seconds `67.50`, LSTM `0.0366`, delta `-0.0044`
- tick `10081`, seconds `69.00`, LSTM `0.0068`, delta `-0.0043`
- tick `9089`, seconds `53.50`, LSTM `0.0158`, delta `+0.0038`
- tick `9249`, seconds `56.00`, LSTM `0.0217`, delta `+0.0031`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000726`, |coef| `0.000726`
- `lag_00__T4__shots_fired`: coefficient `-0.000286`, |coef| `0.000286`
- `lag_15__T_place_OUTSIDELONG`: coefficient `-0.000276`, |coef| `0.000276`
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000274`, |coef| `0.000274`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000264`, |coef| `0.000264`
- `lag_15__T_place_RUINS`: coefficient `0.000256`, |coef| `0.000256`
- `lag_14__CT_place_BACKOFB`: coefficient `0.000250`, |coef| `0.000250`
- `lag_10__CT_place_BACKOFB`: coefficient `0.000246`, |coef| `0.000246`
- `lag_01__T_flash_alpha_mean`: coefficient `0.000215`, |coef| `0.000215`
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.000191`, |coef| `0.000191`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000180`, |coef| `0.000180`
- `lag_04__CT_place_CONNECTOR`: coefficient `-0.000173`, |coef| `0.000173`
- `lag_11__CT2__is_walking`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_07__CT_place_BACKOFB`: coefficient `-0.000160`, |coef| `0.000160`
- `lag_00__CT1__is_walking`: coefficient `0.000158`, |coef| `0.000158`

## Top 10 utility ridge features

- `lag_01__CT_flash_alpha_mean`: coefficient `0.000274` (raises CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `0.000215` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000108` (raises CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000095` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000075` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000074` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000073` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000067` (lowers CT win probability)
- `lag_10__CT_flash_alpha_mean`: coefficient `0.000067` (raises CT win probability)
- `lag_05__CT_flash_alpha_mean`: coefficient `0.000066` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000726` (lowers CT win probability)
- `lag_00__T4__shots_fired`: coefficient `-0.000286` (lowers CT win probability)
- `lag_15__T_place_OUTSIDELONG`: coefficient `-0.000276` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000264` (lowers CT win probability)
- `lag_15__T_place_RUINS`: coefficient `0.000256` (raises CT win probability)
- `lag_14__CT_place_BACKOFB`: coefficient `0.000250` (raises CT win probability)
- `lag_10__CT_place_BACKOFB`: coefficient `0.000246` (raises CT win probability)
- `lag_05__CT_place_CONNECTOR`: coefficient `-0.000191` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000180` (raises CT win probability)
- `lag_04__CT_place_CONNECTOR`: coefficient `-0.000173` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `5697`, seconds `0.50`, LSTM delta `-0.0320`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.018702`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.002038`
- `lag_01__T_flash_alpha_mean`: contribution `-0.001252`
- `lag_01__T_place_TSPAWN`: contribution `-0.000619`
- `lag_00__T_velocity_mean`: contribution `-0.000371`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.002038`
- `lag_01__T_flash_alpha_mean`: contribution `-0.001252`
- `lag_01__smoke_inv_diff`: contribution `-0.000344`
- `lag_01__utility_inv_diff`: contribution `-0.000270`
- `lag_01__T3__utility_total`: contribution `-0.000177`

### tick `10017`, seconds `68.00`, LSTM delta `-0.0203`

Top all feature movements:
- `lag_14__CT_place_BACKOFB`: contribution `-0.001428`
- `lag_10__CT_place_BACKOFB`: contribution `-0.001405`
- `lag_00__T_shots_fired_sum`: contribution `-0.001385`
- `lag_15__T_place_RUINS`: contribution `-0.001363`
- `lag_15__T_place_OUTSIDELONG`: contribution `-0.001357`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9537`, seconds `60.50`, LSTM delta `+0.0056`

Top all feature movements:
- `lag_06__CT_place_BRICKS`: contribution `+0.001194`
- `lag_08__CT_place_BACKOFB`: contribution `-0.000501`
- `lag_09__CT_place_BACKOFB`: contribution `+0.000485`
- `lag_08__CT_place_BRICKS`: contribution `+0.000466`
- `lag_11__CT2__is_walking`: contribution `-0.000398`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10049`, seconds `68.50`, LSTM delta `-0.0052`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.001187`
- `lag_00__T4__shots_fired`: contribution `-0.001059`
- `lag_05__CT_place_CONNECTOR`: contribution `-0.000685`
- `lag_14__CT4__duck_amount`: contribution `+0.000474`
- `lag_01__T4__shots_fired`: contribution `-0.000370`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5793`, seconds `2.00`, LSTM delta `-0.0047`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `+0.003749`
- `lag_04__CT_place_CTSIDEUPPER`: contribution `-0.002374`
- `lag_00__CT_place_CTSIDEUPPER`: contribution `-0.001855`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `-0.000943`
- `lag_02__CT_smokes_last_5s`: contribution `-0.000766`

Top utility-only movements:
- `lag_02__CT_smokes_last_5s`: contribution `-0.000766`
- `lag_04__CT_flash_alpha_mean`: contribution `-0.000448`
- `lag_04__T_flash_alpha_mean`: contribution `-0.000275`
- `lag_04__smoke_inv_diff`: contribution `-0.000063`
- `lag_04__utility_inv_diff`: contribution `-0.000050`
