# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `12`

## Largest probability jumps

- tick `65522`, seconds `54.00`, LSTM `0.3835`, delta `-0.2099`
- tick `65554`, seconds `54.50`, LSTM `0.2518`, delta `-0.1317`
- tick `65586`, seconds `55.00`, LSTM `0.1515`, delta `-0.1003`
- tick `65650`, seconds `56.00`, LSTM `0.0337`, delta `-0.0682`
- tick `65618`, seconds `55.50`, LSTM `0.1020`, delta `-0.0496`
- tick `62322`, seconds `4.00`, LSTM `0.6713`, delta `-0.0297`
- tick `62482`, seconds `6.50`, LSTM `0.6424`, delta `-0.0271`
- tick `62130`, seconds `1.00`, LSTM `0.6738`, delta `+0.0245`
- tick `62354`, seconds `4.50`, LSTM `0.6477`, delta `-0.0236`
- tick `62834`, seconds `12.00`, LSTM `0.5951`, delta `-0.0212`

## Top 15 local ridge features

- `lag_10__T_place_LOWERPARK`: coefficient `-0.003071`, |coef| `0.003071`
- `lag_11__T_place_LOWERPARK`: coefficient `-0.002527`, |coef| `0.002527`
- `lag_10__T_place_FOUNTAIN`: coefficient `0.002370`, |coef| `0.002370`
- `lag_12__T_place_LOWERPARK`: coefficient `-0.002031`, |coef| `0.002031`
- `lag_11__T_place_FOUNTAIN`: coefficient `0.001829`, |coef| `0.001829`
- `lag_09__T_place_LOWERPARK`: coefficient `-0.001657`, |coef| `0.001657`
- `lag_04__T_place_CONNECTOR`: coefficient `-0.001650`, |coef| `0.001650`
- `lag_12__T_place_UPPERPARK`: coefficient `0.001407`, |coef| `0.001407`
- `lag_13__T_place_LOWERPARK`: coefficient `-0.001407`, |coef| `0.001407`
- `lag_12__T_place_FOUNTAIN`: coefficient `0.001331`, |coef| `0.001331`
- `lag_05__T_place_CONNECTOR`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_08__T_place_LOWERPARK`: coefficient `-0.001199`, |coef| `0.001199`
- `lag_09__CT_place_BACKOFA`: coefficient `-0.001151`, |coef| `0.001151`
- `lag_07__CT_place_STAIRS`: coefficient `0.001148`, |coef| `0.001148`
- `lag_01__CT3__is_scoped`: coefficient `0.001106`, |coef| `0.001106`

## Top 10 utility ridge features

- `lag_02__T_A_site_active_infernos`: coefficient `-0.000922` (lowers CT win probability)
- `lag_05__T1__molly`: coefficient `0.000691` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000688` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.000638` (lowers CT win probability)
- `lag_06__T1__molly`: coefficient `0.000575` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.000524` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000516` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000506` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000504` (raises CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.000467` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_LOWERPARK`: coefficient `-0.003071` (lowers CT win probability)
- `lag_11__T_place_LOWERPARK`: coefficient `-0.002527` (lowers CT win probability)
- `lag_10__T_place_FOUNTAIN`: coefficient `0.002370` (raises CT win probability)
- `lag_12__T_place_LOWERPARK`: coefficient `-0.002031` (lowers CT win probability)
- `lag_11__T_place_FOUNTAIN`: coefficient `0.001829` (raises CT win probability)
- `lag_09__T_place_LOWERPARK`: coefficient `-0.001657` (lowers CT win probability)
- `lag_04__T_place_CONNECTOR`: coefficient `-0.001650` (lowers CT win probability)
- `lag_12__T_place_UPPERPARK`: coefficient `0.001407` (raises CT win probability)
- `lag_13__T_place_LOWERPARK`: coefficient `-0.001407` (lowers CT win probability)
- `lag_12__T_place_FOUNTAIN`: coefficient `0.001331` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `65522`, seconds `54.00`, LSTM delta `-0.2099`

Top all feature movements:
- `lag_10__T_place_LOWERPARK`: contribution `-0.024762`
- `lag_10__T_place_FOUNTAIN`: contribution `-0.022407`
- `lag_02__CT_place_BACKOFA`: contribution `-0.010596`
- `lag_07__CT_place_BACKOFA`: contribution `-0.009805`
- `lag_07__CT_place_STAIRS`: contribution `-0.008931`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `-0.002743`

### tick `65554`, seconds `54.50`, LSTM delta `-0.1317`

Top all feature movements:
- `lag_11__T_place_LOWERPARK`: contribution `-0.020376`
- `lag_11__T_place_FOUNTAIN`: contribution `-0.017289`
- `lag_08__CT_place_BACKOFA`: contribution `-0.008153`
- `lag_10__CT_place_STAIRS`: contribution `-0.007465`
- `lag_03__CT_place_BACKOFA`: contribution `-0.007005`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.002048`

### tick `65586`, seconds `55.00`, LSTM delta `-0.1003`

Top all feature movements:
- `lag_12__T_place_LOWERPARK`: contribution `-0.016377`
- `lag_12__T_place_FOUNTAIN`: contribution `-0.012586`
- `lag_09__CT_place_BACKOFA`: contribution `-0.011110`
- `lag_11__CT_place_STAIRS`: contribution `-0.007649`
- `lag_09__CT_place_STAIRS`: contribution `+0.005638`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.001535`

### tick `65650`, seconds `56.00`, LSTM delta `-0.0682`

Top all feature movements:
- `lag_14__T_place_LOWERPARK`: contribution `-0.008605`
- `lag_11__CT_place_STAIRS`: contribution `+0.007649`
- `lag_00__CT2__flash_duration`: contribution `-0.005060`
- `lag_13__CT_place_STAIRS`: contribution `-0.004847`
- `lag_14__T_place_FOUNTAIN`: contribution `-0.004630`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.005060`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001429`

### tick `65618`, seconds `55.50`, LSTM delta `-0.0496`

Top all feature movements:
- `lag_13__T_place_LOWERPARK`: contribution `-0.011349`
- `lag_10__CT_place_BACKOFA`: contribution `-0.008049`
- `lag_10__CT_place_STAIRS`: contribution `+0.007465`
- `lag_13__T_place_FOUNTAIN`: contribution `-0.006822`
- `lag_12__CT_place_STAIRS`: contribution `-0.003702`

Top utility-only movements:
- No utility movement among the top local contributors.
