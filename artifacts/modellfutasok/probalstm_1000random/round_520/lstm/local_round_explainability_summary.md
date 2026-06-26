# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `30304`, seconds `54.50`, LSTM `0.7637`, delta `+0.1650`
- tick `29728`, seconds `45.50`, LSTM `0.7101`, delta `+0.1235`
- tick `30720`, seconds `61.00`, LSTM `0.9468`, delta `+0.0780`
- tick `27936`, seconds `17.50`, LSTM `0.6223`, delta `+0.0433`
- tick `27872`, seconds `16.50`, LSTM `0.5844`, delta `+0.0408`
- tick `30688`, seconds `60.50`, LSTM `0.8688`, delta `+0.0363`
- tick `30528`, seconds `58.00`, LSTM `0.8267`, delta `+0.0361`
- tick `29312`, seconds `39.00`, LSTM `0.5799`, delta `-0.0330`
- tick `27616`, seconds `12.50`, LSTM `0.5746`, delta `-0.0308`
- tick `29920`, seconds `48.50`, LSTM `0.6458`, delta `-0.0293`

## Top 15 local ridge features

- `lag_03__CT_place_SECRET`: coefficient `0.002047`, |coef| `0.002047`
- `lag_08__CT_place_LOCKERROOM`: coefficient `0.001111`, |coef| `0.001111`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001001`, |coef| `0.001001`
- `lag_02__CT_place_SECRET`: coefficient `0.000934`, |coef| `0.000934`
- `lag_00__T_place_SECRET`: coefficient `-0.000924`, |coef| `0.000924`
- `lag_07__CT_place_LOCKERROOM`: coefficient `0.000878`, |coef| `0.000878`
- `lag_11__CT_place_HEAVEN`: coefficient `-0.000867`, |coef| `0.000867`
- `lag_00__CT_kills_last_3s`: coefficient `0.000866`, |coef| `0.000866`
- `lag_13__CT_place_VENTS`: coefficient `0.000862`, |coef| `0.000862`
- `lag_04__CT_place_SECRET`: coefficient `0.000823`, |coef| `0.000823`
- `lag_02__CT_place_LOCKERROOM`: coefficient `0.000822`, |coef| `0.000822`
- `lag_10__CT_place_HEAVEN`: coefficient `-0.000792`, |coef| `0.000792`
- `lag_07__CT_place_HELL`: coefficient `0.000776`, |coef| `0.000776`
- `lag_05__CT_place_SECRET`: coefficient `0.000774`, |coef| `0.000774`
- `lag_02__T5__flash_duration`: coefficient `-0.000769`, |coef| `0.000769`

## Top 10 utility ridge features

- `lag_02__T5__flash_duration`: coefficient `-0.000769` (lowers CT win probability)
- `lag_15__T5__flash_duration`: coefficient `0.000462` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.000453` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.000442` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000440` (lowers CT win probability)
- `lag_07__CT3__molly`: coefficient `-0.000421` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000357` (raises CT win probability)
- `lag_01__CT5__smoke`: coefficient `-0.000353` (lowers CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `-0.000337` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000326` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_SECRET`: coefficient `0.002047` (raises CT win probability)
- `lag_08__CT_place_LOCKERROOM`: coefficient `0.001111` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001001` (raises CT win probability)
- `lag_02__CT_place_SECRET`: coefficient `0.000934` (raises CT win probability)
- `lag_00__T_place_SECRET`: coefficient `-0.000924` (lowers CT win probability)
- `lag_07__CT_place_LOCKERROOM`: coefficient `0.000878` (raises CT win probability)
- `lag_11__CT_place_HEAVEN`: coefficient `-0.000867` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000866` (raises CT win probability)
- `lag_13__CT_place_VENTS`: coefficient `0.000862` (raises CT win probability)
- `lag_04__CT_place_SECRET`: coefficient `0.000823` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `30304`, seconds `54.50`, LSTM delta `+0.1650`

Top all feature movements:
- `lag_03__CT_place_SECRET`: contribution `+0.021068`
- `lag_00__T_place_SECRET`: contribution `+0.004864`
- `lag_11__CT_place_HEAVEN`: contribution `+0.004682`
- `lag_07__CT_place_HELL`: contribution `+0.004210`
- `lag_11__T1__is_scoped`: contribution `+0.004109`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29728`, seconds `45.50`, LSTM delta `+0.1235`

Top all feature movements:
- `lag_13__CT_place_VENTS`: contribution `+0.007236`
- `lag_07__CT_place_VENTS`: contribution `+0.005900`
- `lag_02__T5__flash_duration`: contribution `+0.005688`
- `lag_08__CT_place_GARAGE`: contribution `+0.004922`
- `lag_10__CT_place_HEAVEN`: contribution `+0.004276`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.005688`
- `lag_15__T5__flash_duration`: contribution `+0.003420`
- `lag_15__CT5__flash_duration`: contribution `+0.002474`
- `lag_05__CT5__flash_duration`: contribution `+0.002411`

### tick `30720`, seconds `61.00`, LSTM delta `+0.0780`

Top all feature movements:
- `lag_08__CT_place_LOCKERROOM`: contribution `+0.013827`
- `lag_10__CT_place_HEAVEN`: contribution `+0.004276`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003477`
- `lag_10__T_place_CONTROL`: contribution `+0.003026`
- `lag_09__CT_place_RAFTERS`: contribution `+0.002735`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27936`, seconds `17.50`, LSTM delta `+0.0433`

Top all feature movements:
- `lag_02__CT_place_CRANE`: contribution `+0.007852`
- `lag_06__T1__duck_amount`: contribution `+0.001839`
- `lag_02__CT5__is_scoped`: contribution `+0.001796`
- `lag_00__T_A_site_active_infernos`: contribution `+0.001552`
- `lag_00__T_B_site_active_infernos`: contribution `+0.001401`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.001552`
- `lag_00__T_B_site_active_infernos`: contribution `+0.001401`
- `lag_14__T_A_site_active_infernos`: contribution `+0.001181`
- `lag_14__T_B_site_active_infernos`: contribution `+0.001059`
- `lag_00__T_active_infernos`: contribution `+0.000760`

### tick `27872`, seconds `16.50`, LSTM delta `+0.0408`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `+0.011912`
- `lag_11__CT_place_RAFTERS`: contribution `-0.002044`
- `lag_00__CT5__is_scoped`: contribution `+0.001833`
- `lag_00__CT_place_RAFTERS`: contribution `+0.001473`
- `lag_02__T5__is_walking`: contribution `+0.001408`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `+0.001175`
- `lag_12__T_B_site_active_infernos`: contribution `+0.001058`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.001011`
