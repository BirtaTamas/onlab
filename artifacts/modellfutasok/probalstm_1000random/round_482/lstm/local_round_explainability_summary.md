# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m3-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `106236`, seconds `17.00`, LSTM `0.0731`, delta `-0.2378`
- tick `106108`, seconds `15.00`, LSTM `0.3335`, delta `-0.1413`
- tick `106140`, seconds `15.50`, LSTM `0.2846`, delta `-0.0489`
- tick `106300`, seconds `18.00`, LSTM `0.0221`, delta `-0.0361`
- tick `105980`, seconds `13.00`, LSTM `0.5054`, delta `-0.0337`
- tick `106012`, seconds `13.50`, LSTM `0.4742`, delta `-0.0312`
- tick `105884`, seconds `11.50`, LSTM `0.5235`, delta `+0.0192`
- tick `105788`, seconds `10.00`, LSTM `0.4967`, delta `-0.0187`
- tick `106268`, seconds `17.50`, LSTM `0.0582`, delta `-0.0149`
- tick `105948`, seconds `12.50`, LSTM `0.5391`, delta `+0.0149`

## Top 15 local ridge features

- `lag_03__T_shots_fired_sum`: coefficient `0.001295`, |coef| `0.001295`
- `lag_07__T_shots_fired_sum`: coefficient `0.001255`, |coef| `0.001255`
- `lag_08__CT4__shots_fired`: coefficient `0.001090`, |coef| `0.001090`
- `lag_03__T2__shots_fired`: coefficient `0.001086`, |coef| `0.001086`
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.001052`, |coef| `0.001052`
- `lag_07__T2__shots_fired`: coefficient `0.001024`, |coef| `0.001024`
- `lag_05__CT3__flash_duration`: coefficient `0.000989`, |coef| `0.000989`
- `lag_10__T2__flash_duration`: coefficient `-0.000982`, |coef| `0.000982`
- `lag_00__T_kills_last_3s`: coefficient `-0.000885`, |coef| `0.000885`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000871`, |coef| `0.000871`
- `lag_13__CT_place_SIDEENTRANCE`: coefficient `-0.000868`, |coef| `0.000868`
- `lag_04__T_place_WATER`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_08__CT_shots_fired_sum`: coefficient `0.000797`, |coef| `0.000797`
- `lag_00__CT1__alive`: coefficient `0.000796`, |coef| `0.000796`
- `lag_00__CT1__hp`: coefficient `0.000785`, |coef| `0.000785`

## Top 10 utility ridge features

- `lag_05__CT3__flash_duration`: coefficient `0.000989` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.000982` (lowers CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000706` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.000699` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.000671` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000666` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000640` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000638` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.000595` (raises CT win probability)
- `lag_12__CT_B_site_active_smokes`: coefficient `-0.000582` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_shots_fired_sum`: coefficient `0.001295` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `0.001255` (raises CT win probability)
- `lag_08__CT4__shots_fired`: coefficient `0.001090` (raises CT win probability)
- `lag_03__T2__shots_fired`: coefficient `0.001086` (raises CT win probability)
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.001052` (lowers CT win probability)
- `lag_07__T2__shots_fired`: coefficient `0.001024` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000885` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.000871` (raises CT win probability)
- `lag_13__CT_place_SIDEENTRANCE`: coefficient `-0.000868` (lowers CT win probability)
- `lag_04__T_place_WATER`: coefficient `-0.000812` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `106236`, seconds `17.00`, LSTM delta `-0.2378`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `-0.012227`
- `lag_08__CT4__shots_fired`: contribution `-0.008219`
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.007907`
- `lag_07__T2__shots_fired`: contribution `-0.007830`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.006550`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `-0.005698`
- `lag_05__CT3__flash_duration`: contribution `-0.005363`

### tick `106108`, seconds `15.00`, LSTM delta `-0.1413`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.012620`
- `lag_03__T2__shots_fired`: contribution `-0.008304`
- `lag_04__CT4__shots_fired`: contribution `-0.004963`
- `lag_04__CT_shots_fired_sum`: contribution `-0.004075`
- `lag_01__CT3__flash_duration`: contribution `-0.003789`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.003789`
- `lag_06__T2__flash_duration`: contribution `-0.002586`
- `lag_12__T_B_site_active_infernos`: contribution `-0.002408`

### tick `106140`, seconds `15.50`, LSTM delta `-0.0489`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.004585`
- `lag_06__CT_shots_fired_sum`: contribution `-0.004193`
- `lag_04__T_shots_fired_sum`: contribution `+0.003712`
- `lag_07__T_shots_fired_sum`: contribution `+0.002822`
- `lag_07__CT_shots_fired_sum`: contribution `+0.002458`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `-0.001553`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001485`
- `lag_13__T_B_site_active_infernos`: contribution `-0.001416`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.001388`
- `lag_13__CT3__flash_duration`: contribution `-0.001364`

### tick `106300`, seconds `18.00`, LSTM delta `-0.0361`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.005118`
- `lag_09__CT_shots_fired_sum`: contribution `+0.004395`
- `lag_11__CT_shots_fired_sum`: contribution `-0.003513`
- `lag_10__CT_shots_fired_sum`: contribution `+0.003224`
- `lag_09__T2__shots_fired`: contribution `+0.002885`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `-0.001847`
- `lag_10__T2__flash_duration`: contribution `+0.001627`
- `lag_14__T_utility_damage_last_5s`: contribution `-0.001626`
- `lag_14__CT_B_site_active_infernos`: contribution `-0.001620`

### tick `105980`, seconds `13.00`, LSTM delta `-0.0337`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.002820`
- `lag_00__CT4__shots_fired`: contribution `-0.002561`
- `lag_01__CT_shots_fired_sum`: contribution `+0.001667`
- `lag_00__T_shots_fired_sum`: contribution `-0.001632`
- `lag_00__CT1__molly`: contribution `-0.001216`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.001216`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.001098`
- `lag_00__CT5__flash`: contribution `-0.001055`
- `lag_13__T_B_site_active_smokes`: contribution `-0.000804`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.000803`
