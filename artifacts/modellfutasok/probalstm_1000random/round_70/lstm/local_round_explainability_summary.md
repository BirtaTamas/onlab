# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `15372`, seconds `25.50`, LSTM `0.2208`, delta `-0.3264`
- tick `15532`, seconds `28.00`, LSTM `0.0940`, delta `-0.2864`
- tick `15468`, seconds `27.00`, LSTM `0.3763`, delta `+0.1969`
- tick `16204`, seconds `38.50`, LSTM `0.1643`, delta `+0.1272`
- tick `15404`, seconds `26.00`, LSTM `0.1432`, delta `-0.0775`
- tick `14444`, seconds `11.00`, LSTM `0.5910`, delta `+0.0523`
- tick `16236`, seconds `39.00`, LSTM `0.1193`, delta `-0.0450`
- tick `15788`, seconds `32.00`, LSTM `0.0566`, delta `+0.0421`
- tick `16588`, seconds `44.50`, LSTM `0.0938`, delta `-0.0377`
- tick `15436`, seconds `26.50`, LSTM `0.1794`, delta `+0.0362`

## Top 15 local ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.003180`, |coef| `0.003180`
- `lag_02__T_place_STAIRS`: coefficient `-0.003136`, |coef| `0.003136`
- `lag_03__T_place_CONNECTOR`: coefficient `-0.002534`, |coef| `0.002534`
- `lag_00__T_place_STAIRS`: coefficient `0.002123`, |coef| `0.002123`
- `lag_09__CT_place_SNIPERSNEST`: coefficient `0.002121`, |coef| `0.002121`
- `lag_04__T_A_site_active_infernos`: coefficient `-0.001956`, |coef| `0.001956`
- `lag_00__kill_diff_last_3s`: coefficient `0.001948`, |coef| `0.001948`
- `lag_00__T_kills_last_3s`: coefficient `-0.001792`, |coef| `0.001792`
- `lag_12__T_place_PALACEALLEY`: coefficient `0.001765`, |coef| `0.001765`
- `lag_10__CT3__duck_amount`: coefficient `-0.001757`, |coef| `0.001757`
- `lag_01__CT5__duck_amount`: coefficient `-0.001704`, |coef| `0.001704`
- `lag_04__T_place_PALACEINTERIOR`: coefficient `0.001682`, |coef| `0.001682`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_05__CT_place_CONNECTOR`: coefficient `0.001635`, |coef| `0.001635`
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001553`, |coef| `0.001553`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `-0.001956` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.001553` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001429` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.001346` (lowers CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.001259` (raises CT win probability)
- `lag_08__CT2__molly`: coefficient `0.001205` (raises CT win probability)
- `lag_07__T3__molly`: coefficient `0.001095` (raises CT win probability)
- `lag_01__CT1__smoke`: coefficient `0.001085` (raises CT win probability)
- `lag_03__T2__molly`: coefficient `0.001083` (raises CT win probability)
- `lag_03__T5__molly`: coefficient `0.001077` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.003180` (raises CT win probability)
- `lag_02__T_place_STAIRS`: coefficient `-0.003136` (lowers CT win probability)
- `lag_03__T_place_CONNECTOR`: coefficient `-0.002534` (lowers CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `0.002123` (raises CT win probability)
- `lag_09__CT_place_SNIPERSNEST`: coefficient `0.002121` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001948` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001792` (lowers CT win probability)
- `lag_12__T_place_PALACEALLEY`: coefficient `0.001765` (raises CT win probability)
- `lag_10__CT3__duck_amount`: coefficient `-0.001757` (lowers CT win probability)
- `lag_01__CT5__duck_amount`: coefficient `-0.001704` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15372`, seconds `25.50`, LSTM delta `-0.3264`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.020510`
- `lag_03__T_place_CONNECTOR`: contribution `-0.012270`
- `lag_09__CT_place_SNIPERSNEST`: contribution `-0.011358`
- `lag_02__T_place_CONNECTOR`: contribution `-0.008108`
- `lag_10__CT3__duck_amount`: contribution `-0.006537`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.005822`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.005334`

### tick `15532`, seconds `28.00`, LSTM delta `-0.2864`

Top all feature movements:
- `lag_02__T_place_STAIRS`: contribution `-0.060037`
- `lag_04__T_A_site_active_infernos`: contribution `-0.011643`
- `lag_01__CT_shots_fired_sum`: contribution `-0.010015`
- `lag_02__T_place_CONNECTOR`: contribution `+0.008108`
- `lag_02__T_A_site_active_infernos`: contribution `-0.007492`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.011643`
- `lag_02__T_A_site_active_infernos`: contribution `-0.007492`
- `lag_04__T_active_infernos`: contribution `-0.005608`
- `lag_02__T_active_infernos`: contribution `-0.003750`

### tick `15468`, seconds `27.00`, LSTM delta `+0.1969`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.040643`
- `lag_00__T_A_site_active_infernos`: contribution `+0.008505`
- `lag_02__T_A_site_active_infernos`: contribution `+0.007492`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005170`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005008`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `+0.008505`
- `lag_02__T_A_site_active_infernos`: contribution `+0.007492`
- `lag_00__T_active_infernos`: contribution `+0.004164`
- `lag_02__T_active_infernos`: contribution `+0.003750`

### tick `16204`, seconds `38.50`, LSTM delta `+0.1272`

Top all feature movements:
- `lag_10__T_place_JUNGLE`: contribution `+0.017013`
- `lag_00__T_place_JUNGLE`: contribution `+0.016066`
- `lag_02__T_shots_fired_sum`: contribution `+0.006980`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006204`
- `lag_09__T_place_JUNGLE`: contribution `+0.005531`

Top utility-only movements:
- `lag_03__CT_flash_alpha_mean`: contribution `+0.001692`

### tick `15404`, seconds `26.00`, LSTM delta `-0.0775`

Top all feature movements:
- `lag_03__T_place_CONNECTOR`: contribution `-0.012270`
- `lag_00__T_A_site_active_infernos`: contribution `-0.008505`
- `lag_02__T5__duck_amount`: contribution `+0.005722`
- `lag_00__T_active_infernos`: contribution `-0.004164`
- `lag_05__T_place_PALACEINTERIOR`: contribution `+0.003778`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.008505`
- `lag_00__T_active_infernos`: contribution `-0.004164`
- `lag_00__active_infernos_total`: contribution `-0.001871`
- `lag_05__T_A_site_active_infernos`: contribution `-0.001852`
