# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `8`

## Largest probability jumps

- tick `46946`, seconds `24.50`, LSTM `0.8064`, delta `+0.1520`
- tick `46338`, seconds `15.00`, LSTM `0.6029`, delta `-0.1441`
- tick `48226`, seconds `44.50`, LSTM `0.9550`, delta `+0.1378`
- tick `46402`, seconds `16.00`, LSTM `0.5233`, delta `-0.1164`
- tick `46306`, seconds `14.50`, LSTM `0.7470`, delta `+0.1154`
- tick `46114`, seconds `11.50`, LSTM `0.5910`, delta `-0.1049`
- tick `46914`, seconds `24.00`, LSTM `0.6544`, delta `+0.0768`
- tick `47234`, seconds `29.00`, LSTM `0.7913`, delta `-0.0433`
- tick `49506`, seconds `64.50`, LSTM `0.9123`, delta `+0.0419`
- tick `49410`, seconds `63.00`, LSTM `0.8920`, delta `-0.0418`

## Top 15 local ridge features

- `lag_00__CT_place_TROPHY`: coefficient `0.002593`, |coef| `0.002593`
- `lag_12__CT_place_SQUEAKY`: coefficient `0.002005`, |coef| `0.002005`
- `lag_14__CT_place_ADMIN`: coefficient `-0.001599`, |coef| `0.001599`
- `lag_02__CT_place_TROPHY`: coefficient `0.001268`, |coef| `0.001268`
- `lag_11__CT_place_SQUEAKY`: coefficient `0.001235`, |coef| `0.001235`
- `lag_09__CT_place_CONTROL`: coefficient `-0.001228`, |coef| `0.001228`
- `lag_07__CT_place_HEAVEN`: coefficient `0.001226`, |coef| `0.001226`
- `lag_07__CT_place_HUT`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001189`, |coef| `0.001189`
- `lag_14__CT1__is_walking`: coefficient `0.001186`, |coef| `0.001186`
- `lag_09__CT_place_RAFTERS`: coefficient `0.001163`, |coef| `0.001163`
- `lag_03__CT_place_LOBBY`: coefficient `-0.001162`, |coef| `0.001162`
- `lag_00__CT1__is_walking`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_00__kill_diff_last_3s`: coefficient `0.001114`, |coef| `0.001114`
- `lag_00__T_place_VENDING`: coefficient `-0.001087`, |coef| `0.001087`

## Top 10 utility ridge features

- `lag_13__T_smokes_last_5s`: coefficient `-0.000894` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000779` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `-0.000773` (lowers CT win probability)
- `lag_14__T_smokes_last_5s`: coefficient `0.000747` (raises CT win probability)
- `lag_07__T_smokes_last_5s`: coefficient `0.000713` (raises CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.000708` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000673` (lowers CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `-0.000616` (lowers CT win probability)
- `lag_11__T_smokes_last_5s`: coefficient `-0.000610` (lowers CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `-0.000597` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TROPHY`: coefficient `0.002593` (raises CT win probability)
- `lag_12__CT_place_SQUEAKY`: coefficient `0.002005` (raises CT win probability)
- `lag_14__CT_place_ADMIN`: coefficient `-0.001599` (lowers CT win probability)
- `lag_02__CT_place_TROPHY`: coefficient `0.001268` (raises CT win probability)
- `lag_11__CT_place_SQUEAKY`: coefficient `0.001235` (raises CT win probability)
- `lag_09__CT_place_CONTROL`: coefficient `-0.001228` (lowers CT win probability)
- `lag_07__CT_place_HEAVEN`: coefficient `0.001226` (raises CT win probability)
- `lag_07__CT_place_HUT`: coefficient `-0.001217` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001189` (raises CT win probability)
- `lag_14__CT1__is_walking`: coefficient `0.001186` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46946`, seconds `24.50`, LSTM delta `+0.1520`

Top all feature movements:
- `lag_12__CT_place_SQUEAKY`: contribution `+0.026662`
- `lag_14__CT_place_ADMIN`: contribution `+0.011110`
- `lag_15__CT_place_CONTROL`: contribution `+0.010577`
- `lag_00__T_place_VENDING`: contribution `+0.005509`
- `lag_14__CT_place_HELL`: contribution `+0.004388`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.002728`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.002678`
- `lag_12__T_A_site_active_infernos`: contribution `+0.002108`

### tick `46338`, seconds `15.00`, LSTM delta `-0.1441`

Top all feature movements:
- `lag_07__CT_place_HUT`: contribution `-0.023738`
- `lag_03__CT_place_LOBBY`: contribution `-0.019028`
- `lag_03__CT_place_HUT`: contribution `-0.017634`
- `lag_14__T_smokes_last_5s`: contribution `-0.010951`
- `lag_07__CT_place_RAFTERS`: contribution `+0.007165`

Top utility-only movements:
- `lag_14__T_smokes_last_5s`: contribution `-0.010951`

### tick `48226`, seconds `44.50`, LSTM delta `+0.1378`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `+0.038294`
- `lag_07__CT_place_HEAVEN`: contribution `+0.006617`
- `lag_10__CT_place_HEAVEN`: contribution `+0.005799`
- `lag_03__T_place_GARAGE`: contribution `+0.004783`
- `lag_10__CT_place_RAFTERS`: contribution `+0.004591`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46402`, seconds `16.00`, LSTM delta `-0.1164`

Top all feature movements:
- `lag_05__CT_place_HUT`: contribution `-0.015634`
- `lag_09__CT_place_HUT`: contribution `-0.013427`
- `lag_09__CT_place_RAFTERS`: contribution `-0.012425`
- `lag_05__CT_place_LOBBY`: contribution `-0.009785`
- `lag_00__CT_place_LOBBY`: contribution `-0.006262`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `-0.001891`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001620`

### tick `46306`, seconds `14.50`, LSTM delta `+0.1154`

Top all feature movements:
- `lag_06__CT_place_HUT`: contribution `+0.014561`
- `lag_13__T_smokes_last_5s`: contribution `+0.013114`
- `lag_02__CT_place_LOBBY`: contribution `+0.012153`
- `lag_02__CT_place_HUT`: contribution `+0.008417`
- `lag_06__CT_place_RAFTERS`: contribution `+0.007619`

Top utility-only movements:
- `lag_13__T_smokes_last_5s`: contribution `+0.013114`
