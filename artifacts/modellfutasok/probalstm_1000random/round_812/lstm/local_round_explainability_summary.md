# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `10257`, seconds `55.00`, LSTM `0.1079`, delta `-0.1645`
- tick `9841`, seconds `48.50`, LSTM `0.3301`, delta `-0.1031`
- tick `9809`, seconds `48.00`, LSTM `0.4332`, delta `-0.0823`
- tick `7409`, seconds `10.50`, LSTM `0.3835`, delta `+0.0607`
- tick `9873`, seconds `49.00`, LSTM `0.2714`, delta `-0.0587`
- tick `7153`, seconds `6.50`, LSTM `0.2677`, delta `+0.0572`
- tick `11089`, seconds `68.00`, LSTM `0.0332`, delta `-0.0558`
- tick `7377`, seconds `10.00`, LSTM `0.3228`, delta `+0.0525`
- tick `9969`, seconds `50.50`, LSTM `0.3334`, delta `+0.0479`
- tick `6769`, seconds `0.50`, LSTM `0.2938`, delta `-0.0455`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002119`, |coef| `0.002119`
- `lag_00__T_place_ROOF`: coefficient `-0.001473`, |coef| `0.001473`
- `lag_01__T_place_SECRET`: coefficient `0.001404`, |coef| `0.001404`
- `lag_06__T_place_TROPHY`: coefficient `0.001153`, |coef| `0.001153`
- `lag_02__T_place_SECRET`: coefficient `0.001133`, |coef| `0.001133`
- `lag_10__CT_place_CONTROL`: coefficient `-0.001098`, |coef| `0.001098`
- `lag_00__T4__shots_fired`: coefficient `-0.001044`, |coef| `0.001044`
- `lag_07__T_place_TROPHY`: coefficient `0.000982`, |coef| `0.000982`
- `lag_00__CT_place_HELL`: coefficient `-0.000927`, |coef| `0.000927`
- `lag_02__CT5__is_walking`: coefficient `0.000915`, |coef| `0.000915`
- `lag_03__CT_place_RAMP`: coefficient `0.000845`, |coef| `0.000845`
- `lag_01__CT_place_HEAVEN`: coefficient `-0.000832`, |coef| `0.000832`
- `lag_10__T4__shots_fired`: coefficient `0.000804`, |coef| `0.000804`
- `lag_10__T_place_SECRET`: coefficient `0.000799`, |coef| `0.000799`
- `lag_15__T_place_TROPHY`: coefficient `-0.000791`, |coef| `0.000791`

## Top 10 utility ridge features

- `lag_08__CT1__smoke`: coefficient `0.000616` (raises CT win probability)
- `lag_03__T5__molly`: coefficient `0.000609` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000606` (lowers CT win probability)
- `lag_07__CT1__smoke`: coefficient `0.000593` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000571` (lowers CT win probability)
- `lag_01__T5__smoke`: coefficient `-0.000569` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000549` (lowers CT win probability)
- `lag_13__CT1__smoke`: coefficient `0.000547` (raises CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `-0.000536` (lowers CT win probability)
- `lag_02__T5__molly`: coefficient `0.000521` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.002119` (lowers CT win probability)
- `lag_00__T_place_ROOF`: coefficient `-0.001473` (lowers CT win probability)
- `lag_01__T_place_SECRET`: coefficient `0.001404` (raises CT win probability)
- `lag_06__T_place_TROPHY`: coefficient `0.001153` (raises CT win probability)
- `lag_02__T_place_SECRET`: coefficient `0.001133` (raises CT win probability)
- `lag_10__CT_place_CONTROL`: coefficient `-0.001098` (lowers CT win probability)
- `lag_00__T4__shots_fired`: coefficient `-0.001044` (lowers CT win probability)
- `lag_07__T_place_TROPHY`: coefficient `0.000982` (raises CT win probability)
- `lag_00__CT_place_HELL`: coefficient `-0.000927` (lowers CT win probability)
- `lag_02__CT5__is_walking`: coefficient `0.000915` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `10257`, seconds `55.00`, LSTM delta `-0.1645`

Top all feature movements:
- `lag_10__T4__shots_fired`: contribution `-0.014402`
- `lag_10__CT_place_CONTROL`: contribution `-0.011402`
- `lag_09__T5__shots_fired`: contribution `-0.010413`
- `lag_00__T_place_ROOF`: contribution `-0.008344`
- `lag_00__T_shots_fired_sum`: contribution `-0.007944`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9841`, seconds `48.50`, LSTM delta `-0.1031`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.015888`
- `lag_07__T_place_TROPHY`: contribution `-0.006224`
- `lag_05__T_place_TROPHY`: contribution `-0.004627`
- `lag_01__T_shots_fired_sum`: contribution `-0.004553`
- `lag_03__T_place_SQUEAKY`: contribution `-0.003309`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.001803`
- `lag_01__T_B_site_active_infernos`: contribution `-0.001614`
- `lag_03__T5__molly`: contribution `-0.001347`
- `lag_08__CT1__smoke`: contribution `-0.001335`

### tick `9809`, seconds `48.00`, LSTM delta `-0.0823`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.012710`
- `lag_06__T_place_TROPHY`: contribution `-0.007314`
- `lag_15__T_place_TROPHY`: contribution `-0.005018`
- `lag_04__T_place_TROPHY`: contribution `-0.004460`
- `lag_00__T4__shots_fired`: contribution `-0.003871`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.001325`
- `lag_07__CT1__smoke`: contribution `-0.001284`

### tick `7409`, seconds `10.50`, LSTM delta `+0.0607`

Top all feature movements:
- `lag_00__T_place_ROOF`: contribution `+0.008344`
- `lag_07__CT_place_HELL`: contribution `+0.007446`
- `lag_08__CT_place_HELL`: contribution `+0.005219`
- `lag_04__CT_place_RAMP`: contribution `+0.004416`
- `lag_11__CT_place_HELL`: contribution `+0.004399`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9873`, seconds `49.00`, LSTM delta `-0.0587`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.019065`
- `lag_06__T_place_TROPHY`: contribution `-0.007314`
- `lag_01__T_shots_fired_sum`: contribution `-0.005692`
- `lag_00__T4__shots_fired`: contribution `-0.003871`
- `lag_04__T_place_SQUEAKY`: contribution `-0.003191`

Top utility-only movements:
- No utility movement among the top local contributors.
