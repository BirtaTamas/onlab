# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `50773`, seconds `59.00`, LSTM `0.1496`, delta `-0.1499`
- tick `50293`, seconds `51.50`, LSTM `0.4058`, delta `-0.1202`
- tick `50357`, seconds `52.50`, LSTM `0.2534`, delta `-0.0809`
- tick `50901`, seconds `61.00`, LSTM `0.0279`, delta `-0.0735`
- tick `50325`, seconds `52.00`, LSTM `0.3343`, delta `-0.0714`
- tick `50261`, seconds `51.00`, LSTM `0.5260`, delta `-0.0430`
- tick `50709`, seconds `58.00`, LSTM `0.2843`, delta `+0.0396`
- tick `50805`, seconds `59.50`, LSTM `0.1179`, delta `-0.0317`
- tick `50485`, seconds `54.50`, LSTM `0.2461`, delta `+0.0303`
- tick `47509`, seconds `8.00`, LSTM `0.6393`, delta `-0.0221`

## Top 15 local ridge features

- `lag_15__T_place_RAMP`: coefficient `-0.002024`, |coef| `0.002024`
- `lag_12__T_place_RAMP`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_13__T_place_RAMP`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_00__T_place_RAMP`: coefficient `-0.001635`, |coef| `0.001635`
- `lag_01__T_place_RAMP`: coefficient `-0.001624`, |coef| `0.001624`
- `lag_00__T_kills_last_3s`: coefficient `-0.001420`, |coef| `0.001420`
- `lag_14__T_place_RAMP`: coefficient `-0.001416`, |coef| `0.001416`
- `lag_02__T_place_RAMP`: coefficient `-0.001405`, |coef| `0.001405`
- `lag_11__T_place_RAMP`: coefficient `-0.001251`, |coef| `0.001251`
- `lag_09__T_place_RAMP`: coefficient `-0.001214`, |coef| `0.001214`
- `lag_01__T_place_CONTROL`: coefficient `0.001133`, |coef| `0.001133`
- `lag_00__T_place_TROPHY`: coefficient `0.001131`, |coef| `0.001131`
- `lag_00__CT_place_ADMIN`: coefficient `0.001102`, |coef| `0.001102`
- `lag_00__kill_diff_last_3s`: coefficient `0.001079`, |coef| `0.001079`
- `lag_14__T_place_CONTROL`: coefficient `-0.001056`, |coef| `0.001056`

## Top 10 utility ridge features

- `lag_00__CT_smoke_inv`: coefficient `0.000647` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000646` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000540` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000540` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000523` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000514` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000508` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000485` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000456` (raises CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.000432` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_RAMP`: coefficient `-0.002024` (lowers CT win probability)
- `lag_12__T_place_RAMP`: coefficient `-0.001850` (lowers CT win probability)
- `lag_13__T_place_RAMP`: coefficient `-0.001708` (lowers CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.001635` (lowers CT win probability)
- `lag_01__T_place_RAMP`: coefficient `-0.001624` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001420` (lowers CT win probability)
- `lag_14__T_place_RAMP`: coefficient `-0.001416` (lowers CT win probability)
- `lag_02__T_place_RAMP`: coefficient `-0.001405` (lowers CT win probability)
- `lag_11__T_place_RAMP`: coefficient `-0.001251` (lowers CT win probability)
- `lag_09__T_place_RAMP`: coefficient `-0.001214` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `50773`, seconds `59.00`, LSTM delta `-0.1499`

Top all feature movements:
- `lag_13__T_place_CONTROL`: contribution `-0.007260`
- `lag_15__T_place_RAMP`: contribution `-0.007157`
- `lag_12__T_place_RAMP`: contribution `-0.006544`
- `lag_13__T_place_TROPHY`: contribution `-0.005358`
- `lag_15__CT_place_ADMIN`: contribution `-0.004788`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50293`, seconds `51.50`, LSTM delta `-0.1202`

Top all feature movements:
- `lag_01__T_place_CONTROL`: contribution `-0.008053`
- `lag_00__CT_place_ADMIN`: contribution `-0.007653`
- `lag_13__T_place_CONTROL`: contribution `-0.007260`
- `lag_08__T_place_TROPHY`: contribution `-0.006040`
- `lag_00__T_place_RAMP`: contribution `-0.005784`

Top utility-only movements:
- `lag_00__CT5__smoke`: contribution `-0.001417`

### tick `50357`, seconds `52.50`, LSTM delta `-0.0809`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `-0.007172`
- `lag_06__T_place_CONTROL`: contribution `-0.007048`
- `lag_02__T_place_CONTROL`: contribution `-0.005376`
- `lag_02__T_place_RAMP`: contribution `-0.004969`
- `lag_11__CT_place_GARAGE`: contribution `-0.004913`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50901`, seconds `61.00`, LSTM delta `-0.0735`

Top all feature movements:
- `lag_13__T_place_CONTROL`: contribution `+0.007260`
- `lag_13__T_place_RAMP`: contribution `-0.006040`
- `lag_00__T_kills_last_3s`: contribution `-0.004498`
- `lag_04__CT_place_HELL`: contribution `-0.002881`
- `lag_00__kill_diff_last_3s`: contribution `-0.002596`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `-0.001254`

### tick `50325`, seconds `52.00`, LSTM delta `-0.0714`

Top all feature movements:
- `lag_01__T_place_CONTROL`: contribution `-0.008053`
- `lag_14__T_place_CONTROL`: contribution `-0.007502`
- `lag_01__T_place_RAMP`: contribution `-0.005744`
- `lag_02__T_place_CONTROL`: contribution `-0.005376`
- `lag_02__T_place_RAMP`: contribution `-0.004969`

Top utility-only movements:
- No utility movement among the top local contributors.
