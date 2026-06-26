# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `22`

## Largest probability jumps

- tick `200398`, seconds `86.50`, LSTM `0.8912`, delta `+0.2525`
- tick `199950`, seconds `79.50`, LSTM `0.4324`, delta `+0.2164`
- tick `199406`, seconds `71.00`, LSTM `0.3799`, delta `-0.1495`
- tick `199982`, seconds `80.00`, LSTM `0.5638`, delta `+0.1313`
- tick `198030`, seconds `49.50`, LSTM `0.6623`, delta `+0.0636`
- tick `199150`, seconds `67.00`, LSTM `0.5910`, delta `-0.0611`
- tick `199438`, seconds `71.50`, LSTM `0.3305`, delta `-0.0494`
- tick `200622`, seconds `90.00`, LSTM `0.9671`, delta `+0.0430`
- tick `200046`, seconds `81.00`, LSTM `0.6358`, delta `+0.0418`
- tick `199278`, seconds `69.00`, LSTM `0.5964`, delta `-0.0407`

## Top 15 local ridge features

- `lag_00__T_place_RESTROOM`: coefficient `-0.003623`, |coef| `0.003623`
- `lag_14__T_place_RESTROOM`: coefficient `-0.002513`, |coef| `0.002513`
- `lag_01__T_place_RESTROOM`: coefficient `-0.002008`, |coef| `0.002008`
- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.001877`, |coef| `0.001877`
- `lag_00__kill_diff_last_3s`: coefficient `0.001797`, |coef| `0.001797`
- `lag_06__CT_place_STORAGEROOM`: coefficient `-0.001777`, |coef| `0.001777`
- `lag_00__CT_kills_last_3s`: coefficient `0.001669`, |coef| `0.001669`
- `lag_12__CT_place_WATER`: coefficient `0.001494`, |coef| `0.001494`
- `lag_08__CT_place_STORAGEROOM`: coefficient `-0.001397`, |coef| `0.001397`
- `lag_01__T_place_ALLEY`: coefficient `-0.001367`, |coef| `0.001367`
- `lag_00__damage_diff_last_5s`: coefficient `0.001355`, |coef| `0.001355`
- `lag_00__CT_damage_last_5s`: coefficient `0.001290`, |coef| `0.001290`
- `lag_04__T_place_RESTROOM`: coefficient `-0.001216`, |coef| `0.001216`
- `lag_07__T_place_TSTAIRS`: coefficient `-0.001215`, |coef| `0.001215`
- `lag_12__CT_place_SNIPERSNEST`: coefficient `-0.001214`, |coef| `0.001214`

## Top 10 utility ridge features

- `lag_00__T2__molly`: coefficient `-0.000654` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000644` (lowers CT win probability)
- `lag_14__T3__smoke`: coefficient `-0.000620` (lowers CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000595` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `-0.000509` (lowers CT win probability)
- `lag_01__T3__smoke`: coefficient `-0.000429` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000411` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `-0.000405` (lowers CT win probability)
- `lag_08__T_mollies_last_5s`: coefficient `-0.000404` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_RESTROOM`: coefficient `-0.003623` (lowers CT win probability)
- `lag_14__T_place_RESTROOM`: coefficient `-0.002513` (lowers CT win probability)
- `lag_01__T_place_RESTROOM`: coefficient `-0.002008` (lowers CT win probability)
- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.001877` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001797` (raises CT win probability)
- `lag_06__CT_place_STORAGEROOM`: coefficient `-0.001777` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001669` (raises CT win probability)
- `lag_12__CT_place_WATER`: coefficient `0.001494` (raises CT win probability)
- `lag_08__CT_place_STORAGEROOM`: coefficient `-0.001397` (lowers CT win probability)
- `lag_01__T_place_ALLEY`: coefficient `-0.001367` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `200398`, seconds `86.50`, LSTM delta `+0.2525`

Top all feature movements:
- `lag_14__T_place_RESTROOM`: contribution `+0.048481`
- `lag_12__CT_place_WATER`: contribution `+0.009080`
- `lag_07__T_place_TSTAIRS`: contribution `+0.006887`
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.006501`
- `lag_10__T_place_TSTAIRS`: contribution `+0.005998`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `199950`, seconds `79.50`, LSTM delta `+0.2164`

Top all feature movements:
- `lag_00__T_place_RESTROOM`: contribution `+0.069885`
- `lag_05__CT_place_STORAGEROOM`: contribution `+0.040160`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.005206`
- `lag_00__CT_kills_last_3s`: contribution `+0.004818`
- `lag_00__kill_diff_last_3s`: contribution `+0.004325`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `199406`, seconds `71.00`, LSTM delta `-0.1495`

Top all feature movements:
- `lag_08__CT_place_STORAGEROOM`: contribution `-0.029884`
- `lag_04__T_place_RESTROOM`: contribution `-0.023448`
- `lag_08__CT_place_BACKOFA`: contribution `-0.010049`
- `lag_12__CT_place_STAIRS`: contribution `-0.009189`
- `lag_08__CT_place_CANAL`: contribution `-0.004358`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `199982`, seconds `80.00`, LSTM delta `+0.1313`

Top all feature movements:
- `lag_01__T_place_RESTROOM`: contribution `+0.038742`
- `lag_06__CT_place_STORAGEROOM`: contribution `+0.038007`
- `lag_09__T_place_FOUNTAIN`: contribution `+0.003652`
- `lag_06__CT_place_SNIPERSNEST`: contribution `+0.003232`
- `lag_12__T_kills_last_3s`: contribution `+0.003022`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `198030`, seconds `49.50`, LSTM delta `+0.0636`

Top all feature movements:
- `lag_12__T_place_PLAYGROUND`: contribution `+0.013319`
- `lag_00__CT_kills_last_3s`: contribution `+0.004818`
- `lag_10__T_place_PIPE`: contribution `+0.004435`
- `lag_00__kill_diff_last_3s`: contribution `+0.004325`
- `lag_05__T_flashes_last_5s`: contribution `+0.003401`

Top utility-only movements:
- `lag_05__T_flashes_last_5s`: contribution `+0.003401`
