# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `28746`, seconds `63.00`, LSTM `0.1781`, delta `-0.1558`
- tick `30986`, seconds `98.00`, LSTM `0.0136`, delta `-0.0844`
- tick `30954`, seconds `97.50`, LSTM `0.0981`, delta `+0.0722`
- tick `25450`, seconds `11.50`, LSTM `0.3397`, delta `+0.0638`
- tick `26858`, seconds `33.50`, LSTM `0.3721`, delta `-0.0569`
- tick `27946`, seconds `50.50`, LSTM `0.3982`, delta `+0.0550`
- tick `27882`, seconds `49.50`, LSTM `0.3196`, delta `+0.0490`
- tick `27370`, seconds `41.50`, LSTM `0.2245`, delta `-0.0443`
- tick `27306`, seconds `40.50`, LSTM `0.2715`, delta `-0.0439`
- tick `25066`, seconds `5.50`, LSTM `0.2474`, delta `-0.0438`

## Top 15 local ridge features

- `lag_10__T_place_VENTS`: coefficient `-0.002036`, |coef| `0.002036`
- `lag_11__T_place_VENTS`: coefficient `0.001728`, |coef| `0.001728`
- `lag_15__CT_place_VENDING`: coefficient `0.001670`, |coef| `0.001670`
- `lag_06__CT_place_HUT`: coefficient `-0.001593`, |coef| `0.001593`
- `lag_13__CT_place_TROPHY`: coefficient `0.001563`, |coef| `0.001563`
- `lag_06__CT_place_LOBBY`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__CT_place_HUT`: coefficient `0.001513`, |coef| `0.001513`
- `lag_00__kill_diff_last_3s`: coefficient `0.001173`, |coef| `0.001173`
- `lag_00__T_kills_last_3s`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_02__CT_place_SQUEAKY`: coefficient `-0.001001`, |coef| `0.001001`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.000966`, |coef| `0.000966`
- `lag_08__CT_place_HUT`: coefficient `-0.000914`, |coef| `0.000914`
- `lag_03__CT_place_VENDING`: coefficient `0.000886`, |coef| `0.000886`
- `lag_01__CT_place_CONTROL`: coefficient `0.000843`, |coef| `0.000843`
- `lag_10__CT_place_TROPHY`: coefficient `0.000837`, |coef| `0.000837`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_smokes`: coefficient `0.000578` (raises CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `0.000555` (raises CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `-0.000551` (lowers CT win probability)
- `lag_12__CT_he_last_5s`: coefficient `-0.000535` (lowers CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `0.000521` (raises CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `-0.000519` (lowers CT win probability)
- `lag_12__CT_smokes_last_5s`: coefficient `-0.000504` (lowers CT win probability)
- `lag_11__CT_he_last_5s`: coefficient `-0.000503` (lowers CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `0.000495` (raises CT win probability)
- `lag_11__CT_smokes_last_5s`: coefficient `-0.000474` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_VENTS`: coefficient `-0.002036` (lowers CT win probability)
- `lag_11__T_place_VENTS`: coefficient `0.001728` (raises CT win probability)
- `lag_15__CT_place_VENDING`: coefficient `0.001670` (raises CT win probability)
- `lag_06__CT_place_HUT`: coefficient `-0.001593` (lowers CT win probability)
- `lag_13__CT_place_TROPHY`: coefficient `0.001563` (raises CT win probability)
- `lag_06__CT_place_LOBBY`: coefficient `0.001532` (raises CT win probability)
- `lag_00__CT_place_HUT`: coefficient `0.001513` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001173` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001063` (lowers CT win probability)
- `lag_02__CT_place_SQUEAKY`: coefficient `-0.001001` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `28746`, seconds `63.00`, LSTM delta `-0.1558`

Top all feature movements:
- `lag_15__CT_place_VENDING`: contribution `-0.028616`
- `lag_13__CT_place_TROPHY`: contribution `-0.023090`
- `lag_06__CT_place_HUT`: contribution `-0.015540`
- `lag_00__CT_place_HUT`: contribution `-0.014759`
- `lag_06__CT_place_LOBBY`: contribution `-0.012538`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30986`, seconds `98.00`, LSTM delta `-0.0844`

Top all feature movements:
- `lag_11__T_place_VENTS`: contribution `-0.023308`
- `lag_00__T_kills_last_3s`: contribution `-0.003368`
- `lag_00__kill_diff_last_3s`: contribution `-0.002825`
- `lag_02__CT3__duck_amount`: contribution `-0.002526`
- `lag_06__CT3__duck_amount`: contribution `-0.002181`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30954`, seconds `97.50`, LSTM delta `+0.0722`

Top all feature movements:
- `lag_10__T_place_VENTS`: contribution `+0.027460`
- `lag_00__kill_diff_last_3s`: contribution `+0.002825`
- `lag_14__T1__duck_amount`: contribution `+0.002432`
- `lag_13__T5__duck_amount`: contribution `+0.002187`
- `lag_15__T1__duck_amount`: contribution `+0.002176`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `25450`, seconds `11.50`, LSTM delta `+0.0638`

Top all feature movements:
- `lag_12__CT_he_last_5s`: contribution `+0.009809`
- `lag_12__CT_smokes_last_5s`: contribution `+0.008706`
- `lag_12__CT_place_HELL`: contribution `+0.007676`
- `lag_06__CT_place_ADMIN`: contribution `+0.005097`
- `lag_09__CT_place_ADMIN`: contribution `+0.003592`

Top utility-only movements:
- `lag_12__CT_he_last_5s`: contribution `+0.009809`
- `lag_12__CT_smokes_last_5s`: contribution `+0.008706`
- `lag_01__CT4__flash_duration`: contribution `+0.002659`
- `lag_00__CT_A_site_active_smokes`: contribution `+0.000930`

### tick `26858`, seconds `33.50`, LSTM delta `-0.0569`

Top all feature movements:
- `lag_15__CT_place_ADMIN`: contribution `-0.005549`
- `lag_09__CT_place_HUTROOF`: contribution `-0.004616`
- `lag_00__CT5__duck_amount`: contribution `-0.003157`
- `lag_15__T_place_SECRET`: contribution `-0.002866`
- `lag_00__CT2__duck_amount`: contribution `+0.002161`

Top utility-only movements:
- No utility movement among the top local contributors.
