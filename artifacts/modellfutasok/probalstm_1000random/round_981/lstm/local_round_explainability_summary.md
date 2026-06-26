# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-b8-bo3--nzkpOWiS4qFgkFOwM8Hun/legacy-vs-b8-m2-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `34060`, seconds `0.50`, LSTM `0.0449`, delta `-0.0511`
- tick `36204`, seconds `34.00`, LSTM `0.0680`, delta `-0.0470`
- tick `36236`, seconds `34.50`, LSTM `0.0216`, delta `-0.0464`
- tick `35468`, seconds `22.50`, LSTM `0.1236`, delta `+0.0405`
- tick `36172`, seconds `33.50`, LSTM `0.1149`, delta `-0.0316`
- tick `36108`, seconds `32.50`, LSTM `0.1477`, delta `-0.0180`
- tick `35980`, seconds `30.50`, LSTM `0.1736`, delta `+0.0167`
- tick `34668`, seconds `10.00`, LSTM `0.0642`, delta `+0.0136`
- tick `36076`, seconds `32.00`, LSTM `0.1657`, delta `+0.0127`
- tick `34092`, seconds `1.00`, LSTM `0.0324`, delta `-0.0125`

## Top 15 local ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000987`, |coef| `0.000987`
- `lag_08__T_place_TSIDELOWER`: coefficient `0.000563`, |coef| `0.000563`
- `lag_08__CT_place_SIDEHALL`: coefficient `0.000549`, |coef| `0.000549`
- `lag_00__T5__has_bomb`: coefficient `0.000541`, |coef| `0.000541`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000539`, |coef| `0.000539`
- `lag_07__T_place_TSIDELOWER`: coefficient `0.000532`, |coef| `0.000532`
- `lag_13__T1__duck_amount`: coefficient `0.000491`, |coef| `0.000491`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.000479`, |coef| `0.000479`
- `lag_05__CT_place_SIDEHALL`: coefficient `0.000470`, |coef| `0.000470`
- `lag_07__CT_place_SIDEHALL`: coefficient `0.000466`, |coef| `0.000466`
- `lag_07__CT2__is_walking`: coefficient `-0.000442`, |coef| `0.000442`
- `lag_03__CT2__duck_amount`: coefficient `-0.000438`, |coef| `0.000438`
- `lag_14__T1__duck_amount`: coefficient `0.000438`, |coef| `0.000438`
- `lag_09__T_place_TSIDELOWER`: coefficient `0.000437`, |coef| `0.000437`
- `lag_00__T_kills_last_3s`: coefficient `-0.000437`, |coef| `0.000437`

## Top 10 utility ridge features

- `lag_00__T_B_site_active_infernos`: coefficient `-0.000423` (lowers CT win probability)
- `lag_08__T_active_smokes`: coefficient `0.000358` (raises CT win probability)
- `lag_02__T2__molly`: coefficient `0.000350` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000349` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000338` (lowers CT win probability)
- `lag_09__T_active_smokes`: coefficient `0.000325` (raises CT win probability)
- `lag_07__T_active_smokes`: coefficient `0.000315` (raises CT win probability)
- `lag_03__T2__molly`: coefficient `0.000302` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000302` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000290` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_UNKNOWN`: coefficient `-0.000987` (lowers CT win probability)
- `lag_08__T_place_TSIDELOWER`: coefficient `0.000563` (raises CT win probability)
- `lag_08__CT_place_SIDEHALL`: coefficient `0.000549` (raises CT win probability)
- `lag_00__T5__has_bomb`: coefficient `0.000541` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.000539` (raises CT win probability)
- `lag_07__T_place_TSIDELOWER`: coefficient `0.000532` (raises CT win probability)
- `lag_13__T1__duck_amount`: coefficient `0.000491` (raises CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.000479` (raises CT win probability)
- `lag_05__CT_place_SIDEHALL`: coefficient `0.000470` (raises CT win probability)
- `lag_07__CT_place_SIDEHALL`: coefficient `0.000466` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `34060`, seconds `0.50`, LSTM delta `-0.0511`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.034623`
- `lag_01__T5__has_bomb`: contribution `+0.000839`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000811`
- `lag_01__T_place_TSPAWN`: contribution `-0.000675`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000635`

Top utility-only movements:
- `lag_01__T2__molly`: contribution `+0.000412`
- `lag_01__flash_inv_diff`: contribution `-0.000388`
- `lag_01__utility_inv_diff`: contribution `-0.000351`
- `lag_01__smoke_inv_diff`: contribution `-0.000272`
- `lag_01__T3__utility_total`: contribution `-0.000230`

### tick `36204`, seconds `34.00`, LSTM delta `-0.0470`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `-0.002469`
- `lag_08__CT_place_SIDEHALL`: contribution `-0.002348`
- `lag_08__T_place_TSIDELOWER`: contribution `-0.002112`
- `lag_13__T1__duck_amount`: contribution `-0.001922`
- `lag_11__T5__shots_fired`: contribution `-0.001866`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `-0.001196`
- `lag_02__T2__molly`: contribution `-0.000779`

### tick `36236`, seconds `34.50`, LSTM delta `-0.0464`

Top all feature movements:
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.001926`
- `lag_09__CT_place_SIDEHALL`: contribution `-0.001839`
- `lag_14__T1__duck_amount`: contribution `-0.001714`
- `lag_09__T_place_TSIDELOWER`: contribution `-0.001639`
- `lag_00__T_kills_last_3s`: contribution `-0.001384`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `-0.000988`

### tick `35468`, seconds `22.50`, LSTM delta `+0.0405`

Top all feature movements:
- `lag_15__T_place_TUNNEL`: contribution `+0.002309`
- `lag_05__CT_place_SIDEHALL`: contribution `+0.002010`
- `lag_03__CT2__duck_amount`: contribution `+0.001670`
- `lag_00__T5__has_bomb`: contribution `+0.001586`
- `lag_09__T_place_WATER`: contribution `+0.001575`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36172`, seconds `33.50`, LSTM delta `-0.0316`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `-0.002550`
- `lag_10__T5__shots_fired`: contribution `-0.002277`
- `lag_07__T_place_TSIDELOWER`: contribution `-0.001995`
- `lag_07__CT_place_SIDEHALL`: contribution `-0.001993`
- `lag_00__T5__has_bomb`: contribution `-0.001586`

Top utility-only movements:
- `lag_15__T2__flash_duration`: contribution `-0.000835`
- `lag_07__T_active_smokes`: contribution `-0.000659`
