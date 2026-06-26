# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `117785`, seconds `40.50`, LSTM `0.7564`, delta `+0.2050`
- tick `117433`, seconds `35.00`, LSTM `0.4634`, delta `+0.1026`
- tick `117881`, seconds `42.00`, LSTM `0.9321`, delta `+0.0793`
- tick `117721`, seconds `39.50`, LSTM `0.4798`, delta `+0.0785`
- tick `117273`, seconds `32.50`, LSTM `0.4671`, delta `-0.0749`
- tick `117753`, seconds `40.00`, LSTM `0.5514`, delta `+0.0716`
- tick `117657`, seconds `38.50`, LSTM `0.4170`, delta `-0.0539`
- tick `117817`, seconds `41.00`, LSTM `0.8072`, delta `+0.0508`
- tick `116569`, seconds `21.50`, LSTM `0.5428`, delta `+0.0497`
- tick `116601`, seconds `22.00`, LSTM `0.5887`, delta `+0.0459`

## Top 15 local ridge features

- `lag_11__CT_place_HUTROOF`: coefficient `-0.002583`, |coef| `0.002583`
- `lag_14__T_place_RAMP`: coefficient `-0.002385`, |coef| `0.002385`
- `lag_11__CT_place_HUT`: coefficient `0.002072`, |coef| `0.002072`
- `lag_00__kill_diff_last_3s`: coefficient `0.001851`, |coef| `0.001851`
- `lag_00__damage_diff_last_5s`: coefficient `0.001783`, |coef| `0.001783`
- `lag_00__CT_kills_last_3s`: coefficient `0.001723`, |coef| `0.001723`
- `lag_02__T_bomb_zone_count`: coefficient `-0.001716`, |coef| `0.001716`
- `lag_13__T_place_RAMP`: coefficient `-0.001661`, |coef| `0.001661`
- `lag_15__T_place_RAMP`: coefficient `-0.001648`, |coef| `0.001648`
- `lag_08__CT_place_HUT`: coefficient `-0.001443`, |coef| `0.001443`
- `lag_10__CT_place_HUT`: coefficient `-0.001380`, |coef| `0.001380`
- `lag_01__kill_diff_last_3s`: coefficient `0.001376`, |coef| `0.001376`
- `lag_01__CT_kills_last_3s`: coefficient `0.001373`, |coef| `0.001373`
- `lag_05__T_bomb_zone_count`: coefficient `0.001347`, |coef| `0.001347`
- `lag_00__CT_damage_last_5s`: coefficient `0.001322`, |coef| `0.001322`

## Top 10 utility ridge features

- `lag_09__T3__smoke`: coefficient `-0.000304` (lowers CT win probability)
- `lag_10__T3__smoke`: coefficient `-0.000274` (lowers CT win probability)
- `lag_11__T3__smoke`: coefficient `-0.000266` (lowers CT win probability)
- `lag_08__T3__smoke`: coefficient `-0.000262` (lowers CT win probability)
- `lag_02__T3__flash`: coefficient `-0.000218` (lowers CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `-0.000209` (lowers CT win probability)
- `lag_07__T_A_site_active_smokes`: coefficient `-0.000198` (lowers CT win probability)
- `lag_04__T3__flash`: coefficient `-0.000197` (lowers CT win probability)
- `lag_03__T3__flash`: coefficient `-0.000196` (lowers CT win probability)
- `lag_02__T3__utility_total`: coefficient `-0.000195` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_HUTROOF`: coefficient `-0.002583` (lowers CT win probability)
- `lag_14__T_place_RAMP`: coefficient `-0.002385` (lowers CT win probability)
- `lag_11__CT_place_HUT`: coefficient `0.002072` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001851` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001783` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001723` (raises CT win probability)
- `lag_02__T_bomb_zone_count`: coefficient `-0.001716` (lowers CT win probability)
- `lag_13__T_place_RAMP`: coefficient `-0.001661` (lowers CT win probability)
- `lag_15__T_place_RAMP`: coefficient `-0.001648` (lowers CT win probability)
- `lag_08__CT_place_HUT`: coefficient `-0.001443` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `117785`, seconds `40.50`, LSTM delta `+0.2050`

Top all feature movements:
- `lag_11__CT_place_HUT`: contribution `+0.020206`
- `lag_11__CT_place_HUTROOF`: contribution `+0.018075`
- `lag_10__CT_place_HUT`: contribution `+0.013454`
- `lag_02__T_bomb_zone_count`: contribution `+0.009987`
- `lag_14__T_place_RAMP`: contribution `+0.008437`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117433`, seconds `35.00`, LSTM delta `+0.1026`

Top all feature movements:
- `lag_05__CT_place_VENTS`: contribution `+0.010744`
- `lag_00__CT_place_HUT`: contribution `+0.009792`
- `lag_00__CT_place_HUTROOF`: contribution `+0.006723`
- `lag_08__CT_place_MINI`: contribution `+0.005961`
- `lag_15__T_place_RAMP`: contribution `-0.005828`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117881`, seconds `42.00`, LSTM delta `+0.0793`

Top all feature movements:
- `lag_02__CT_place_OBSERVATION`: contribution `+0.013338`
- `lag_05__T_bomb_zone_count`: contribution `-0.007841`
- `lag_07__CT_place_RAFTERS`: contribution `+0.005433`
- `lag_14__CT_place_HUTROOF`: contribution `+0.005152`
- `lag_00__CT_kills_last_3s`: contribution `+0.004973`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117721`, seconds `39.50`, LSTM delta `+0.0785`

Top all feature movements:
- `lag_08__CT_place_HUT`: contribution `+0.014070`
- `lag_14__CT_place_VENTS`: contribution `+0.008790`
- `lag_00__T_bomb_zone_count`: contribution `+0.006044`
- `lag_13__T_place_RAMP`: contribution `+0.005873`
- `lag_09__CT_place_HUTROOF`: contribution `+0.004874`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117273`, seconds `32.50`, LSTM delta `-0.0749`

Top all feature movements:
- `lag_14__T_place_RAMP`: contribution `-0.008437`
- `lag_14__T_place_TROPHY`: contribution `-0.006048`
- `lag_10__T_place_CONTROL`: contribution `-0.005972`
- `lag_13__T_place_RAMP`: contribution `-0.005873`
- `lag_13__T_place_CONTROL`: contribution `-0.004976`

Top utility-only movements:
- No utility movement among the top local contributors.
