# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `32050`, seconds `35.50`, LSTM `0.8043`, delta `+0.1661`
- tick `30482`, seconds `11.00`, LSTM `0.7881`, delta `+0.1418`
- tick `31986`, seconds `34.50`, LSTM `0.6699`, delta `-0.1148`
- tick `32402`, seconds `41.00`, LSTM `0.9570`, delta `+0.0808`
- tick `30738`, seconds `15.00`, LSTM `0.9109`, delta `+0.0688`
- tick `30514`, seconds `11.50`, LSTM `0.8403`, delta `+0.0522`
- tick `31890`, seconds `33.00`, LSTM `0.7969`, delta `-0.0441`
- tick `32018`, seconds `35.00`, LSTM `0.6382`, delta `-0.0318`
- tick `30450`, seconds `10.50`, LSTM `0.6463`, delta `-0.0293`
- tick `32210`, seconds `38.00`, LSTM `0.8639`, delta `+0.0282`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002143`, |coef| `0.002143`
- `lag_00__kill_diff_last_3s`: coefficient `0.001940`, |coef| `0.001940`
- `lag_02__CT2__is_scoped`: coefficient `-0.001879`, |coef| `0.001879`
- `lag_04__CT2__duck_amount`: coefficient `-0.001642`, |coef| `0.001642`
- `lag_05__CT2__is_scoped`: coefficient `0.001505`, |coef| `0.001505`
- `lag_00__CT_kills_last_3s`: coefficient `0.001466`, |coef| `0.001466`
- `lag_09__CT2__is_scoped`: coefficient `0.001449`, |coef| `0.001449`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001375`, |coef| `0.001375`
- `lag_11__CT2__is_scoped`: coefficient `-0.001360`, |coef| `0.001360`
- `lag_08__CT4__duck_amount`: coefficient `-0.001310`, |coef| `0.001310`
- `lag_14__CT_place_HOUSE`: coefficient `0.001275`, |coef| `0.001275`
- `lag_00__CT_damage_last_5s`: coefficient `0.001128`, |coef| `0.001128`
- `lag_05__CT2__duck_amount`: coefficient `0.001097`, |coef| `0.001097`
- `lag_00__CT1__shots_fired`: coefficient `0.001044`, |coef| `0.001044`
- `lag_10__T1__duck_amount`: coefficient `-0.001041`, |coef| `0.001041`

## Top 10 utility ridge features

- `lag_10__T2__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.000663` (lowers CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `-0.000606` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000543` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `-0.000526` (lowers CT win probability)
- `lag_11__CT_active_smokes`: coefficient `0.000512` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.000505` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000504` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000503` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000452` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.002143` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001940` (raises CT win probability)
- `lag_02__CT2__is_scoped`: coefficient `-0.001879` (lowers CT win probability)
- `lag_04__CT2__duck_amount`: coefficient `-0.001642` (lowers CT win probability)
- `lag_05__CT2__is_scoped`: coefficient `0.001505` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001466` (raises CT win probability)
- `lag_09__CT2__is_scoped`: coefficient `0.001449` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001375` (raises CT win probability)
- `lag_11__CT2__is_scoped`: coefficient `-0.001360` (lowers CT win probability)
- `lag_08__CT4__duck_amount`: coefficient `-0.001310` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `32050`, seconds `35.50`, LSTM delta `+0.1661`

Top all feature movements:
- `lag_02__CT2__is_scoped`: contribution `+0.011501`
- `lag_05__CT2__is_scoped`: contribution `+0.009213`
- `lag_11__CT2__is_scoped`: contribution `+0.008326`
- `lag_04__CT2__duck_amount`: contribution `+0.006257`
- `lag_00__damage_diff_last_5s`: contribution `+0.004834`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `+0.002178`

### tick `30482`, seconds `11.00`, LSTM delta `+0.1418`

Top all feature movements:
- `lag_14__CT_place_HOUSE`: contribution `+0.013513`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009553`
- `lag_12__CT_place_HOUSE`: contribution `+0.007026`
- `lag_00__kill_diff_last_3s`: contribution `+0.004669`
- `lag_00__CT_kills_last_3s`: contribution `+0.004233`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.002033`

### tick `31986`, seconds `34.50`, LSTM delta `-0.1148`

Top all feature movements:
- `lag_09__CT2__is_scoped`: contribution `-0.008870`
- `lag_03__CT2__is_scoped`: contribution `-0.006015`
- `lag_00__damage_diff_last_5s`: contribution `-0.004834`
- `lag_08__CT4__duck_amount`: contribution `-0.004812`
- `lag_00__kill_diff_last_3s`: contribution `-0.004669`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32402`, seconds `41.00`, LSTM delta `+0.0808`

Top all feature movements:
- `lag_13__CT2__is_scoped`: contribution `+0.006344`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004776`
- `lag_00__kill_diff_last_3s`: contribution `+0.004669`
- `lag_00__CT_kills_last_3s`: contribution `+0.004233`
- `lag_03__T3__duck_amount`: contribution `-0.003237`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30738`, seconds `15.00`, LSTM delta `+0.0688`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004669`
- `lag_00__CT_kills_last_3s`: contribution `+0.004233`
- `lag_12__T_place_TUNNEL`: contribution `+0.004195`
- `lag_01__T_place_WATER`: contribution `+0.003916`
- `lag_08__CT_shots_fired_sum`: contribution `+0.003175`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.002033`
- `lag_07__CT4__flash_duration`: contribution `-0.001782`
