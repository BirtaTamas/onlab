# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `4378`, seconds `48.50`, LSTM `0.7917`, delta `+0.2053`
- tick `5754`, seconds `70.00`, LSTM `0.8752`, delta `-0.0826`
- tick `5498`, seconds `66.00`, LSTM `0.9196`, delta `+0.0739`
- tick `4570`, seconds `51.50`, LSTM `0.9518`, delta `+0.0652`
- tick `2938`, seconds `26.00`, LSTM `0.5684`, delta `-0.0643`
- tick `4410`, seconds `49.00`, LSTM `0.8529`, delta `+0.0612`
- tick `6074`, seconds `75.00`, LSTM `0.8865`, delta `+0.0569`
- tick `2746`, seconds `23.00`, LSTM `0.5480`, delta `+0.0514`
- tick `5306`, seconds `63.00`, LSTM `0.8530`, delta `-0.0347`
- tick `5786`, seconds `70.50`, LSTM `0.8430`, delta `-0.0322`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002015`, |coef| `0.002015`
- `lag_00__damage_diff_last_5s`: coefficient `0.001933`, |coef| `0.001933`
- `lag_04__CT_place_MIDDOORS`: coefficient `0.001858`, |coef| `0.001858`
- `lag_03__CT_place_MIDDOORS`: coefficient `0.001855`, |coef| `0.001855`
- `lag_00__kill_diff_last_3s`: coefficient `0.001820`, |coef| `0.001820`
- `lag_05__CT_place_MIDDOORS`: coefficient `0.001806`, |coef| `0.001806`
- `lag_00__CT_damage_last_5s`: coefficient `0.001787`, |coef| `0.001787`
- `lag_14__CT5__duck_amount`: coefficient `0.001552`, |coef| `0.001552`
- `lag_00__T3__alive`: coefficient `-0.001521`, |coef| `0.001521`
- `lag_00__T3__hp`: coefficient `-0.001498`, |coef| `0.001498`
- `lag_00__T3__armor`: coefficient `-0.001453`, |coef| `0.001453`
- `lag_15__CT5__duck_amount`: coefficient `0.001445`, |coef| `0.001445`
- `lag_03__CT5__duck_amount`: coefficient `-0.001343`, |coef| `0.001343`
- `lag_02__CT_place_MIDDOORS`: coefficient `0.001335`, |coef| `0.001335`
- `lag_12__T2__smoke`: coefficient `-0.001315`, |coef| `0.001315`

## Top 10 utility ridge features

- `lag_12__T2__smoke`: coefficient `-0.001315` (lowers CT win probability)
- `lag_07__T3__smoke`: coefficient `0.001155` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.001058` (raises CT win probability)
- `lag_03__T3__smoke`: coefficient `-0.000998` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.000931` (raises CT win probability)
- `lag_13__T2__smoke`: coefficient `-0.000902` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000899` (lowers CT win probability)
- `lag_11__T2__smoke`: coefficient `-0.000779` (lowers CT win probability)
- `lag_08__T3__smoke`: coefficient `0.000763` (raises CT win probability)
- `lag_01__T2__flash`: coefficient `-0.000721` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.002015` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001933` (raises CT win probability)
- `lag_04__CT_place_MIDDOORS`: coefficient `0.001858` (raises CT win probability)
- `lag_03__CT_place_MIDDOORS`: coefficient `0.001855` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001820` (raises CT win probability)
- `lag_05__CT_place_MIDDOORS`: coefficient `0.001806` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001787` (raises CT win probability)
- `lag_14__CT5__duck_amount`: coefficient `0.001552` (raises CT win probability)
- `lag_00__T3__alive`: coefficient `-0.001521` (lowers CT win probability)
- `lag_00__T3__hp`: coefficient `-0.001498` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `4378`, seconds `48.50`, LSTM delta `+0.2053`

Top all feature movements:
- `lag_06__CT_place_BDOORS`: contribution `+0.006198`
- `lag_00__CT_kills_last_3s`: contribution `+0.005816`
- `lag_03__CT_place_MIDDOORS`: contribution `+0.005355`
- `lag_05__CT_place_MIDDOORS`: contribution `+0.005213`
- `lag_03__CT5__duck_amount`: contribution `+0.005068`

Top utility-only movements:
- `lag_12__T2__smoke`: contribution `+0.002887`

### tick `5754`, seconds `70.00`, LSTM delta `-0.0826`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `-0.011994`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.006146`
- `lag_07__CT_place_HOLE`: contribution `-0.005644`
- `lag_14__CT_place_HOLE`: contribution `-0.005445`
- `lag_05__CT_place_MIDDOORS`: contribution `+0.005213`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.001781`
- `lag_08__T2__flash_duration`: contribution `-0.001761`

### tick `5498`, seconds `66.00`, LSTM delta `+0.0739`

Top all feature movements:
- `lag_10__CT_place_OUTSIDELONG`: contribution `+0.010919`
- `lag_06__CT_place_HOLE`: contribution `+0.006226`
- `lag_06__CT_place_BDOORS`: contribution `-0.006198`
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.006146`
- `lag_00__CT_kills_last_3s`: contribution `+0.005816`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.004265`
- `lag_00__T2__flash_duration`: contribution `+0.003704`
- `lag_00__T_flash_alpha_mean`: contribution `+0.001809`

### tick `4570`, seconds `51.50`, LSTM delta `+0.0652`

Top all feature movements:
- `lag_03__T3__flash_duration`: contribution `+0.006746`
- `lag_11__CT5__duck_amount`: contribution `-0.004536`
- `lag_00__damage_diff_last_5s`: contribution `+0.004361`
- `lag_03__CT5__flash_duration`: contribution `+0.003993`
- `lag_00__CT_damage_last_5s`: contribution `+0.003896`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `+0.006746`
- `lag_03__CT5__flash_duration`: contribution `+0.003993`

### tick `2938`, seconds `26.00`, LSTM delta `-0.0643`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.005816`
- `lag_00__kill_diff_last_3s`: contribution `-0.004381`
- `lag_15__CT_place_PIT`: contribution `-0.002812`
- `lag_00__CT_place_MIDDOORS`: contribution `-0.002764`
- `lag_11__T_flashed_players`: contribution `-0.002435`

Top utility-only movements:
- No utility movement among the top local contributors.
