# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `101590`, seconds `66.00`, LSTM `0.5319`, delta `-0.2960`
- tick `98582`, seconds `19.00`, LSTM `0.7157`, delta `+0.2198`
- tick `98646`, seconds `20.00`, LSTM `0.9117`, delta `+0.1935`
- tick `98518`, seconds `18.00`, LSTM `0.4890`, delta `+0.1776`
- tick `98454`, seconds `17.00`, LSTM `0.3931`, delta `-0.1320`
- tick `100726`, seconds `52.50`, LSTM `0.8398`, delta `-0.0923`
- tick `98486`, seconds `17.50`, LSTM `0.3113`, delta `-0.0818`
- tick `98390`, seconds `16.00`, LSTM `0.5424`, delta `-0.0627`
- tick `102294`, seconds `77.00`, LSTM `0.4937`, delta `-0.0530`
- tick `98294`, seconds `14.50`, LSTM `0.6168`, delta `+0.0503`

## Top 15 local ridge features

- `lag_11__CT_place_ADMIN`: coefficient `0.004419`, |coef| `0.004419`
- `lag_00__kill_diff_last_3s`: coefficient `0.003791`, |coef| `0.003791`
- `lag_00__T_kills_last_3s`: coefficient `-0.003581`, |coef| `0.003581`
- `lag_00__CT_place_RAMP`: coefficient `0.003263`, |coef| `0.003263`
- `lag_00__damage_diff_last_5s`: coefficient `0.003259`, |coef| `0.003259`
- `lag_10__CT_place_ADMIN`: coefficient `0.002898`, |coef| `0.002898`
- `lag_12__T_place_TROPHY`: coefficient `0.002641`, |coef| `0.002641`
- `lag_00__T_damage_last_5s`: coefficient `-0.002570`, |coef| `0.002570`
- `lag_00__CT4__alive`: coefficient `0.002469`, |coef| `0.002469`
- `lag_10__T_place_VENDING`: coefficient `-0.002342`, |coef| `0.002342`
- `lag_00__CT4__hp`: coefficient `0.002306`, |coef| `0.002306`
- `lag_00__CT4__armor`: coefficient `0.002220`, |coef| `0.002220`
- `lag_00__T_place_CONTROL`: coefficient `-0.002103`, |coef| `0.002103`
- `lag_13__T_place_VENDING`: coefficient `0.002020`, |coef| `0.002020`
- `lag_02__CT4__is_walking`: coefficient `0.001990`, |coef| `0.001990`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `0.000834` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.000800` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `0.000716` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000696` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000626` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000601` (raises CT win probability)
- `lag_02__CT5__smoke`: coefficient `0.000562` (raises CT win probability)
- `lag_07__CT5__smoke`: coefficient `0.000512` (raises CT win probability)
- `lag_09__CT5__smoke`: coefficient `0.000507` (raises CT win probability)
- `lag_08__CT5__smoke`: coefficient `0.000495` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_ADMIN`: coefficient `0.004419` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003791` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003581` (lowers CT win probability)
- `lag_00__CT_place_RAMP`: coefficient `0.003263` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003259` (raises CT win probability)
- `lag_10__CT_place_ADMIN`: coefficient `0.002898` (raises CT win probability)
- `lag_12__T_place_TROPHY`: coefficient `0.002641` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002570` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.002469` (raises CT win probability)
- `lag_10__T_place_VENDING`: coefficient `-0.002342` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `101590`, seconds `66.00`, LSTM delta `-0.2960`

Top all feature movements:
- `lag_11__CT_place_ADMIN`: contribution `-0.030697`
- `lag_12__T_place_TROPHY`: contribution `-0.016751`
- `lag_00__T_kills_last_3s`: contribution `-0.011345`
- `lag_12__T_place_CONTROL`: contribution `-0.010236`
- `lag_00__CT_place_RAMP`: contribution `-0.009750`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `98582`, seconds `19.00`, LSTM delta `+0.2198`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.014944`
- `lag_01__CT_place_GARAGE`: contribution `+0.012391`
- `lag_10__T_place_VENDING`: contribution `+0.011873`
- `lag_02__T_place_CONTROL`: contribution `+0.011153`
- `lag_13__T_place_VENDING`: contribution `+0.010240`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `98646`, seconds `20.00`, LSTM delta `+0.1935`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.018248`
- `lag_12__T_place_TROPHY`: contribution `+0.016751`
- `lag_00__T_place_CONTROL`: contribution `+0.014944`
- `lag_10__T_place_VENDING`: contribution `+0.011873`
- `lag_00__T_kills_last_3s`: contribution `+0.011345`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `98518`, seconds `18.00`, LSTM delta `+0.1776`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.014944`
- `lag_10__T_place_VENDING`: contribution `+0.011873`
- `lag_02__T_place_CONTROL`: contribution `-0.011153`
- `lag_13__T_place_VENDING`: contribution `+0.010240`
- `lag_15__T_place_TROPHY`: contribution `+0.009737`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `98454`, seconds `17.00`, LSTM delta `-0.1320`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `-0.014944`
- `lag_10__T_place_VENDING`: contribution `-0.011873`
- `lag_00__T_kills_last_3s`: contribution `-0.011345`
- `lag_02__T_place_CONTROL`: contribution `-0.011153`
- `lag_13__T_place_VENDING`: contribution `-0.010240`

Top utility-only movements:
- No utility movement among the top local contributors.
