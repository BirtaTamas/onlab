# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-m80-vs-flyquest-bo3-ji2oWF2IQJDeDBfGP8d4J9/m80-vs-flyquest-m2-dust2.csv`
- round_num: `21`

## Largest probability jumps

- tick `180633`, seconds `69.50`, LSTM `0.4544`, delta `-0.3694`
- tick `180313`, seconds `64.50`, LSTM `0.7820`, delta `+0.2455`
- tick `181209`, seconds `78.50`, LSTM `0.0738`, delta `-0.2303`
- tick `177369`, seconds `18.50`, LSTM `0.5778`, delta `-0.1538`
- tick `180921`, seconds `74.00`, LSTM `0.2330`, delta `-0.1234`
- tick `176697`, seconds `8.00`, LSTM `0.7579`, delta `+0.0448`
- tick `177209`, seconds `16.00`, LSTM `0.7336`, delta `-0.0421`
- tick `180409`, seconds `66.00`, LSTM `0.8300`, delta `+0.0389`
- tick `181561`, seconds `84.00`, LSTM `0.0454`, delta `+0.0351`
- tick `177529`, seconds `21.00`, LSTM `0.5688`, delta `-0.0322`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.004599`, |coef| `0.004599`
- `lag_00__damage_diff_last_5s`: coefficient `0.004300`, |coef| `0.004300`
- `lag_00__kill_diff_last_3s`: coefficient `0.003685`, |coef| `0.003685`
- `lag_12__T_place_LONGA`: coefficient `-0.003088`, |coef| `0.003088`
- `lag_07__T_place_EXTENDEDA`: coefficient `-0.003059`, |coef| `0.003059`
- `lag_15__T2__duck_amount`: coefficient `0.002970`, |coef| `0.002970`
- `lag_06__CT_damage_last_5s`: coefficient `0.002866`, |coef| `0.002866`
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.002815`, |coef| `0.002815`
- `lag_00__CT_damage_last_5s`: coefficient `0.002609`, |coef| `0.002609`
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.002326`, |coef| `0.002326`
- `lag_13__T4__duck_amount`: coefficient `-0.002301`, |coef| `0.002301`
- `lag_00__CT3__utility_total`: coefficient `0.002242`, |coef| `0.002242`
- `lag_13__T2__duck_amount`: coefficient `0.002236`, |coef| `0.002236`
- `lag_03__CT_flashed_players`: coefficient `-0.002120`, |coef| `0.002120`
- `lag_00__CT_place_LONGA`: coefficient `0.002100`, |coef| `0.002100`

## Top 10 utility ridge features

- `lag_00__CT3__utility_total`: coefficient `0.002242` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001934` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001732` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.001701` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001699` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001445` (raises CT win probability)
- `lag_03__T2__flash_duration`: coefficient `-0.001214` (lowers CT win probability)
- `lag_03__T2__molly`: coefficient `-0.001189` (lowers CT win probability)
- `lag_06__T1__smoke`: coefficient `-0.001079` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.001039` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.004599` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004300` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003685` (raises CT win probability)
- `lag_12__T_place_LONGA`: coefficient `-0.003088` (lowers CT win probability)
- `lag_07__T_place_EXTENDEDA`: coefficient `-0.003059` (lowers CT win probability)
- `lag_15__T2__duck_amount`: coefficient `0.002970` (raises CT win probability)
- `lag_06__CT_damage_last_5s`: coefficient `0.002866` (raises CT win probability)
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.002815` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002609` (raises CT win probability)
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.002326` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `180633`, seconds `69.50`, LSTM delta `-0.3694`

Top all feature movements:
- `lag_00__damage_diff_last_5s`: contribution `-0.016781`
- `lag_00__T_kills_last_3s`: contribution `-0.014570`
- `lag_03__T_place_SHORTSTAIRS`: contribution `-0.011829`
- `lag_15__T2__duck_amount`: contribution `-0.011357`
- `lag_03__CT_flashed_players`: contribution `-0.009285`

Top utility-only movements:
- `lag_00__CT3__utility_total`: contribution `-0.006420`
- `lag_03__CT5__flash_duration`: contribution `-0.005927`

### tick `180313`, seconds `64.50`, LSTM delta `+0.2455`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `+0.014570`
- `lag_00__damage_diff_last_5s`: contribution `+0.009700`
- `lag_00__kill_diff_last_3s`: contribution `+0.008869`
- `lag_13__T2__duck_amount`: contribution `+0.008550`
- `lag_09__T_place_LOWERTUNNEL`: contribution `+0.007617`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `181209`, seconds `78.50`, LSTM delta `-0.2303`

Top all feature movements:
- `lag_12__T_place_LONGA`: contribution `-0.026311`
- `lag_07__T_place_EXTENDEDA`: contribution `-0.015166`
- `lag_00__T_kills_last_3s`: contribution `-0.014570`
- `lag_07__T_place_SHORTSTAIRS`: contribution `-0.009776`
- `lag_00__kill_diff_last_3s`: contribution `-0.008869`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `-0.005936`

### tick `177369`, seconds `18.50`, LSTM delta `-0.1538`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.014570`
- `lag_06__T_flashes_last_5s`: contribution `-0.009035`
- `lag_00__kill_diff_last_3s`: contribution `-0.008869`
- `lag_00__damage_diff_last_5s`: contribution `-0.006790`
- `lag_02__CT5__duck_amount`: contribution `-0.005775`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.009035`
- `lag_00__CT4__flash_duration`: contribution `-0.002517`
- `lag_13__CT4__flash_duration`: contribution `-0.002483`

### tick `180921`, seconds `74.00`, LSTM delta `-0.1234`

Top all feature movements:
- `lag_03__T_place_LONGA`: contribution `-0.016478`
- `lag_07__T_place_EXTENDEDA`: contribution `-0.015166`
- `lag_03__T_place_SHORTSTAIRS`: contribution `-0.011829`
- `lag_07__T_place_SHORTSTAIRS`: contribution `-0.009776`
- `lag_06__CT_place_BDOORS`: contribution `+0.009172`

Top utility-only movements:
- No utility movement among the top local contributors.
