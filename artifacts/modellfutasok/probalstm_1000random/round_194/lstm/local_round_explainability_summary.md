# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `75798`, seconds `25.00`, LSTM `0.8972`, delta `+0.2475`
- tick `75638`, seconds `22.50`, LSTM `0.6120`, delta `+0.1451`
- tick `75318`, seconds `17.50`, LSTM `0.4082`, delta `-0.1183`
- tick `77238`, seconds `47.50`, LSTM `0.9356`, delta `+0.1065`
- tick `77334`, seconds `49.00`, LSTM `0.8551`, delta `-0.0996`
- tick `79478`, seconds `82.50`, LSTM `0.8438`, delta `+0.0757`
- tick `75926`, seconds `27.00`, LSTM `0.8465`, delta `-0.0747`
- tick `75542`, seconds `21.00`, LSTM `0.4370`, delta `+0.0546`
- tick `75606`, seconds `22.00`, LSTM `0.4670`, delta `+0.0534`
- tick `77206`, seconds `47.00`, LSTM `0.8291`, delta `+0.0507`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003013`, |coef| `0.003013`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002669`, |coef| `0.002669`
- `lag_00__damage_diff_last_5s`: coefficient `0.002637`, |coef| `0.002637`
- `lag_00__CT_kills_last_3s`: coefficient `0.002590`, |coef| `0.002590`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002218`, |coef| `0.002218`
- `lag_00__CT_damage_last_5s`: coefficient `0.001947`, |coef| `0.001947`
- `lag_00__CT4__duck_amount`: coefficient `0.001866`, |coef| `0.001866`
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_13__T_velocity_mean`: coefficient `-0.001536`, |coef| `0.001536`
- `lag_07__CT3__duck_amount`: coefficient `0.001491`, |coef| `0.001491`
- `lag_03__CT_place_BANANA`: coefficient `0.001490`, |coef| `0.001490`
- `lag_00__bomb_events_last_5s`: coefficient `0.001487`, |coef| `0.001487`
- `lag_12__T_velocity_mean`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_03__T5__duck_amount`: coefficient `0.001456`, |coef| `0.001456`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.001408`, |coef| `0.001408`

## Top 10 utility ridge features

- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001676` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.001225` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001208` (lowers CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `0.001089` (raises CT win probability)
- `lag_02__CT4__molly`: coefficient `-0.001054` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000961` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000940` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `0.000928` (raises CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.000928` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `-0.000897` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003013` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002669` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002637` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002590` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.002218` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001947` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.001866` (raises CT win probability)
- `lag_13__T_velocity_mean`: coefficient `-0.001536` (lowers CT win probability)
- `lag_07__CT3__duck_amount`: coefficient `0.001491` (raises CT win probability)
- `lag_03__CT_place_BANANA`: coefficient `0.001490` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `75798`, seconds `25.00`, LSTM delta `+0.2475`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.014955`
- `lag_00__kill_diff_last_3s`: contribution `+0.014504`
- `lag_06__T_utility_damage_last_5s`: contribution `+0.011542`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009273`
- `lag_00__damage_diff_last_5s`: contribution `+0.007733`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `+0.011542`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.006831`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.005471`
- `lag_15__T_utility_damage_last_5s`: contribution `+0.005034`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.004666`

### tick `75638`, seconds `22.50`, LSTM delta `+0.1451`

Top all feature movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.015796`
- `lag_00__damage_diff_last_5s`: contribution `+0.007733`
- `lag_00__CT_kills_last_3s`: contribution `+0.007477`
- `lag_00__kill_diff_last_3s`: contribution `+0.007252`
- `lag_00__CT4__duck_amount`: contribution `+0.006855`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.015796`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.006553`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.005601`
- `lag_11__T_utility_damage_last_5s`: contribution `+0.005547`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.005033`

### tick `75318`, seconds `17.50`, LSTM delta `-0.1183`

Top all feature movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.015796`
- `lag_00__kill_diff_last_3s`: contribution `-0.007252`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.006553`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005601`
- `lag_15__T2__flash_duration`: contribution `-0.004892`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `-0.015796`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.006553`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.005601`
- `lag_15__T2__flash_duration`: contribution `-0.004892`
- `lag_15__T_flash_duration_sum`: contribution `-0.003308`

### tick `77238`, seconds `47.50`, LSTM delta `+0.1065`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.007477`
- `lag_00__kill_diff_last_3s`: contribution `+0.007252`
- `lag_03__CT_place_BANANA`: contribution `+0.004410`
- `lag_00__T_place_UNDERPASS`: contribution `+0.004408`
- `lag_00__T5__has_bomb`: contribution `+0.003591`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77334`, seconds `49.00`, LSTM delta `-0.0996`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.007252`
- `lag_00__damage_diff_last_5s`: contribution `-0.005948`
- `lag_07__CT3__duck_amount`: contribution `-0.005547`
- `lag_03__T5__duck_amount`: contribution `-0.005530`
- `lag_03__T_place_UNDERPASS`: contribution `-0.004611`

Top utility-only movements:
- `lag_00__CT5__utility_total`: contribution `-0.001532`
