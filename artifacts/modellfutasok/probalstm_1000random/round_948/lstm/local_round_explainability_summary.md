# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `109627`, seconds `75.00`, LSTM `0.6615`, delta `+0.3310`
- tick `110299`, seconds `85.50`, LSTM `0.4222`, delta `-0.2960`
- tick `109819`, seconds `78.00`, LSTM `0.8306`, delta `+0.2316`
- tick `109531`, seconds `73.50`, LSTM `0.5067`, delta `+0.2083`
- tick `110875`, seconds `94.50`, LSTM `0.6747`, delta `+0.2071`
- tick `109595`, seconds `74.50`, LSTM `0.3305`, delta `-0.1792`
- tick `111003`, seconds `96.50`, LSTM `0.5550`, delta `-0.1575`
- tick `109659`, seconds `75.50`, LSTM `0.5732`, delta `-0.0883`
- tick `109787`, seconds `77.50`, LSTM `0.5990`, delta `+0.0881`
- tick `105371`, seconds `8.50`, LSTM `0.1505`, delta `-0.0621`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004114`, |coef| `0.004114`
- `lag_02__CT_place_SHOP`: coefficient `0.003752`, |coef| `0.003752`
- `lag_00__CT_kills_last_3s`: coefficient `0.003039`, |coef| `0.003039`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002980`, |coef| `0.002980`
- `lag_08__CT_place_STAIRS`: coefficient `0.002661`, |coef| `0.002661`
- `lag_07__T_utility_damage_last_5s`: coefficient `0.002571`, |coef| `0.002571`
- `lag_15__CT_place_SHOP`: coefficient `0.002571`, |coef| `0.002571`
- `lag_00__damage_diff_last_5s`: coefficient `0.002522`, |coef| `0.002522`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002300`, |coef| `0.002300`
- `lag_04__CT_place_SHOP`: coefficient `0.002259`, |coef| `0.002259`
- `lag_15__CT_place_JUNGLE`: coefficient `0.002127`, |coef| `0.002127`
- `lag_00__T_kills_last_3s`: coefficient `-0.002081`, |coef| `0.002081`
- `lag_15__CT_place_STAIRS`: coefficient `-0.001912`, |coef| `0.001912`
- `lag_02__T_utility_damage_last_5s`: coefficient `0.001818`, |coef| `0.001818`
- `lag_03__T_shots_fired_sum`: coefficient `0.001796`, |coef| `0.001796`

## Top 10 utility ridge features

- `lag_07__T_utility_damage_last_5s`: coefficient `0.002571` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `0.001818` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.001629` (lowers CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `-0.001151` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.001125` (raises CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `0.001021` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.000982` (raises CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `0.000837` (raises CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000827` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.000824` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004114` (raises CT win probability)
- `lag_02__CT_place_SHOP`: coefficient `0.003752` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003039` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002980` (lowers CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `0.002661` (raises CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `0.002571` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002522` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002300` (raises CT win probability)
- `lag_04__CT_place_SHOP`: coefficient `0.002259` (raises CT win probability)
- `lag_15__CT_place_JUNGLE`: coefficient `0.002127` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `109627`, seconds `75.00`, LSTM delta `+0.3310`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.020108`
- `lag_02__CT_place_SHOP`: contribution `+0.018818`
- `lag_00__CT_shots_fired_sum`: contribution `+0.012784`
- `lag_04__T_shots_fired_sum`: contribution `+0.011248`
- `lag_00__kill_diff_last_3s`: contribution `+0.009903`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110299`, seconds `85.50`, LSTM delta `-0.2960`

Top all feature movements:
- `lag_08__CT_place_STAIRS`: contribution `-0.020708`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.020192`
- `lag_02__CT_place_SHOP`: contribution `-0.018818`
- `lag_15__CT_place_STAIRS`: contribution `-0.014881`
- `lag_15__CT_place_SHOP`: contribution `-0.012894`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.020192`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.008087`

### tick `109819`, seconds `78.00`, LSTM delta `+0.2316`

Top all feature movements:
- `lag_02__CT_place_SHOP`: contribution `+0.018818`
- `lag_02__T_utility_damage_last_5s`: contribution `+0.014276`
- `lag_00__CT_place_STAIRS`: contribution `+0.012841`
- `lag_11__CT_place_JUNGLE`: contribution `+0.010990`
- `lag_05__CT_shots_fired_sum`: contribution `+0.009383`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `+0.014276`
- `lag_02__utility_damage_diff_last_5s`: contribution `+0.005716`

### tick `109531`, seconds `73.50`, LSTM delta `+0.2083`

Top all feature movements:
- `lag_02__CT_place_JUNGLE`: contribution `+0.011237`
- `lag_00__kill_diff_last_3s`: contribution `+0.009903`
- `lag_00__CT_kills_last_3s`: contribution `+0.008774`
- `lag_01__T3__shots_fired`: contribution `+0.006723`
- `lag_00__damage_diff_last_5s`: contribution `+0.005689`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110875`, seconds `94.50`, LSTM delta `+0.2071`

Top all feature movements:
- `lag_02__CT_place_SHOP`: contribution `+0.018818`
- `lag_15__CT_place_JUNGLE`: contribution `+0.013646`
- `lag_04__CT_place_SHOP`: contribution `+0.011331`
- `lag_11__CT_place_JUNGLE`: contribution `+0.010990`
- `lag_00__kill_diff_last_3s`: contribution `+0.009903`

Top utility-only movements:
- No utility movement among the top local contributors.
