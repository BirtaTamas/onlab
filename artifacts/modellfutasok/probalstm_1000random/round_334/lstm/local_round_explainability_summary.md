# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `113536`, seconds `50.50`, LSTM `0.5455`, delta `+0.3674`
- tick `113248`, seconds `46.00`, LSTM `0.3104`, delta `-0.2827`
- tick `115328`, seconds `78.50`, LSTM `0.8385`, delta `+0.2668`
- tick `113984`, seconds `57.50`, LSTM `0.7257`, delta `+0.2031`
- tick `114336`, seconds `63.00`, LSTM `0.8511`, delta `+0.1919`
- tick `114880`, seconds `71.50`, LSTM `0.5815`, delta `-0.1516`
- tick `114656`, seconds `68.00`, LSTM `0.7313`, delta `-0.0819`
- tick `115360`, seconds `79.00`, LSTM `0.9196`, delta `+0.0811`
- tick `115232`, seconds `77.00`, LSTM `0.6633`, delta `+0.0744`
- tick `113280`, seconds `46.50`, LSTM `0.2412`, delta `-0.0692`

## Top 15 local ridge features

- `lag_11__CT_place_TUNNEL`: coefficient `0.003875`, |coef| `0.003875`
- `lag_03__CT_place_TUNNEL`: coefficient `-0.003758`, |coef| `0.003758`
- `lag_00__kill_diff_last_3s`: coefficient `0.003483`, |coef| `0.003483`
- `lag_00__CT_kills_last_3s`: coefficient `0.003011`, |coef| `0.003011`
- `lag_02__CT_place_TUNNEL`: coefficient `-0.002888`, |coef| `0.002888`
- `lag_14__CT_place_TSIDELOWER`: coefficient `0.002569`, |coef| `0.002569`
- `lag_00__damage_diff_last_5s`: coefficient `0.002366`, |coef| `0.002366`
- `lag_13__T_shots_fired_sum`: coefficient `-0.002198`, |coef| `0.002198`
- `lag_00__CT_place_TUNNEL`: coefficient `-0.001839`, |coef| `0.001839`
- `lag_11__T_utility_damage_last_5s`: coefficient `-0.001814`, |coef| `0.001814`
- `lag_13__T4__shots_fired`: coefficient `-0.001758`, |coef| `0.001758`
- `lag_00__CT_damage_last_5s`: coefficient `0.001654`, |coef| `0.001654`
- `lag_11__CT_place_WATER`: coefficient `-0.001609`, |coef| `0.001609`
- `lag_11__CT_place_TSIDELOWER`: coefficient `-0.001580`, |coef| `0.001580`
- `lag_14__CT_place_RUINS`: coefficient `-0.001561`, |coef| `0.001561`

## Top 10 utility ridge features

- `lag_11__T_utility_damage_last_5s`: coefficient `-0.001814` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001283` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.001277` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001151` (raises CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001143` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001054` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.000969` (lowers CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `0.000954` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `-0.000832` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.000812` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_TUNNEL`: coefficient `0.003875` (raises CT win probability)
- `lag_03__CT_place_TUNNEL`: coefficient `-0.003758` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003483` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003011` (raises CT win probability)
- `lag_02__CT_place_TUNNEL`: coefficient `-0.002888` (lowers CT win probability)
- `lag_14__CT_place_TSIDELOWER`: coefficient `0.002569` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002366` (raises CT win probability)
- `lag_13__T_shots_fired_sum`: coefficient `-0.002198` (lowers CT win probability)
- `lag_00__CT_place_TUNNEL`: coefficient `-0.001839` (lowers CT win probability)
- `lag_13__T4__shots_fired`: coefficient `-0.001758` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `113536`, seconds `50.50`, LSTM delta `+0.3674`

Top all feature movements:
- `lag_11__CT_place_TUNNEL`: contribution `+0.062238`
- `lag_03__CT_place_TUNNEL`: contribution `+0.060360`
- `lag_11__CT_place_TSIDELOWER`: contribution `+0.021462`
- `lag_11__CT_place_TSIDEUPPER`: contribution `+0.011237`
- `lag_11__CT_place_WATER`: contribution `+0.009775`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113248`, seconds `46.00`, LSTM delta `-0.2827`

Top all feature movements:
- `lag_02__CT_place_TUNNEL`: contribution `-0.046381`
- `lag_14__CT_place_TSIDELOWER`: contribution `-0.034895`
- `lag_02__CT_place_TSIDELOWER`: contribution `-0.018382`
- `lag_02__CT_place_TSIDEUPPER`: contribution `-0.010660`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.008883`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115328`, seconds `78.50`, LSTM delta `+0.2668`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `+0.016478`
- `lag_13__T4__shots_fired`: contribution `+0.010863`
- `lag_11__T_utility_damage_last_5s`: contribution `+0.010362`
- `lag_00__CT_kills_last_3s`: contribution `+0.008692`
- `lag_00__kill_diff_last_3s`: contribution `+0.008383`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `+0.010362`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.004158`

### tick `113984`, seconds `57.50`, LSTM delta `+0.2031`

Top all feature movements:
- `lag_11__CT_place_TUNNEL`: contribution `+0.062238`
- `lag_05__CT_place_TUNNEL`: contribution `+0.018942`
- `lag_00__CT_kills_last_3s`: contribution `+0.008692`
- `lag_00__kill_diff_last_3s`: contribution `+0.008383`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004887`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114336`, seconds `63.00`, LSTM delta `+0.1919`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008692`
- `lag_00__kill_diff_last_3s`: contribution `+0.008383`
- `lag_07__T3__flash_duration`: contribution `+0.007982`
- `lag_03__T_bomb_zone_count`: contribution `+0.006508`
- `lag_07__T2__flash_duration`: contribution `+0.006485`

Top utility-only movements:
- `lag_07__T3__flash_duration`: contribution `+0.007982`
- `lag_07__T2__flash_duration`: contribution `+0.006485`
- `lag_07__T4__flash_duration`: contribution `+0.005676`
- `lag_07__T_flash_duration_sum`: contribution `+0.004574`
