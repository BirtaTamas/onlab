# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `51745`, seconds `50.00`, LSTM `0.8527`, delta `+0.2910`
- tick `50657`, seconds `33.00`, LSTM `0.5399`, delta `+0.2466`
- tick `50977`, seconds `38.00`, LSTM `0.6947`, delta `+0.1234`
- tick `50945`, seconds `37.50`, LSTM `0.5714`, delta `-0.1071`
- tick `50721`, seconds `34.00`, LSTM `0.6323`, delta `+0.1045`
- tick `49921`, seconds `21.50`, LSTM `0.2229`, delta `-0.0947`
- tick `50369`, seconds `28.50`, LSTM `0.3613`, delta `-0.0768`
- tick `50529`, seconds `31.00`, LSTM `0.3340`, delta `+0.0751`
- tick `49409`, seconds `13.50`, LSTM `0.5209`, delta `-0.0707`
- tick `50785`, seconds `35.00`, LSTM `0.7279`, delta `+0.0616`

## Top 15 local ridge features

- `lag_00__T_place_WALKWAY`: coefficient `-0.002527`, |coef| `0.002527`
- `lag_04__T_place_WALKWAY`: coefficient `0.002288`, |coef| `0.002288`
- `lag_00__T_place_MAIN`: coefficient `-0.002171`, |coef| `0.002171`
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `-0.002098`, |coef| `0.002098`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001996`, |coef| `0.001996`
- `lag_00__kill_diff_last_3s`: coefficient `0.001981`, |coef| `0.001981`
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `0.001978`, |coef| `0.001978`
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `-0.001966`, |coef| `0.001966`
- `lag_04__CT_place_TUNNEL`: coefficient `-0.001881`, |coef| `0.001881`
- `lag_00__CT_kills_last_3s`: coefficient `0.001770`, |coef| `0.001770`
- `lag_09__CT_place_TUNNELSTAIRS`: coefficient `0.001690`, |coef| `0.001690`
- `lag_00__CT_place_HEAVEN`: coefficient `0.001543`, |coef| `0.001543`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `-0.001367`, |coef| `0.001367`
- `lag_00__CT3__duck_amount`: coefficient `0.001365`, |coef| `0.001365`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `0.001360`, |coef| `0.001360`

## Top 10 utility ridge features

- `lag_14__T_flashes_last_5s`: coefficient `0.001171` (raises CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.001152` (lowers CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `0.000936` (raises CT win probability)
- `lag_11__CT_active_infernos`: coefficient `-0.000913` (lowers CT win probability)
- `lag_11__active_infernos_total`: coefficient `-0.000839` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000836` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.000816` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.000792` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000781` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.000732` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_WALKWAY`: coefficient `-0.002527` (lowers CT win probability)
- `lag_04__T_place_WALKWAY`: coefficient `0.002288` (raises CT win probability)
- `lag_00__T_place_MAIN`: coefficient `-0.002171` (lowers CT win probability)
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `-0.002098` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001996` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001981` (raises CT win probability)
- `lag_15__CT_place_TUNNELSTAIRS`: coefficient `0.001978` (raises CT win probability)
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `-0.001966` (lowers CT win probability)
- `lag_04__CT_place_TUNNEL`: coefficient `-0.001881` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001770` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `51745`, seconds `50.00`, LSTM delta `+0.2910`

Top all feature movements:
- `lag_00__T_place_WALKWAY`: contribution `+0.034365`
- `lag_04__T_place_WALKWAY`: contribution `+0.031117`
- `lag_15__CT_place_TUNNELSTAIRS`: contribution `+0.027858`
- `lag_09__CT_place_TUNNELSTAIRS`: contribution `+0.023807`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.019251`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50657`, seconds `33.00`, LSTM delta `+0.2466`

Top all feature movements:
- `lag_04__CT_place_TUNNEL`: contribution `+0.030213`
- `lag_02__CT_place_TUNNELSTAIRS`: contribution `+0.027696`
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `+0.010640`
- `lag_15__T_place_MAIN`: contribution `+0.007699`
- `lag_09__CT_place_TUNNEL`: contribution `+0.006683`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50977`, seconds `38.00`, LSTM delta `+0.1234`

Top all feature movements:
- `lag_14__CT_place_TUNNEL`: contribution `+0.019311`
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `+0.010332`
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.009736`
- `lag_09__CT_shots_fired_sum`: contribution `+0.006161`
- `lag_00__CT_kills_last_3s`: contribution `+0.005109`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50945`, seconds `37.50`, LSTM delta `-0.1071`

Top all feature movements:
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `-0.019151`
- `lag_04__T_flashes_last_5s`: contribution `-0.010437`
- `lag_00__kill_diff_last_3s`: contribution `-0.004769`
- `lag_13__CT_place_TUNNEL`: contribution `-0.004469`
- `lag_08__CT_shots_fired_sum`: contribution `-0.004055`

Top utility-only movements:
- `lag_04__T_flashes_last_5s`: contribution `-0.010437`

### tick `50721`, seconds `34.00`, LSTM delta `+0.1045`

Top all feature movements:
- `lag_11__CT_place_TUNNEL`: contribution `+0.018951`
- `lag_00__T_place_MAIN`: contribution `+0.014033`
- `lag_04__CT_place_TUNNELSTAIRS`: contribution `-0.010640`
- `lag_04__CT_place_HEAVEN`: contribution `+0.006042`
- `lag_00__CT_kills_last_3s`: contribution `+0.005109`

Top utility-only movements:
- No utility movement among the top local contributors.
