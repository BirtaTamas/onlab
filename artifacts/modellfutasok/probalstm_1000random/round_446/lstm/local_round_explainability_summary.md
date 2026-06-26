# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-inner-circle-bo3-YbhHiIk4CcU9clhSbtidF_/spirit-vs-inner-circle-m1-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `78739`, seconds `45.00`, LSTM `0.7008`, delta `+0.3170`
- tick `79507`, seconds `57.00`, LSTM `0.9088`, delta `+0.2521`
- tick `77651`, seconds `28.00`, LSTM `0.2463`, delta `-0.2462`
- tick `78995`, seconds `49.00`, LSTM `0.8964`, delta `+0.2246`
- tick `78675`, seconds `44.00`, LSTM `0.3067`, delta `+0.2051`
- tick `79347`, seconds `54.50`, LSTM `0.7210`, delta `-0.1806`
- tick `78707`, seconds `44.50`, LSTM `0.3839`, delta `+0.0772`
- tick `78931`, seconds `48.00`, LSTM `0.6358`, delta `-0.0756`
- tick `77043`, seconds `18.50`, LSTM `0.5646`, delta `+0.0637`
- tick `76851`, seconds `15.50`, LSTM `0.5414`, delta `-0.0620`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004048`, |coef| `0.004048`
- `lag_00__T_place_RAMP`: coefficient `-0.003450`, |coef| `0.003450`
- `lag_00__CT_kills_last_3s`: coefficient `0.003360`, |coef| `0.003360`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002736`, |coef| `0.002736`
- `lag_00__damage_diff_last_5s`: coefficient `0.002735`, |coef| `0.002735`
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.002122`, |coef| `0.002122`
- `lag_00__CT_damage_last_5s`: coefficient `0.002013`, |coef| `0.002013`
- `lag_15__T_place_TUNNEL`: coefficient `-0.001965`, |coef| `0.001965`
- `lag_02__T3__has_bomb`: coefficient `-0.001945`, |coef| `0.001945`
- `lag_14__T_place_SIDEENTRANCE`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_03__T3__has_bomb`: coefficient `-0.001876`, |coef| `0.001876`
- `lag_10__CT_he_last_5s`: coefficient `0.001853`, |coef| `0.001853`
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001849`, |coef| `0.001849`
- `lag_13__T_place_TUNNEL`: coefficient `-0.001834`, |coef| `0.001834`
- `lag_14__T1__flash_duration`: coefficient `0.001804`, |coef| `0.001804`

## Top 10 utility ridge features

- `lag_10__CT_he_last_5s`: coefficient `0.001853` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001849` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `0.001804` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `0.001716` (raises CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.001715` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `0.001386` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.001323` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001276` (raises CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `0.001253` (raises CT win probability)
- `lag_09__CT_he_last_5s`: coefficient `0.001211` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004048` (raises CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.003450` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003360` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002736` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002735` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.002122` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002013` (raises CT win probability)
- `lag_15__T_place_TUNNEL`: coefficient `-0.001965` (lowers CT win probability)
- `lag_02__T3__has_bomb`: coefficient `-0.001945` (lowers CT win probability)
- `lag_14__T_place_SIDEENTRANCE`: coefficient `-0.001876` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78739`, seconds `45.00`, LSTM delta `+0.3170`

Top all feature movements:
- `lag_08__CT5__flash_duration`: contribution `+0.012258`
- `lag_00__T_place_RAMP`: contribution `+0.012200`
- `lag_15__T_place_TUNNEL`: contribution `+0.011934`
- `lag_00__kill_diff_last_3s`: contribution `+0.009743`
- `lag_00__CT_kills_last_3s`: contribution `+0.009700`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `+0.012258`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.008977`

### tick `79507`, seconds `57.00`, LSTM delta `+0.2521`

Top all feature movements:
- `lag_14__T1__flash_duration`: contribution `+0.013221`
- `lag_14__CT5__flash_duration`: contribution `+0.011235`
- `lag_00__kill_diff_last_3s`: contribution `+0.009743`
- `lag_00__CT_kills_last_3s`: contribution `+0.009700`
- `lag_14__T_place_SIDEENTRANCE`: contribution `+0.009158`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `+0.013221`
- `lag_14__CT5__flash_duration`: contribution `+0.011235`
- `lag_01__T1__flash_duration`: contribution `+0.008677`
- `lag_14__CT_flash_duration_sum`: contribution `+0.006844`
- `lag_14__CT3__flash_duration`: contribution `+0.006578`

### tick `77651`, seconds `28.00`, LSTM delta `-0.2462`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.020511`
- `lag_09__T_place_TUNNEL`: contribution `-0.010709`
- `lag_00__kill_diff_last_3s`: contribution `-0.009743`
- `lag_03__T_place_TUNNEL`: contribution `-0.009224`
- `lag_03__T_place_WATER`: contribution `-0.008183`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.003741`

### tick `78995`, seconds `49.00`, LSTM delta `+0.2246`

Top all feature movements:
- `lag_00__T_place_RAMP`: contribution `+0.012200`
- `lag_02__CT_place_MAINHALL`: contribution `+0.010097`
- `lag_00__kill_diff_last_3s`: contribution `+0.009743`
- `lag_00__CT_kills_last_3s`: contribution `+0.009700`
- `lag_13__T_utility_damage_last_5s`: contribution `+0.005436`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.005436`
- `lag_03__CT5__flash_duration`: contribution `+0.004928`
- `lag_14__T_B_site_active_infernos`: contribution `+0.003741`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.003036`

### tick `78675`, seconds `44.00`, LSTM delta `+0.2051`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.016409`
- `lag_00__T_place_RAMP`: contribution `+0.012200`
- `lag_13__T_place_TUNNEL`: contribution `+0.011139`
- `lag_00__kill_diff_last_3s`: contribution `+0.009743`
- `lag_00__CT_kills_last_3s`: contribution `+0.009700`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.009120`
- `lag_03__T_utility_damage_last_5s`: contribution `+0.003036`
- `lag_04__T_B_site_active_infernos`: contribution `+0.002829`
