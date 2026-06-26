# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `135803`, seconds `65.50`, LSTM `0.7511`, delta `+0.2297`
- tick `134139`, seconds `39.50`, LSTM `0.9097`, delta `+0.1467`
- tick `135835`, seconds `66.00`, LSTM `0.8920`, delta `+0.1408`
- tick `134491`, seconds `45.00`, LSTM `0.7262`, delta `-0.1200`
- tick `135771`, seconds `65.00`, LSTM `0.5214`, delta `-0.0897`
- tick `134459`, seconds `44.50`, LSTM `0.8461`, delta `-0.0485`
- tick `133467`, seconds `29.00`, LSTM `0.7353`, delta `+0.0480`
- tick `135867`, seconds `66.50`, LSTM `0.9398`, delta `+0.0478`
- tick `135611`, seconds `62.50`, LSTM `0.6125`, delta `-0.0466`
- tick `134523`, seconds `45.50`, LSTM `0.6881`, delta `-0.0381`

## Top 15 local ridge features

- `lag_10__T_place_QUAD`: coefficient `0.003368`, |coef| `0.003368`
- `lag_11__T_place_QUAD`: coefficient `0.001898`, |coef| `0.001898`
- `lag_06__T_place_QUAD`: coefficient `-0.001700`, |coef| `0.001700`
- `lag_00__kill_diff_last_3s`: coefficient `0.001606`, |coef| `0.001606`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001385`, |coef| `0.001385`
- `lag_00__CT_kills_last_3s`: coefficient `0.001370`, |coef| `0.001370`
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.001263`, |coef| `0.001263`
- `lag_15__CT_place_LIBRARY`: coefficient `-0.001138`, |coef| `0.001138`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001117`, |coef| `0.001117`
- `lag_14__CT_place_LIBRARY`: coefficient `-0.001102`, |coef| `0.001102`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001045`, |coef| `0.001045`
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_00__T_place_TRAMP`: coefficient `0.001016`, |coef| `0.001016`
- `lag_00__damage_diff_last_5s`: coefficient `0.000996`, |coef| `0.000996`
- `lag_07__T_place_QUAD`: coefficient `-0.000940`, |coef| `0.000940`

## Top 10 utility ridge features

- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.001263` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.001034` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000873` (lowers CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.000853` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000676` (raises CT win probability)
- `lag_15__CT3__flash_duration`: coefficient `-0.000622` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.000569` (lowers CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000566` (lowers CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.000546` (raises CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.000545` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_QUAD`: coefficient `0.003368` (raises CT win probability)
- `lag_11__T_place_QUAD`: coefficient `0.001898` (raises CT win probability)
- `lag_06__T_place_QUAD`: coefficient `-0.001700` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001606` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001385` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001370` (raises CT win probability)
- `lag_15__CT_place_LIBRARY`: coefficient `-0.001138` (lowers CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.001117` (raises CT win probability)
- `lag_14__CT_place_LIBRARY`: coefficient `-0.001102` (lowers CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001045` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `135803`, seconds `65.50`, LSTM delta `+0.2297`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `+0.081129`
- `lag_06__T_place_QUAD`: contribution `+0.040958`
- `lag_14__CT_place_LIBRARY`: contribution `+0.007064`
- `lag_02__CT_flashed_players`: contribution `+0.005332`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004811`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `134139`, seconds `39.50`, LSTM delta `+0.1467`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.011546`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.009594`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.006443`
- `lag_00__CT_kills_last_3s`: contribution `+0.003956`
- `lag_00__kill_diff_last_3s`: contribution `+0.003866`

Top utility-only movements:
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.009594`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.006443`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.002929`

### tick `135835`, seconds `66.00`, LSTM delta `+0.1408`

Top all feature movements:
- `lag_11__T_place_QUAD`: contribution `+0.045717`
- `lag_07__T_place_QUAD`: contribution `+0.022637`
- `lag_00__T_place_BALCONY`: contribution `+0.008669`
- `lag_15__CT_place_LIBRARY`: contribution `+0.007297`
- `lag_00__CT5__flash_duration`: contribution `+0.004847`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.004847`
- `lag_00__CT2__flash_duration`: contribution `+0.003805`

### tick `134491`, seconds `45.00`, LSTM delta `-0.1200`

Top all feature movements:
- `lag_10__CT_shots_fired_sum`: contribution `-0.005546`
- `lag_04__T_shots_fired_sum`: contribution `-0.005433`
- `lag_00__kill_diff_last_3s`: contribution `-0.003866`
- `lag_04__T5__shots_fired`: contribution `-0.003788`
- `lag_06__CT_B_site_active_infernos`: contribution `-0.002929`

Top utility-only movements:
- `lag_06__CT_B_site_active_infernos`: contribution `-0.002929`

### tick `135771`, seconds `65.00`, LSTM delta `-0.0897`

Top all feature movements:
- `lag_09__T_place_QUAD`: contribution `-0.018479`
- `lag_05__T_place_QUAD`: contribution `-0.011743`
- `lag_01__CT_flashed_players`: contribution `-0.004476`
- `lag_00__kill_diff_last_3s`: contribution `-0.003866`
- `lag_00__T_shots_fired_sum`: contribution `-0.003047`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.001953`
