# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `22664`, seconds `68.00`, LSTM `0.3788`, delta `-0.3834`
- tick `22568`, seconds `66.50`, LSTM `0.6121`, delta `+0.2614`
- tick `20424`, seconds `33.00`, LSTM `0.1359`, delta `-0.2475`
- tick `22856`, seconds `71.00`, LSTM `0.0535`, delta `-0.1402`
- tick `21736`, seconds `53.50`, LSTM `0.1416`, delta `+0.1103`
- tick `22504`, seconds `65.50`, LSTM `0.4139`, delta `-0.1048`
- tick `21864`, seconds `55.50`, LSTM `0.3945`, delta `+0.1018`
- tick `22600`, seconds `67.00`, LSTM `0.7047`, delta `+0.0926`
- tick `22696`, seconds `68.50`, LSTM `0.2961`, delta `-0.0827`
- tick `22376`, seconds `63.50`, LSTM `0.4913`, delta `+0.0664`

## Top 15 local ridge features

- `lag_02__CT_shots_fired_sum`: coefficient `0.004923`, |coef| `0.004923`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003530`, |coef| `0.003530`
- `lag_00__damage_diff_last_5s`: coefficient `0.002710`, |coef| `0.002710`
- `lag_02__CT3__shots_fired`: coefficient `0.002313`, |coef| `0.002313`
- `lag_02__T5__is_scoped`: coefficient `-0.002230`, |coef| `0.002230`
- `lag_00__kill_diff_last_3s`: coefficient `0.002180`, |coef| `0.002180`
- `lag_03__T_shots_fired_sum`: coefficient `0.002103`, |coef| `0.002103`
- `lag_02__CT3__duck_amount`: coefficient `0.001992`, |coef| `0.001992`
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001987`, |coef| `0.001987`
- `lag_00__T_damage_last_5s`: coefficient `-0.001972`, |coef| `0.001972`
- `lag_00__T_kills_last_3s`: coefficient `-0.001944`, |coef| `0.001944`
- `lag_01__T_place_EXTENDEDA`: coefficient `-0.001912`, |coef| `0.001912`
- `lag_08__CT_place_MIDDLE`: coefficient `0.001878`, |coef| `0.001878`
- `lag_11__CT_place_LOWERTUNNEL`: coefficient `0.001790`, |coef| `0.001790`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001778`, |coef| `0.001778`

## Top 10 utility ridge features

- `lag_14__T3__flash_duration`: coefficient `-0.001589` (lowers CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.001579` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001553` (raises CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001528` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001211` (raises CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.001011` (raises CT win probability)
- `lag_12__T3__flash_duration`: coefficient `0.000995` (raises CT win probability)
- `lag_11__T3__flash_duration`: coefficient `0.000948` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000924` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.000892` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_shots_fired_sum`: coefficient `0.004923` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003530` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002710` (raises CT win probability)
- `lag_02__CT3__shots_fired`: coefficient `0.002313` (raises CT win probability)
- `lag_02__T5__is_scoped`: coefficient `-0.002230` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002180` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `0.002103` (raises CT win probability)
- `lag_02__CT3__duck_amount`: coefficient `0.001992` (raises CT win probability)
- `lag_04__T_place_EXTENDEDA`: coefficient `-0.001987` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001972` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `22664`, seconds `68.00`, LSTM delta `-0.3834`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `-0.071821`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.019585`
- `lag_02__CT3__shots_fired`: contribution `-0.016655`
- `lag_03__T_shots_fired_sum`: contribution `-0.015765`
- `lag_00__CT_place_CATWALK`: contribution `-0.013524`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `-0.008267`

### tick `22568`, seconds `66.50`, LSTM delta `+0.2614`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.026466`
- `lag_02__CT_shots_fired_sum`: contribution `+0.020520`
- `lag_02__T5__is_scoped`: contribution `+0.010637`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.009854`
- `lag_06__T1__flash_duration`: contribution `+0.008133`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.008133`
- `lag_14__T3__flash_duration`: contribution `+0.007119`

### tick `20424`, seconds `33.00`, LSTM delta `-0.2475`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.010647`
- `lag_01__T_place_EXTENDEDA`: contribution `-0.009481`
- `lag_00__T_place_EXTENDEDA`: contribution `-0.008816`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.006895`
- `lag_00__T_kills_last_3s`: contribution `-0.006159`

Top utility-only movements:
- `lag_00__CT5__flash`: contribution `-0.005423`

### tick `22856`, seconds `71.00`, LSTM delta `-0.1402`

Top all feature movements:
- `lag_00__CT_place_SHORTSTAIRS`: contribution `+0.009793`
- `lag_04__T_place_ARAMP`: contribution `-0.008246`
- `lag_08__CT_shots_fired_sum`: contribution `-0.007635`
- `lag_01__CT_burning_players`: contribution `-0.007076`
- `lag_03__T5__is_scoped`: contribution `-0.006801`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.004293`

### tick `21736`, seconds `53.50`, LSTM delta `+0.1103`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.018526`
- `lag_00__damage_diff_last_5s`: contribution `+0.006113`
- `lag_14__T_bomb_zone_count`: contribution `+0.005442`
- `lag_00__kill_diff_last_3s`: contribution `+0.005248`
- `lag_08__CT_place_BDOORS`: contribution `+0.004925`

Top utility-only movements:
- No utility movement among the top local contributors.
