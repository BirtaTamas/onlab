# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `37829`, seconds `75.50`, LSTM `0.7954`, delta `+0.2347`
- tick `37893`, seconds `76.50`, LSTM `0.7049`, delta `-0.2051`
- tick `37861`, seconds `76.00`, LSTM `0.9101`, delta `+0.1146`
- tick `38053`, seconds `79.00`, LSTM `0.7820`, delta `+0.0685`
- tick `37093`, seconds `64.00`, LSTM `0.4794`, delta `-0.0671`
- tick `38021`, seconds `78.50`, LSTM `0.7135`, delta `-0.0645`
- tick `37637`, seconds `72.50`, LSTM `0.4904`, delta `+0.0514`
- tick `37125`, seconds `64.50`, LSTM `0.4299`, delta `-0.0495`
- tick `37797`, seconds `75.00`, LSTM `0.5607`, delta `+0.0438`
- tick `37989`, seconds `78.00`, LSTM `0.7780`, delta `+0.0424`

## Top 15 local ridge features

- `lag_15__T_place_ARCH`: coefficient `0.002308`, |coef| `0.002308`
- `lag_14__T_place_ARCH`: coefficient `0.001846`, |coef| `0.001846`
- `lag_00__damage_diff_last_5s`: coefficient `0.001638`, |coef| `0.001638`
- `lag_02__T_place_BALCONY`: coefficient `0.001594`, |coef| `0.001594`
- `lag_00__T_place_ARCH`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_05__CT_kills_last_3s`: coefficient `0.001417`, |coef| `0.001417`
- `lag_01__CT4__shots_fired`: coefficient `0.001411`, |coef| `0.001411`
- `lag_00__kill_diff_last_3s`: coefficient `0.001387`, |coef| `0.001387`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001258`, |coef| `0.001258`
- `lag_00__T_kills_last_3s`: coefficient `-0.001224`, |coef| `0.001224`
- `lag_01__CT4__duck_amount`: coefficient `0.001222`, |coef| `0.001222`
- `lag_09__T_place_ARCH`: coefficient `-0.001123`, |coef| `0.001123`
- `lag_10__T_place_ARCH`: coefficient `0.001118`, |coef| `0.001118`
- `lag_11__T_place_ARCH`: coefficient `0.001081`, |coef| `0.001081`
- `lag_00__T_place_BALCONY`: coefficient `-0.001077`, |coef| `0.001077`

## Top 10 utility ridge features

- `lag_09__CT_utility_damage_last_5s`: coefficient `0.001061` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000900` (raises CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `0.000844` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000741` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000727` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000702` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.000659` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000607` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.000556` (raises CT win probability)
- `lag_14__active_infernos_total`: coefficient `-0.000544` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_ARCH`: coefficient `0.002308` (raises CT win probability)
- `lag_14__T_place_ARCH`: coefficient `0.001846` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001638` (raises CT win probability)
- `lag_02__T_place_BALCONY`: coefficient `0.001594` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001436` (lowers CT win probability)
- `lag_05__CT_kills_last_3s`: coefficient `0.001417` (raises CT win probability)
- `lag_01__CT4__shots_fired`: coefficient `0.001411` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001387` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001258` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001224` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `37829`, seconds `75.50`, LSTM delta `+0.2347`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `+0.021475`
- `lag_14__T_place_ARCH`: contribution `+0.017176`
- `lag_00__T_place_BALCONY`: contribution `+0.014805`
- `lag_03__T_place_BALCONY`: contribution `+0.013945`
- `lag_09__T_place_ARCH`: contribution `+0.010447`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.004087`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.002667`
- `lag_14__T_A_site_active_infernos`: contribution `+0.002090`

### tick `37893`, seconds `76.50`, LSTM delta `-0.2051`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `-0.021917`
- `lag_05__T_place_BALCONY`: contribution `-0.013945`
- `lag_10__T_place_ARCH`: contribution `-0.010405`
- `lag_11__T_place_ARCH`: contribution `-0.010060`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006992`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.003468`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.002338`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002298`

### tick `37861`, seconds `76.00`, LSTM delta `+0.1146`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `+0.021475`
- `lag_04__T_place_BALCONY`: contribution `+0.012370`
- `lag_09__T_place_ARCH`: contribution `+0.010447`
- `lag_10__T_place_ARCH`: contribution `-0.010405`
- `lag_00__T_kills_last_3s`: contribution `+0.003877`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38053`, seconds `79.00`, LSTM delta `+0.0685`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `-0.021475`
- `lag_10__T_place_BALCONY`: contribution `+0.011056`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004370`
- `lag_07__CT_shots_fired_sum`: contribution `+0.004145`
- `lag_00__kill_diff_last_3s`: contribution `-0.003338`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `37093`, seconds `64.00`, LSTM delta `-0.0671`

Top all feature movements:
- `lag_08__CT_place_BALCONY`: contribution `-0.004923`
- `lag_00__T_kills_last_3s`: contribution `-0.003877`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.003483`
- `lag_00__kill_diff_last_3s`: contribution `-0.003338`
- `lag_00__damage_diff_last_5s`: contribution `-0.002882`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `-0.002615`
- `lag_05__T_A_site_active_infernos`: contribution `-0.001269`
