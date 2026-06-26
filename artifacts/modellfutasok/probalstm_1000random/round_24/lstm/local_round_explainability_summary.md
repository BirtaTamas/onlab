# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `21`

## Largest probability jumps

- tick `169909`, seconds `19.00`, LSTM `0.9159`, delta `+0.1311`
- tick `169813`, seconds `17.50`, LSTM `0.7891`, delta `+0.1148`
- tick `169557`, seconds `13.50`, LSTM `0.6934`, delta `+0.0737`
- tick `169973`, seconds `20.00`, LSTM `0.8834`, delta `-0.0440`
- tick `170485`, seconds `28.00`, LSTM `0.9284`, delta `+0.0415`
- tick `171765`, seconds `48.00`, LSTM `0.8916`, delta `-0.0353`
- tick `170133`, seconds `22.50`, LSTM `0.8113`, delta `-0.0351`
- tick `170357`, seconds `26.00`, LSTM `0.8641`, delta `-0.0327`
- tick `169749`, seconds `16.50`, LSTM `0.6473`, delta `-0.0323`
- tick `170165`, seconds `23.00`, LSTM `0.8419`, delta `+0.0306`

## Top 15 local ridge features

- `lag_10__T_place_CONTROL`: coefficient `0.001003`, |coef| `0.001003`
- `lag_13__T_place_CONTROL`: coefficient `0.000994`, |coef| `0.000994`
- `lag_00__kill_diff_last_3s`: coefficient `0.000786`, |coef| `0.000786`
- `lag_00__CT_burning_players`: coefficient `0.000734`, |coef| `0.000734`
- `lag_14__T_place_CONTROL`: coefficient `0.000629`, |coef| `0.000629`
- `lag_00__T_place_GARAGE`: coefficient `0.000600`, |coef| `0.000600`
- `lag_00__CT_kills_last_3s`: coefficient `0.000598`, |coef| `0.000598`
- `lag_08__T_place_CONTROL`: coefficient `-0.000596`, |coef| `0.000596`
- `lag_03__CT_burning_players`: coefficient `0.000594`, |coef| `0.000594`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000589`, |coef| `0.000589`
- `lag_02__CT_burning_players`: coefficient `0.000562`, |coef| `0.000562`
- `lag_10__T_place_TROPHY`: coefficient `-0.000550`, |coef| `0.000550`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000549`, |coef| `0.000549`
- `lag_02__CT_place_VENTS`: coefficient `0.000547`, |coef| `0.000547`
- `lag_09__CT3__duck_amount`: coefficient `0.000541`, |coef| `0.000541`

## Top 10 utility ridge features

- `lag_13__CT_utility_damage_last_5s`: coefficient `0.000519` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.000395` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000366` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000356` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.000346` (raises CT win probability)
- `lag_01__CT3__molly`: coefficient `0.000326` (raises CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000323` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000316` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000308` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.000303` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_CONTROL`: coefficient `0.001003` (raises CT win probability)
- `lag_13__T_place_CONTROL`: coefficient `0.000994` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000786` (raises CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.000734` (raises CT win probability)
- `lag_14__T_place_CONTROL`: coefficient `0.000629` (raises CT win probability)
- `lag_00__T_place_GARAGE`: coefficient `0.000600` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000598` (raises CT win probability)
- `lag_08__T_place_CONTROL`: coefficient `-0.000596` (lowers CT win probability)
- `lag_03__CT_burning_players`: coefficient `0.000594` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.000589` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `169909`, seconds `19.00`, LSTM delta `+0.1311`

Top all feature movements:
- `lag_13__T_place_CONTROL`: contribution `+0.014131`
- `lag_13__T_place_TROPHY`: contribution `+0.005033`
- `lag_14__T_place_CONTROL`: contribution `+0.004468`
- `lag_08__T_place_CONTROL`: contribution `+0.004236`
- `lag_05__CT_place_VENTS`: contribution `+0.004074`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.002134`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.001532`

### tick `169813`, seconds `17.50`, LSTM delta `+0.1148`

Top all feature movements:
- `lag_10__T_place_CONTROL`: contribution `+0.014254`
- `lag_10__T_place_TROPHY`: contribution `+0.006978`
- `lag_02__CT_place_VENTS`: contribution `+0.004593`
- `lag_08__T_place_CONTROL`: contribution `+0.004236`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.003656`

Top utility-only movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.003656`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.002280`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002171`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.001543`

### tick `169557`, seconds `13.50`, LSTM delta `+0.0737`

Top all feature movements:
- `lag_02__T_place_CONTROL`: contribution `+0.007493`
- `lag_02__T_place_TROPHY`: contribution `+0.004621`
- `lag_10__CT_place_RAFTERS`: contribution `+0.003128`
- `lag_15__CT_place_ADMIN`: contribution `+0.002642`
- `lag_13__CT_place_HEAVEN`: contribution `+0.002577`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.001985`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.001283`
- `lag_04__T_A_site_active_infernos`: contribution `+0.001091`

### tick `169973`, seconds `20.00`, LSTM delta `-0.0440`

Top all feature movements:
- `lag_10__T_place_CONTROL`: contribution `-0.007127`
- `lag_13__T_place_CONTROL`: contribution `-0.007065`
- `lag_15__T_place_TROPHY`: contribution `-0.006461`
- `lag_14__T_place_CONTROL`: contribution `+0.004468`
- `lag_04__CT_place_ADMIN`: contribution `-0.002430`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.001369`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.000987`

### tick `170485`, seconds `28.00`, LSTM delta `+0.0415`

Top all feature movements:
- `lag_09__CT_place_DECON`: contribution `+0.006656`
- `lag_03__CT_place_DECON`: contribution `+0.005955`
- `lag_13__T_place_HUT`: contribution `+0.002447`
- `lag_13__CT_place_VENTS`: contribution `+0.001914`
- `lag_06__T_place_HUT`: contribution `+0.001752`

Top utility-only movements:
- `lag_00__CT2__utility_total`: contribution `+0.000736`
