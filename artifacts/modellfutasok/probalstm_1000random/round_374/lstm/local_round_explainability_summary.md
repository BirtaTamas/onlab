# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `82331`, seconds `33.50`, LSTM `0.3372`, delta `-0.1314`
- tick `85531`, seconds `83.50`, LSTM `0.0333`, delta `-0.1291`
- tick `82299`, seconds `33.00`, LSTM `0.4686`, delta `-0.1049`
- tick `83003`, seconds `44.00`, LSTM `0.0730`, delta `-0.1006`
- tick `85243`, seconds `79.00`, LSTM `0.1538`, delta `+0.1001`
- tick `82363`, seconds `34.00`, LSTM `0.2564`, delta `-0.0808`
- tick `82907`, seconds `42.50`, LSTM `0.2128`, delta `-0.0704`
- tick `83707`, seconds `55.00`, LSTM `0.0363`, delta `-0.0488`
- tick `85339`, seconds `80.50`, LSTM `0.2112`, delta `+0.0480`
- tick `85467`, seconds `82.50`, LSTM `0.1834`, delta `+0.0462`

## Top 15 local ridge features

- `lag_00__T_place_PIT`: coefficient `-0.001856`, |coef| `0.001856`
- `lag_00__T_kills_last_3s`: coefficient `-0.001735`, |coef| `0.001735`
- `lag_00__kill_diff_last_3s`: coefficient `0.001561`, |coef| `0.001561`
- `lag_00__T_place_BALCONY`: coefficient `-0.001516`, |coef| `0.001516`
- `lag_04__CT1__flash_duration`: coefficient `-0.001478`, |coef| `0.001478`
- `lag_05__CT3__flash_duration`: coefficient `0.001272`, |coef| `0.001272`
- `lag_00__T4__is_scoped`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_07__CT_place_BALCONY`: coefficient `0.001222`, |coef| `0.001222`
- `lag_09__CT_shots_fired_sum`: coefficient `0.001200`, |coef| `0.001200`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001174`, |coef| `0.001174`
- `lag_07__CT_place_PIT`: coefficient `-0.001159`, |coef| `0.001159`
- `lag_00__T_damage_last_5s`: coefficient `-0.001143`, |coef| `0.001143`
- `lag_15__T_place_BALCONY`: coefficient `0.001134`, |coef| `0.001134`
- `lag_06__CT_place_BALCONY`: coefficient `0.001117`, |coef| `0.001117`
- `lag_01__CT1__flash_duration`: coefficient `0.001077`, |coef| `0.001077`

## Top 10 utility ridge features

- `lag_04__CT1__flash_duration`: coefficient `-0.001478` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.001272` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.001077` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `-0.001019` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001018` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001009` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000978` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000931` (lowers CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000896` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `0.000861` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_PIT`: coefficient `-0.001856` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001735` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001561` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001516` (lowers CT win probability)
- `lag_00__T4__is_scoped`: coefficient `-0.001233` (lowers CT win probability)
- `lag_07__CT_place_BALCONY`: coefficient `0.001222` (raises CT win probability)
- `lag_09__CT_shots_fired_sum`: coefficient `0.001200` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001174` (lowers CT win probability)
- `lag_07__CT_place_PIT`: coefficient `-0.001159` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001143` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `82331`, seconds `33.50`, LSTM delta `-0.1314`

Top all feature movements:
- `lag_04__CT1__flash_duration`: contribution `-0.009961`
- `lag_07__CT_place_BALCONY`: contribution `-0.007840`
- `lag_01__CT1__flash_duration`: contribution `-0.007259`
- `lag_07__CT_place_PIT`: contribution `-0.004989`
- `lag_14__T4__is_scoped`: contribution `-0.003773`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `-0.009961`
- `lag_01__CT1__flash_duration`: contribution `-0.007259`
- `lag_01__T_A_site_active_infernos`: contribution `-0.002912`
- `lag_01__CT1__utility_total`: contribution `-0.002263`
- `lag_01__CT1__molly`: contribution `-0.001719`

### tick `85531`, seconds `83.50`, LSTM delta `-0.1291`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `-0.009174`
- `lag_05__CT3__flash_duration`: contribution `-0.009050`
- `lag_00__T_shots_fired_sum`: contribution `-0.006159`
- `lag_13__CT_duck_amount_mean`: contribution `-0.006156`
- `lag_00__T_kills_last_3s`: contribution `-0.005497`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `-0.009050`

### tick `82299`, seconds `33.00`, LSTM delta `-0.1049`

Top all feature movements:
- `lag_06__CT_place_BALCONY`: contribution `-0.007171`
- `lag_00__CT1__flash_duration`: contribution `-0.006859`
- `lag_03__CT1__flash_duration`: contribution `-0.006272`
- `lag_00__T_kills_last_3s`: contribution `-0.005497`
- `lag_00__kill_diff_last_3s`: contribution `-0.003756`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.006859`
- `lag_03__CT1__flash_duration`: contribution `-0.006272`
- `lag_00__CT1__utility_total`: contribution `-0.002842`
- `lag_00__CT1__molly`: contribution `-0.002230`
- `lag_00__T_A_site_active_infernos`: contribution `-0.002130`

### tick `83003`, seconds `44.00`, LSTM delta `-0.1006`

Top all feature movements:
- `lag_07__T_place_QUAD`: contribution `-0.020506`
- `lag_10__T_place_QUAD`: contribution `-0.013399`
- `lag_14__T_place_QUAD`: contribution `-0.008194`
- `lag_08__T_place_QUAD`: contribution `+0.008018`
- `lag_04__T_place_QUAD`: contribution `-0.006215`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.001411`
- `lag_00__CT2__molly`: contribution `-0.001315`

### tick `85243`, seconds `79.00`, LSTM delta `+0.1001`

Top all feature movements:
- `lag_15__T_place_BALCONY`: contribution `+0.015589`
- `lag_00__T_place_PIT`: contribution `+0.011709`
- `lag_09__CT3__flash_duration`: contribution `+0.006124`
- `lag_09__T_place_BALCONY`: contribution `+0.005442`
- `lag_13__T_place_PIT`: contribution `+0.004971`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.006124`
- `lag_09__CT_flash_duration_sum`: contribution `+0.001471`
