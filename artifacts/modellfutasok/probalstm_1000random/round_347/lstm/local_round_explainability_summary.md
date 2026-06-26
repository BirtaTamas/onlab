# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `74802`, seconds `32.50`, LSTM `0.6843`, delta `+0.1054`
- tick `74962`, seconds `35.00`, LSTM `0.8235`, delta `+0.0978`
- tick `77458`, seconds `74.00`, LSTM `0.9256`, delta `+0.0674`
- tick `77778`, seconds `79.00`, LSTM `0.9727`, delta `+0.0373`
- tick `76690`, seconds `62.00`, LSTM `0.8679`, delta `-0.0278`
- tick `75538`, seconds `44.00`, LSTM `0.7772`, delta `-0.0251`
- tick `74930`, seconds `34.50`, LSTM `0.7257`, delta `+0.0243`
- tick `76658`, seconds `61.50`, LSTM `0.8956`, delta `+0.0240`
- tick `74386`, seconds `26.00`, LSTM `0.6014`, delta `+0.0239`
- tick `73714`, seconds `15.50`, LSTM `0.5637`, delta `-0.0226`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001692`, |coef| `0.001692`
- `lag_00__kill_diff_last_3s`: coefficient `0.001569`, |coef| `0.001569`
- `lag_11__T_place_BALCONY`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_13__T_place_BALCONY`: coefficient `0.000971`, |coef| `0.000971`
- `lag_00__damage_diff_last_5s`: coefficient `0.000934`, |coef| `0.000934`
- `lag_02__T5__is_walking`: coefficient `0.000924`, |coef| `0.000924`
- `lag_00__T3__alive`: coefficient `-0.000891`, |coef| `0.000891`
- `lag_07__CT_place_ARCH`: coefficient `0.000862`, |coef| `0.000862`
- `lag_00__CT_damage_last_5s`: coefficient `0.000848`, |coef| `0.000848`
- `lag_04__CT1__is_walking`: coefficient `0.000805`, |coef| `0.000805`
- `lag_00__T3__smoke`: coefficient `-0.000800`, |coef| `0.000800`
- `lag_12__T1__duck_amount`: coefficient `0.000799`, |coef| `0.000799`
- `lag_00__CT4__smoke`: coefficient `-0.000798`, |coef| `0.000798`
- `lag_15__T_place_SECONDMID`: coefficient `-0.000790`, |coef| `0.000790`
- `lag_00__T3__has_helmet`: coefficient `-0.000771`, |coef| `0.000771`

## Top 10 utility ridge features

- `lag_00__T3__smoke`: coefficient `-0.000800` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `-0.000798` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000720` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000641` (lowers CT win probability)
- `lag_02__CT4__smoke`: coefficient `-0.000614` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000607` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000580` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.000573` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000563` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `0.000562` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001692` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001569` (raises CT win probability)
- `lag_11__T_place_BALCONY`: coefficient `-0.001297` (lowers CT win probability)
- `lag_13__T_place_BALCONY`: coefficient `0.000971` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000934` (raises CT win probability)
- `lag_02__T5__is_walking`: coefficient `0.000924` (raises CT win probability)
- `lag_00__T3__alive`: coefficient `-0.000891` (lowers CT win probability)
- `lag_07__CT_place_ARCH`: coefficient `0.000862` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000848` (raises CT win probability)
- `lag_04__CT1__is_walking`: coefficient `0.000805` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `74802`, seconds `32.50`, LSTM delta `+0.1054`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `+0.017831`
- `lag_13__T_place_BALCONY`: contribution `+0.013354`
- `lag_00__CT_kills_last_3s`: contribution `+0.004886`
- `lag_00__kill_diff_last_3s`: contribution `+0.003776`
- `lag_15__T_place_SECONDMID`: contribution `+0.002587`

Top utility-only movements:
- `lag_00__T4__utility_total`: contribution `+0.001415`

### tick `74962`, seconds `35.00`, LSTM delta `+0.0978`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004886`
- `lag_00__kill_diff_last_3s`: contribution `+0.003776`
- `lag_07__CT_place_ARCH`: contribution `+0.003516`
- `lag_05__T5__duck_amount`: contribution `+0.002815`
- `lag_06__T4__is_scoped`: contribution `+0.002381`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.001893`
- `lag_01__T_B_site_active_infernos`: contribution `+0.001592`
- `lag_15__T_B_site_active_infernos`: contribution `+0.001589`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.001499`
- `lag_00__T2__molly`: contribution `+0.001429`

### tick `77458`, seconds `74.00`, LSTM delta `+0.0674`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004886`
- `lag_00__kill_diff_last_3s`: contribution `+0.003776`
- `lag_12__T1__duck_amount`: contribution `+0.003128`
- `lag_00__T3__alive`: contribution `+0.002155`
- `lag_02__T5__is_walking`: contribution `+0.002143`

Top utility-only movements:
- `lag_00__T3__smoke`: contribution `+0.001738`
- `lag_00__T3__utility_total`: contribution `+0.001173`

### tick `77778`, seconds `79.00`, LSTM delta `+0.0373`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004886`
- `lag_00__kill_diff_last_3s`: contribution `+0.003776`
- `lag_02__CT_place_ARCH`: contribution `+0.001885`
- `lag_04__CT1__is_walking`: contribution `-0.001880`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001700`

Top utility-only movements:
- `lag_14__CT4__smoke`: contribution `+0.000743`

### tick `76690`, seconds `62.00`, LSTM delta `-0.0278`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.003776`
- `lag_02__T5__is_walking`: contribution `-0.002143`
- `lag_13__T3__duck_amount`: contribution `-0.001960`
- `lag_00__CT5__duck_amount`: contribution `-0.001899`
- `lag_04__CT1__is_walking`: contribution `-0.001880`

Top utility-only movements:
- No utility movement among the top local contributors.
