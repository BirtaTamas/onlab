# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `122094`, seconds `63.00`, LSTM `0.5079`, delta `+0.3126`
- tick `122062`, seconds `62.50`, LSTM `0.1953`, delta `-0.3038`
- tick `122350`, seconds `67.00`, LSTM `0.8135`, delta `+0.2871`
- tick `122254`, seconds `65.50`, LSTM `0.7239`, delta `+0.2119`
- tick `122318`, seconds `66.50`, LSTM `0.5264`, delta `-0.1955`
- tick `121070`, seconds `47.00`, LSTM `0.3854`, delta `+0.1272`
- tick `123534`, seconds `85.50`, LSTM `0.7885`, delta `-0.0963`
- tick `118094`, seconds `0.50`, LSTM `0.1184`, delta `-0.0744`
- tick `120526`, seconds `38.50`, LSTM `0.2446`, delta `+0.0641`
- tick `122606`, seconds `71.00`, LSTM `0.8336`, delta `-0.0436`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.006049`, |coef| `0.006049`
- `lag_00__kill_diff_last_3s`: coefficient `0.005190`, |coef| `0.005190`
- `lag_00__damage_diff_last_5s`: coefficient `0.004051`, |coef| `0.004051`
- `lag_00__CT_kills_last_3s`: coefficient `0.003768`, |coef| `0.003768`
- `lag_00__CT_place_RAMP`: coefficient `0.003389`, |coef| `0.003389`
- `lag_08__CT4__duck_amount`: coefficient `0.003336`, |coef| `0.003336`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003165`, |coef| `0.003165`
- `lag_12__CT4__is_walking`: coefficient `-0.002928`, |coef| `0.002928`
- `lag_00__CT_damage_last_5s`: coefficient `0.002719`, |coef| `0.002719`
- `lag_00__T_kills_last_3s`: coefficient `-0.002697`, |coef| `0.002697`
- `lag_01__T_kills_last_3s`: coefficient `0.002654`, |coef| `0.002654`
- `lag_12__CT5__is_walking`: coefficient `0.002645`, |coef| `0.002645`
- `lag_11__CT_place_RAMP`: coefficient `0.002544`, |coef| `0.002544`
- `lag_12__CT4__duck_amount`: coefficient `0.002505`, |coef| `0.002505`
- `lag_11__CT4__duck_amount`: coefficient `-0.002430`, |coef| `0.002430`

## Top 10 utility ridge features

- `lag_15__T1__smoke`: coefficient `-0.001582` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001373` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001325` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001299` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001291` (lowers CT win probability)
- `lag_03__T5__flash`: coefficient `-0.001099` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.001094` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001048` (lowers CT win probability)
- `lag_03__T5__utility_total`: coefficient `-0.000963` (lowers CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.000955` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.006049` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005190` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004051` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003768` (raises CT win probability)
- `lag_00__CT_place_RAMP`: coefficient `0.003389` (raises CT win probability)
- `lag_08__CT4__duck_amount`: coefficient `0.003336` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.003165` (lowers CT win probability)
- `lag_12__CT4__is_walking`: coefficient `-0.002928` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002719` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002697` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `122094`, seconds `63.00`, LSTM delta `+0.3126`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.027213`
- `lag_00__kill_diff_last_3s`: contribution `+0.012493`
- `lag_08__CT4__duck_amount`: contribution `+0.012251`
- `lag_00__CT_kills_last_3s`: contribution `+0.010879`
- `lag_12__CT4__duck_amount`: contribution `+0.009202`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122062`, seconds `62.50`, LSTM delta `-0.3038`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.022678`
- `lag_00__kill_diff_last_3s`: contribution `-0.012493`
- `lag_08__CT4__duck_amount`: contribution `-0.011531`
- `lag_10__T_place_SIDEENTRANCE`: contribution `-0.010832`
- `lag_00__CT_place_RAMP`: contribution `-0.010126`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122350`, seconds `67.00`, LSTM delta `+0.2871`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.018142`
- `lag_00__kill_diff_last_3s`: contribution `+0.012493`
- `lag_00__CT_kills_last_3s`: contribution `+0.010879`
- `lag_00__damage_diff_last_5s`: contribution `+0.009138`
- `lag_01__T_kills_last_3s`: contribution `+0.008408`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122254`, seconds `65.50`, LSTM delta `+0.2119`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012493`
- `lag_00__CT_kills_last_3s`: contribution `+0.010879`
- `lag_11__CT1__duck_amount`: contribution `+0.008421`
- `lag_03__CT2__duck_amount`: contribution `+0.008104`
- `lag_11__CT_place_RAMP`: contribution `+0.007600`

Top utility-only movements:
- `lag_00__T5__flash`: contribution `+0.003896`

### tick `122318`, seconds `66.50`, LSTM delta `-0.1955`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.018142`
- `lag_00__kill_diff_last_3s`: contribution `-0.012493`
- `lag_12__CT4__duck_amount`: contribution `-0.009202`
- `lag_00__damage_diff_last_5s`: contribution `-0.009138`
- `lag_00__T_kills_last_3s`: contribution `-0.008545`

Top utility-only movements:
- No utility movement among the top local contributors.
