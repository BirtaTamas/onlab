# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `36441`, seconds `26.50`, LSTM `0.8361`, delta `+0.2375`
- tick `36825`, seconds `32.50`, LSTM `0.9458`, delta `+0.0892`
- tick `36153`, seconds `22.00`, LSTM `0.6259`, delta `+0.0852`
- tick `36281`, seconds `24.00`, LSTM `0.6207`, delta `-0.0390`
- tick `36345`, seconds `25.00`, LSTM `0.6000`, delta `-0.0336`
- tick `36697`, seconds `30.50`, LSTM `0.8622`, delta `-0.0334`
- tick `34841`, seconds `1.50`, LSTM `0.5356`, delta `+0.0307`
- tick `36249`, seconds `23.50`, LSTM `0.6597`, delta `+0.0248`
- tick `36633`, seconds `29.50`, LSTM `0.8849`, delta `+0.0190`
- tick `35161`, seconds `6.50`, LSTM `0.5569`, delta `+0.0182`

## Top 15 local ridge features

- `lag_13__T_place_WATER`: coefficient `-0.002267`, |coef| `0.002267`
- `lag_00__CT_kills_last_3s`: coefficient `0.001774`, |coef| `0.001774`
- `lag_00__T4__duck_amount`: coefficient `0.001638`, |coef| `0.001638`
- `lag_00__damage_diff_last_5s`: coefficient `0.001551`, |coef| `0.001551`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001502`, |coef| `0.001502`
- `lag_00__kill_diff_last_3s`: coefficient `0.001479`, |coef| `0.001479`
- `lag_09__T5__duck_amount`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_00__T4__alive`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_00__T4__hp`: coefficient `-0.001248`, |coef| `0.001248`
- `lag_13__T5__duck_amount`: coefficient `-0.001228`, |coef| `0.001228`
- `lag_09__CT4__shots_fired`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_09__T5__alive`: coefficient `-0.001221`, |coef| `0.001221`
- `lag_00__T4__utility_total`: coefficient `-0.001211`, |coef| `0.001211`
- `lag_00__CT_place_TSIDELOWER`: coefficient `-0.001196`, |coef| `0.001196`
- `lag_00__T4__armor`: coefficient `-0.001186`, |coef| `0.001186`

## Top 10 utility ridge features

- `lag_00__T4__utility_total`: coefficient `-0.001211` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.001128` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001125` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001119` (raises CT win probability)
- `lag_07__CT5__molly`: coefficient `-0.001095` (lowers CT win probability)
- `lag_03__CT3__molly`: coefficient `-0.001062` (lowers CT win probability)
- `lag_09__T5__smoke`: coefficient `-0.000996` (lowers CT win probability)
- `lag_09__T5__utility_total`: coefficient `-0.000780` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000777` (raises CT win probability)
- `lag_07__CT2__smoke`: coefficient `-0.000745` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_WATER`: coefficient `-0.002267` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001774` (raises CT win probability)
- `lag_00__T4__duck_amount`: coefficient `0.001638` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001551` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001502` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001479` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001272` (lowers CT win probability)
- `lag_00__T4__hp`: coefficient `-0.001248` (lowers CT win probability)
- `lag_13__T5__duck_amount`: coefficient `-0.001228` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `36441`, seconds `26.50`, LSTM delta `+0.2375`

Top all feature movements:
- `lag_13__T_place_WATER`: contribution `+0.012942`
- `lag_09__T5__duck_amount`: contribution `+0.005457`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005218`
- `lag_00__T4__duck_amount`: contribution `+0.005185`
- `lag_00__CT_kills_last_3s`: contribution `+0.005123`

Top utility-only movements:
- `lag_03__CT_B_site_active_infernos`: contribution `+0.003845`

### tick `36825`, seconds `32.50`, LSTM delta `+0.0892`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.005218`
- `lag_00__CT_kills_last_3s`: contribution `+0.005123`
- `lag_00__kill_diff_last_3s`: contribution `+0.003560`
- `lag_00__damage_diff_last_5s`: contribution `+0.003498`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.002670`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `+0.002670`

### tick `36153`, seconds `22.00`, LSTM delta `+0.0852`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.007306`
- `lag_00__CT_kills_last_3s`: contribution `+0.005123`
- `lag_00__kill_diff_last_3s`: contribution `+0.003560`
- `lag_04__T_place_WATER`: contribution `+0.003554`
- `lag_12__T_place_TUNNEL`: contribution `+0.002221`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `+0.001503`
- `lag_10__T_B_site_active_infernos`: contribution `+0.001335`

### tick `36281`, seconds `24.00`, LSTM delta `-0.0390`

Top all feature movements:
- `lag_09__T5__duck_amount`: contribution `-0.005457`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004175`
- `lag_11__T_place_WATER`: contribution `-0.003835`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.002670`
- `lag_00__T2__duck_amount`: contribution `-0.002302`

Top utility-only movements:
- `lag_15__CT_B_site_active_infernos`: contribution `-0.002670`
- `lag_15__CT_active_infernos`: contribution `-0.001160`

### tick `36345`, seconds `25.00`, LSTM delta `-0.0336`

Top all feature movements:
- `lag_13__T_place_WATER`: contribution `-0.012942`
- `lag_00__CT_kills_last_3s`: contribution `-0.005123`
- `lag_10__T5__duck_amount`: contribution `-0.004292`
- `lag_06__CT_shots_fired_sum`: contribution `-0.003829`
- `lag_00__kill_diff_last_3s`: contribution `-0.003560`

Top utility-only movements:
- No utility movement among the top local contributors.
