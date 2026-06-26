# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-m80-bo3-e7FibL-GpwhFRhM0kGS5r4/the-mongolz-vs-m80-m3-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `78198`, seconds `37.00`, LSTM `0.9021`, delta `+0.1113`
- tick `77974`, seconds `33.50`, LSTM `0.8362`, delta `+0.0985`
- tick `76566`, seconds `11.50`, LSTM `0.6747`, delta `+0.0403`
- tick `77206`, seconds `21.50`, LSTM `0.6966`, delta `-0.0333`
- tick `77174`, seconds `21.00`, LSTM `0.7299`, delta `+0.0322`
- tick `76886`, seconds `16.50`, LSTM `0.7151`, delta `+0.0302`
- tick `78134`, seconds `36.00`, LSTM `0.7991`, delta `-0.0245`
- tick `77942`, seconds `33.00`, LSTM `0.7377`, delta `+0.0237`
- tick `78390`, seconds `40.00`, LSTM `0.9168`, delta `-0.0233`
- tick `76630`, seconds `12.50`, LSTM `0.6970`, delta `+0.0202`

## Top 15 local ridge features

- `lag_06__CT_shots_fired_sum`: coefficient `-0.001205`, |coef| `0.001205`
- `lag_00__CT_kills_last_3s`: coefficient `0.001157`, |coef| `0.001157`
- `lag_14__T1__duck_amount`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_00__kill_diff_last_3s`: coefficient `0.000965`, |coef| `0.000965`
- `lag_06__CT2__shots_fired`: coefficient `-0.000913`, |coef| `0.000913`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000869`, |coef| `0.000869`
- `lag_15__bomb_events_last_5s`: coefficient `-0.000851`, |coef| `0.000851`
- `lag_00__T4__flash`: coefficient `-0.000822`, |coef| `0.000822`
- `lag_00__T4__utility_total`: coefficient `-0.000787`, |coef| `0.000787`
- `lag_08__CT_B_site_active_infernos`: coefficient `0.000786`, |coef| `0.000786`
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000764`, |coef| `0.000764`
- `lag_00__T4__alive`: coefficient `-0.000743`, |coef| `0.000743`
- `lag_00__CT5__is_walking`: coefficient `-0.000739`, |coef| `0.000739`
- `lag_12__CT1__is_walking`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_00__CT2__shots_fired`: coefficient `0.000692`, |coef| `0.000692`

## Top 10 utility ridge features

- `lag_00__T4__flash`: coefficient `-0.000822` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000787` (lowers CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `0.000786` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000764` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000658` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000648` (raises CT win probability)
- `lag_02__CT3__molly`: coefficient `-0.000616` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000572` (lowers CT win probability)
- `lag_07__T4__flash`: coefficient `-0.000565` (lowers CT win probability)
- `lag_08__CT_active_infernos`: coefficient `0.000557` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_shots_fired_sum`: coefficient `-0.001205` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001157` (raises CT win probability)
- `lag_14__T1__duck_amount`: coefficient `-0.000972` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000965` (raises CT win probability)
- `lag_06__CT2__shots_fired`: coefficient `-0.000913` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000869` (raises CT win probability)
- `lag_15__bomb_events_last_5s`: coefficient `-0.000851` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.000743` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000739` (lowers CT win probability)
- `lag_12__CT1__is_walking`: coefficient `-0.000695` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `78198`, seconds `37.00`, LSTM delta `+0.1113`

Top all feature movements:
- `lag_06__CT_shots_fired_sum`: contribution `+0.010886`
- `lag_06__CT2__shots_fired`: contribution `+0.005902`
- `lag_00__CT_kills_last_3s`: contribution `+0.003341`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.002700`
- `lag_00__kill_diff_last_3s`: contribution `+0.002322`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.002700`
- `lag_07__T4__flash`: contribution `+0.001535`
- `lag_09__CT3__molly`: contribution `+0.001328`

### tick `77974`, seconds `33.50`, LSTM delta `+0.0985`

Top all feature movements:
- `lag_14__T1__duck_amount`: contribution `+0.003804`
- `lag_15__bomb_events_last_5s`: contribution `+0.003557`
- `lag_00__CT_kills_last_3s`: contribution `+0.003341`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003020`
- `lag_12__T5__is_scoped`: contribution `+0.002760`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.002625`
- `lag_00__T4__flash`: contribution `+0.002233`
- `lag_00__T4__utility_total`: contribution `+0.001836`
- `lag_13__T_B_site_active_infernos`: contribution `+0.001619`

### tick `76566`, seconds `11.50`, LSTM delta `+0.0403`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.008055`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.005496`
- `lag_09__T_place_LOWERMID`: contribution `+0.002934`
- `lag_09__T_place_TRAMP`: contribution `+0.002327`
- `lag_06__CT_place_ARCH`: contribution `+0.002313`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.008055`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.005496`

### tick `77206`, seconds `21.50`, LSTM delta `-0.0333`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.006558`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.004474`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.002700`
- `lag_00__CT5__is_walking`: contribution `-0.001770`
- `lag_12__CT1__is_walking`: contribution `-0.001623`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.006558`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.004474`
- `lag_08__CT_B_site_active_infernos`: contribution `-0.002700`
- `lag_08__CT_active_infernos`: contribution `-0.001285`
- `lag_08__CT3__smoke`: contribution `+0.000810`

### tick `77174`, seconds `21.00`, LSTM delta `+0.0322`

Top all feature movements:
- `lag_12__CT3__duck_amount`: contribution `+0.001950`
- `lag_00__CT5__is_walking`: contribution `+0.001770`
- `lag_12__CT1__is_walking`: contribution `+0.001623`
- `lag_15__T5__duck_amount`: contribution `+0.001557`
- `lag_08__CT3__duck_amount`: contribution `+0.001397`

Top utility-only movements:
- `lag_06__T_B_site_active_infernos`: contribution `+0.000762`
