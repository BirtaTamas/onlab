# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `59523`, seconds `68.50`, LSTM `0.6085`, delta `+0.4312`
- tick `59459`, seconds `67.50`, LSTM `0.2205`, delta `-0.2443`
- tick `55971`, seconds `13.00`, LSTM `0.0695`, delta `-0.1635`
- tick `58403`, seconds `51.00`, LSTM `0.2444`, delta `+0.1485`
- tick `58851`, seconds `58.00`, LSTM `0.4989`, delta `+0.1170`
- tick `58435`, seconds `51.50`, LSTM `0.3369`, delta `+0.0925`
- tick `59907`, seconds `74.50`, LSTM `0.7025`, delta `+0.0694`
- tick `59555`, seconds `69.00`, LSTM `0.5445`, delta `-0.0639`
- tick `58979`, seconds `60.00`, LSTM `0.4741`, delta `-0.0620`
- tick `59587`, seconds `69.50`, LSTM `0.6049`, delta `+0.0604`

## Top 15 local ridge features

- `lag_07__CT_place_BACKALLEY`: coefficient `0.006098`, |coef| `0.006098`
- `lag_00__T_place_TRUCK`: coefficient `-0.004851`, |coef| `0.004851`
- `lag_15__CT_place_SIDEALLEY`: coefficient `0.003629`, |coef| `0.003629`
- `lag_05__CT_place_BACKALLEY`: coefficient `-0.003431`, |coef| `0.003431`
- `lag_12__T_place_TRUCK`: coefficient `0.002991`, |coef| `0.002991`
- `lag_00__CT_place_SIDEALLEY`: coefficient `0.002862`, |coef| `0.002862`
- `lag_02__CT_place_BACKALLEY`: coefficient `-0.002434`, |coef| `0.002434`
- `lag_00__damage_diff_last_5s`: coefficient `0.002161`, |coef| `0.002161`
- `lag_00__kill_diff_last_3s`: coefficient `0.001938`, |coef| `0.001938`
- `lag_10__T_place_TRUCK`: coefficient `-0.001900`, |coef| `0.001900`
- `lag_05__T5__is_scoped`: coefficient `-0.001869`, |coef| `0.001869`
- `lag_09__T5__is_scoped`: coefficient `-0.001692`, |coef| `0.001692`
- `lag_12__CT2__duck_amount`: coefficient `0.001670`, |coef| `0.001670`
- `lag_14__CT1__is_walking`: coefficient `-0.001663`, |coef| `0.001663`
- `lag_15__CT_place_SHOP`: coefficient `-0.001650`, |coef| `0.001650`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001567` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001216` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.001205` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.001204` (lowers CT win probability)
- `lag_09__T_he_last_5s`: coefficient `0.001048` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.001024` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000991` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000904` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.000899` (lowers CT win probability)
- `lag_15__CT_flashes_last_5s`: coefficient `0.000883` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_BACKALLEY`: coefficient `0.006098` (raises CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.004851` (lowers CT win probability)
- `lag_15__CT_place_SIDEALLEY`: coefficient `0.003629` (raises CT win probability)
- `lag_05__CT_place_BACKALLEY`: coefficient `-0.003431` (lowers CT win probability)
- `lag_12__T_place_TRUCK`: coefficient `0.002991` (raises CT win probability)
- `lag_00__CT_place_SIDEALLEY`: coefficient `0.002862` (raises CT win probability)
- `lag_02__CT_place_BACKALLEY`: coefficient `-0.002434` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002161` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001938` (raises CT win probability)
- `lag_10__T_place_TRUCK`: coefficient `-0.001900` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `59523`, seconds `68.50`, LSTM delta `+0.4312`

Top all feature movements:
- `lag_07__CT_place_BACKALLEY`: contribution `+0.091423`
- `lag_00__T_place_TRUCK`: contribution `+0.084251`
- `lag_12__T_place_TRUCK`: contribution `+0.051936`
- `lag_05__T5__is_scoped`: contribution `+0.008916`
- `lag_15__CT_place_SHOP`: contribution `+0.008275`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59459`, seconds `67.50`, LSTM delta `-0.2443`

Top all feature movements:
- `lag_15__CT_place_SIDEALLEY`: contribution `-0.066217`
- `lag_05__CT_place_BACKALLEY`: contribution `-0.051441`
- `lag_10__T_place_TRUCK`: contribution `-0.033003`
- `lag_09__T5__is_scoped`: contribution `-0.008072`
- `lag_13__CT_place_SHOP`: contribution `-0.006462`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55971`, seconds `13.00`, LSTM delta `-0.1635`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.026184`
- `lag_09__T_he_last_5s`: contribution `-0.013679`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.010471`
- `lag_15__CT_flashes_last_5s`: contribution `-0.009711`
- `lag_15__CT_place_SHOP`: contribution `-0.008275`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.026184`
- `lag_09__T_he_last_5s`: contribution `-0.013679`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.010471`
- `lag_15__CT_flashes_last_5s`: contribution `-0.009711`

### tick `58403`, seconds `51.00`, LSTM delta `+0.1485`

Top all feature movements:
- `lag_10__T_shots_fired_sum`: contribution `+0.008950`
- `lag_04__CT2__flash_duration`: contribution `+0.005893`
- `lag_15__CT1__duck_amount`: contribution `+0.004876`
- `lag_00__CT2__duck_amount`: contribution `-0.004428`
- `lag_11__CT5__shots_fired`: contribution `+0.004195`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.005893`
- `lag_03__T_B_site_active_infernos`: contribution `+0.003408`
- `lag_14__CT2__flash_duration`: contribution `+0.002196`

### tick `58851`, seconds `58.00`, LSTM delta `+0.1170`

Top all feature movements:
- `lag_00__CT_place_SIDEALLEY`: contribution `+0.052221`
- `lag_14__CT_place_TSPAWN`: contribution `+0.011234`
- `lag_05__T5__is_scoped`: contribution `+0.008916`
- `lag_04__T5__is_scoped`: contribution `+0.004811`
- `lag_10__T5__is_scoped`: contribution `+0.003893`

Top utility-only movements:
- No utility movement among the top local contributors.
