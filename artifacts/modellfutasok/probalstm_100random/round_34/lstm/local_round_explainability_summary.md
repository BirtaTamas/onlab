# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `76977`, seconds `45.00`, LSTM `0.2979`, delta `-0.2175`
- tick `77617`, seconds `55.00`, LSTM `0.0614`, delta `-0.2043`
- tick `74865`, seconds `12.00`, LSTM `0.4482`, delta `-0.1543`
- tick `77041`, seconds `46.00`, LSTM `0.2079`, delta `-0.0921`
- tick `77489`, seconds `53.00`, LSTM `0.1864`, delta `+0.0676`
- tick `74961`, seconds `13.50`, LSTM `0.3454`, delta `-0.0524`
- tick `74993`, seconds `14.00`, LSTM `0.3972`, delta `+0.0518`
- tick `77233`, seconds `49.00`, LSTM `0.1607`, delta `-0.0478`
- tick `74833`, seconds `11.50`, LSTM `0.6025`, delta `-0.0477`
- tick `75057`, seconds `15.00`, LSTM `0.4634`, delta `+0.0450`

## Top 15 local ridge features

- `lag_15__CT_place_QUAD`: coefficient `0.003949`, |coef| `0.003949`
- `lag_00__T_kills_last_3s`: coefficient `-0.002599`, |coef| `0.002599`
- `lag_00__kill_diff_last_3s`: coefficient `0.001975`, |coef| `0.001975`
- `lag_04__T_place_BALCONY`: coefficient `0.001850`, |coef| `0.001850`
- `lag_12__T_place_BALCONY`: coefficient `-0.001742`, |coef| `0.001742`
- `lag_00__T_place_BALCONY`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_12__CT_place_BALCONY`: coefficient `-0.001601`, |coef| `0.001601`
- `lag_00__CT5__alive`: coefficient `0.001595`, |coef| `0.001595`
- `lag_00__T_damage_last_5s`: coefficient `-0.001584`, |coef| `0.001584`
- `lag_00__T3__duck_amount`: coefficient `-0.001547`, |coef| `0.001547`
- `lag_06__CT5__is_walking`: coefficient `0.001496`, |coef| `0.001496`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001495`, |coef| `0.001495`
- `lag_15__T_place_TRAMP`: coefficient `0.001482`, |coef| `0.001482`
- `lag_09__T2__is_walking`: coefficient `-0.001480`, |coef| `0.001480`
- `lag_00__CT5__armor`: coefficient `0.001467`, |coef| `0.001467`

## Top 10 utility ridge features

- `lag_00__T1__smoke`: coefficient `0.001350` (raises CT win probability)
- `lag_14__CT2__molly`: coefficient `-0.001311` (lowers CT win probability)
- `lag_00__CT5__flash`: coefficient `0.001164` (raises CT win probability)
- `lag_12__CT1__flash`: coefficient `0.001045` (raises CT win probability)
- `lag_09__CT_he_last_5s`: coefficient `0.001038` (raises CT win probability)
- `lag_11__T5__molly`: coefficient `-0.001005` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000954` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000905` (raises CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `-0.000852` (lowers CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000834` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_QUAD`: coefficient `0.003949` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002599` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001975` (raises CT win probability)
- `lag_04__T_place_BALCONY`: coefficient `0.001850` (raises CT win probability)
- `lag_12__T_place_BALCONY`: coefficient `-0.001742` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001668` (lowers CT win probability)
- `lag_12__CT_place_BALCONY`: coefficient `-0.001601` (lowers CT win probability)
- `lag_00__CT5__alive`: coefficient `0.001595` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001584` (lowers CT win probability)
- `lag_00__T3__duck_amount`: coefficient `-0.001547` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `76977`, seconds `45.00`, LSTM delta `-0.2175`

Top all feature movements:
- `lag_15__CT_place_QUAD`: contribution `-0.031125`
- `lag_00__T_kills_last_3s`: contribution `-0.008235`
- `lag_00__T3__duck_amount`: contribution `-0.005834`
- `lag_00__T_shots_fired_sum`: contribution `-0.005603`
- `lag_05__CT5__duck_amount`: contribution `-0.005120`

Top utility-only movements:
- `lag_14__CT2__molly`: contribution `-0.003231`

### tick `77617`, seconds `55.00`, LSTM delta `-0.2043`

Top all feature movements:
- `lag_04__T_place_BALCONY`: contribution `-0.025446`
- `lag_12__T_place_BALCONY`: contribution `-0.023961`
- `lag_12__CT_place_BALCONY`: contribution `-0.010278`
- `lag_00__T_kills_last_3s`: contribution `-0.008235`
- `lag_00__T3__duck_amount`: contribution `-0.005834`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `-0.002773`

### tick `74865`, seconds `12.00`, LSTM delta `-0.1543`

Top all feature movements:
- `lag_09__CT_he_last_5s`: contribution `-0.019053`
- `lag_00__T_kills_last_3s`: contribution `-0.008235`
- `lag_00__T_shots_fired_sum`: contribution `-0.005603`
- `lag_14__T_place_LOWERMID`: contribution `-0.005466`
- `lag_00__kill_diff_last_3s`: contribution `-0.004753`

Top utility-only movements:
- `lag_09__CT_he_last_5s`: contribution `-0.019053`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004457`
- `lag_02__T5__flash_duration`: contribution `-0.002389`

### tick `77041`, seconds `46.00`, LSTM delta `-0.0921`

Top all feature movements:
- `lag_07__CT1__is_scoped`: contribution `-0.004725`
- `lag_04__CT_place_RUINS`: contribution `-0.003785`
- `lag_01__T_shots_fired_sum`: contribution `+0.003599`
- `lag_06__CT5__is_walking`: contribution `-0.003586`
- `lag_01__CT_shots_fired_sum`: contribution `-0.003440`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77489`, seconds `53.00`, LSTM delta `+0.0676`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.022940`
- `lag_08__T_place_BALCONY`: contribution `+0.009342`
- `lag_13__T2__duck_amount`: contribution `-0.003249`
- `lag_12__T5__is_walking`: contribution `-0.003172`
- `lag_02__T5__duck_amount`: contribution `+0.002670`

Top utility-only movements:
- No utility movement among the top local contributors.
