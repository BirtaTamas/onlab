# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `60324`, seconds `37.00`, LSTM `0.2418`, delta `-0.2247`
- tick `62020`, seconds `63.50`, LSTM `0.0416`, delta `-0.1323`
- tick `61988`, seconds `63.00`, LSTM `0.1739`, delta `+0.0895`
- tick `61348`, seconds `53.00`, LSTM `0.0297`, delta `-0.0875`
- tick `60388`, seconds `38.00`, LSTM `0.2262`, delta `-0.0396`
- tick `60292`, seconds `36.50`, LSTM `0.4665`, delta `-0.0378`
- tick `60516`, seconds `40.00`, LSTM `0.1624`, delta `-0.0362`
- tick `61860`, seconds `61.00`, LSTM `0.0544`, delta `+0.0330`
- tick `62500`, seconds `71.00`, LSTM `0.0057`, delta `-0.0326`
- tick `58596`, seconds `10.00`, LSTM `0.4406`, delta `-0.0287`

## Top 15 local ridge features

- `lag_12__CT_place_EXTENDEDA`: coefficient `0.001688`, |coef| `0.001688`
- `lag_00__T_kills_last_3s`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_00__damage_diff_last_5s`: coefficient `0.001629`, |coef| `0.001629`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001522`, |coef| `0.001522`
- `lag_00__kill_diff_last_3s`: coefficient `0.001481`, |coef| `0.001481`
- `lag_00__T_damage_last_5s`: coefficient `-0.001449`, |coef| `0.001449`
- `lag_01__CT_place_ARAMP`: coefficient `-0.001429`, |coef| `0.001429`
- `lag_04__T1__is_scoped`: coefficient `-0.001362`, |coef| `0.001362`
- `lag_00__CT4__alive`: coefficient `0.001311`, |coef| `0.001311`
- `lag_00__CT4__hp`: coefficient `0.001292`, |coef| `0.001292`
- `lag_12__T_place_CATWALK`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_13__T_place_CATWALK`: coefficient `-0.001216`, |coef| `0.001216`
- `lag_00__CT4__armor`: coefficient `0.001211`, |coef| `0.001211`
- `lag_03__CT4__is_walking`: coefficient `0.001094`, |coef| `0.001094`
- `lag_15__T_place_CATWALK`: coefficient `-0.001078`, |coef| `0.001078`

## Top 10 utility ridge features

- `lag_10__T5__molly`: coefficient `0.000954` (raises CT win probability)
- `lag_13__CT2__smoke`: coefficient `0.000937` (raises CT win probability)
- `lag_10__active_infernos_total`: coefficient `0.000834` (raises CT win probability)
- `lag_10__CT_active_infernos`: coefficient `0.000740` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `0.000642` (raises CT win probability)
- `lag_09__T5__molly`: coefficient `0.000624` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000599` (raises CT win probability)
- `lag_09__T3__molly`: coefficient `-0.000583` (lowers CT win probability)
- `lag_12__CT2__smoke`: coefficient `0.000581` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `0.000540` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_EXTENDEDA`: coefficient `0.001688` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001664` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001629` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.001522` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001481` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001449` (lowers CT win probability)
- `lag_01__CT_place_ARAMP`: coefficient `-0.001429` (lowers CT win probability)
- `lag_04__T1__is_scoped`: coefficient `-0.001362` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001311` (raises CT win probability)
- `lag_00__CT4__hp`: coefficient `0.001292` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `60324`, seconds `37.00`, LSTM delta `-0.2247`

Top all feature movements:
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.009478`
- `lag_01__CT_place_ARAMP`: contribution `-0.008901`
- `lag_04__T1__is_scoped`: contribution `-0.007782`
- `lag_00__T_kills_last_3s`: contribution `-0.005271`
- `lag_13__T_shots_fired_sum`: contribution `-0.004670`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62020`, seconds `63.50`, LSTM delta `-0.1323`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005271`
- `lag_00__damage_diff_last_5s`: contribution `-0.003676`
- `lag_00__kill_diff_last_3s`: contribution `-0.003566`
- `lag_13__T_place_CATWALK`: contribution `-0.003501`
- `lag_00__T_damage_last_5s`: contribution `-0.003475`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61988`, seconds `63.00`, LSTM delta `+0.0895`

Top all feature movements:
- `lag_04__T_place_MIDDOORS`: contribution `+0.003666`
- `lag_12__T_place_CATWALK`: contribution `-0.003561`
- `lag_12__CT_place_BDOORS`: contribution `+0.003021`
- `lag_00__damage_diff_last_5s`: contribution `+0.003014`
- `lag_13__T4__duck_amount`: contribution `+0.002820`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `+0.001522`

### tick `61348`, seconds `53.00`, LSTM delta `-0.0875`

Top all feature movements:
- `lag_06__CT_place_ARAMP`: contribution `-0.006679`
- `lag_00__T_kills_last_3s`: contribution `-0.005271`
- `lag_00__damage_diff_last_5s`: contribution `-0.003676`
- `lag_00__T1__duck_amount`: contribution `-0.003588`
- `lag_00__kill_diff_last_3s`: contribution `-0.003566`

Top utility-only movements:
- `lag_09__T3__molly`: contribution `+0.001295`
- `lag_05__T_A_site_active_infernos`: contribution `-0.001017`

### tick `60388`, seconds `38.00`, LSTM delta `-0.0396`

Top all feature movements:
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.003771`
- `lag_06__T1__is_scoped`: contribution `-0.003310`
- `lag_14__T_place_CATWALK`: contribution `-0.002982`
- `lag_01__T_place_TUNNELSTAIRS`: contribution `-0.002760`
- `lag_03__CT_place_ARAMP`: contribution `-0.002489`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `-0.001411`
- `lag_12__active_infernos_total`: contribution `-0.001198`
