# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `83505`, seconds `60.50`, LSTM `0.2268`, delta `-0.2782`
- tick `82577`, seconds `46.00`, LSTM `0.2403`, delta `-0.2240`
- tick `83441`, seconds `59.50`, LSTM `0.5157`, delta `+0.2203`
- tick `82545`, seconds `45.50`, LSTM `0.4643`, delta `+0.1941`
- tick `83569`, seconds `61.50`, LSTM `0.1466`, delta `-0.0852`
- tick `82449`, seconds `44.00`, LSTM `0.2316`, delta `+0.0742`
- tick `79665`, seconds `0.50`, LSTM `0.1323`, delta `-0.0736`
- tick `86065`, seconds `100.50`, LSTM `0.0479`, delta `-0.0728`
- tick `84145`, seconds `70.50`, LSTM `0.2082`, delta `-0.0701`
- tick `83793`, seconds `65.00`, LSTM `0.1844`, delta `+0.0651`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.005066`, |coef| `0.005066`
- `lag_04__CT_place_EXTENDEDA`: coefficient `0.002562`, |coef| `0.002562`
- `lag_00__kill_diff_last_3s`: coefficient `0.002314`, |coef| `0.002314`
- `lag_00__CT_place_OUTSIDETUNNEL`: coefficient `-0.001987`, |coef| `0.001987`
- `lag_04__T_shots_fired_sum`: coefficient `0.001954`, |coef| `0.001954`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001799`, |coef| `0.001799`
- `lag_00__T_kills_last_3s`: coefficient `-0.001772`, |coef| `0.001772`
- `lag_01__T_place_ARAMP`: coefficient `-0.001672`, |coef| `0.001672`
- `lag_00__damage_diff_last_5s`: coefficient `0.001641`, |coef| `0.001641`
- `lag_04__T3__shots_fired`: coefficient `0.001582`, |coef| `0.001582`
- `lag_00__T_place_ARAMP`: coefficient `-0.001555`, |coef| `0.001555`
- `lag_00__CT_velocity_mean`: coefficient `-0.001463`, |coef| `0.001463`
- `lag_15__CT_place_SHORTSTAIRS`: coefficient `0.001439`, |coef| `0.001439`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_01__T_place_LONGDOORS`: coefficient `0.001428`, |coef| `0.001428`

## Top 10 utility ridge features

- `lag_00__T_A_site_active_infernos`: coefficient `-0.000781` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.000725` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000619` (lowers CT win probability)
- `lag_02__T1__utility_total`: coefficient `0.000592` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000583` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000568` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `0.000563` (raises CT win probability)
- `lag_02__T1__molly`: coefficient `0.000548` (raises CT win probability)
- `lag_03__T2__molly`: coefficient `0.000546` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `-0.000544` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.005066` (raises CT win probability)
- `lag_04__CT_place_EXTENDEDA`: coefficient `0.002562` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002314` (raises CT win probability)
- `lag_00__CT_place_OUTSIDETUNNEL`: coefficient `-0.001987` (lowers CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `0.001954` (raises CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001799` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001772` (lowers CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `-0.001672` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001641` (raises CT win probability)
- `lag_04__T3__shots_fired`: coefficient `0.001582` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `83505`, seconds `60.50`, LSTM delta `-0.2782`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.045758`
- `lag_04__CT_place_EXTENDEDA`: contribution `-0.014380`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `-0.008020`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.007151`
- `lag_08__T5__is_scoped`: contribution `-0.006383`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82577`, seconds `46.00`, LSTM delta `-0.2240`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.045758`
- `lag_04__T_shots_fired_sum`: contribution `-0.032234`
- `lag_04__T3__shots_fired`: contribution `-0.021075`
- `lag_02__CT_place_HOLE`: contribution `-0.008937`
- `lag_00__T5__is_scoped`: contribution `-0.006561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83441`, seconds `59.50`, LSTM delta `+0.2203`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.021119`
- `lag_04__CT_place_EXTENDEDA`: contribution `+0.014380`
- `lag_13__CT_place_EXTENDEDA`: contribution `+0.007969`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.007922`
- `lag_02__CT_place_EXTENDEDA`: contribution `+0.006473`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `82545`, seconds `45.50`, LSTM delta `+0.1941`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.024639`
- `lag_03__T_shots_fired_sum`: contribution `+0.019776`
- `lag_03__T3__shots_fired`: contribution `+0.012858`
- `lag_04__T_shots_fired_sum`: contribution `+0.007326`
- `lag_00__T5__is_scoped`: contribution `+0.006561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83569`, seconds `61.50`, LSTM delta `-0.0852`

Top all feature movements:
- `lag_00__CT_place_OUTSIDETUNNEL`: contribution `-0.042741`
- `lag_02__CT_shots_fired_sum`: contribution `+0.006888`
- `lag_01__T_shots_fired_sum`: contribution `+0.005402`
- `lag_03__T_shots_fired_sum`: contribution `-0.004495`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `-0.004157`

Top utility-only movements:
- No utility movement among the top local contributors.
