# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `15`

## Largest probability jumps

- tick `104011`, seconds `65.50`, LSTM `0.6852`, delta `+0.3081`
- tick `102091`, seconds `35.50`, LSTM `0.2954`, delta `-0.2420`
- tick `103531`, seconds `58.00`, LSTM `0.3889`, delta `+0.1872`
- tick `103755`, seconds `61.50`, LSTM `0.5418`, delta `+0.1585`
- tick `103499`, seconds `57.50`, LSTM `0.2017`, delta `+0.1072`
- tick `104139`, seconds `67.50`, LSTM `0.8540`, delta `+0.0791`
- tick `104203`, seconds `68.50`, LSTM `0.9511`, delta `+0.0716`
- tick `102123`, seconds `36.00`, LSTM `0.2289`, delta `-0.0665`
- tick `103691`, seconds `60.50`, LSTM `0.3999`, delta `-0.0652`
- tick `103947`, seconds `64.50`, LSTM `0.3449`, delta `-0.0597`

## Top 15 local ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002544`, |coef| `0.002544`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002285`, |coef| `0.002285`
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.002268`, |coef| `0.002268`
- `lag_07__CT_place_ARAMP`: coefficient `0.002213`, |coef| `0.002213`
- `lag_00__kill_diff_last_3s`: coefficient `0.002154`, |coef| `0.002154`
- `lag_00__damage_diff_last_5s`: coefficient `0.002020`, |coef| `0.002020`
- `lag_10__T_place_BDOORS`: coefficient `0.001964`, |coef| `0.001964`
- `lag_08__T_place_HOLE`: coefficient `0.001941`, |coef| `0.001941`
- `lag_04__T_place_HOLE`: coefficient `0.001843`, |coef| `0.001843`
- `lag_13__CT_shots_fired_sum`: coefficient `-0.001769`, |coef| `0.001769`
- `lag_10__CT_place_ARAMP`: coefficient `0.001694`, |coef| `0.001694`
- `lag_02__CT_place_EXTENDEDA`: coefficient `0.001620`, |coef| `0.001620`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001599`, |coef| `0.001599`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001595`, |coef| `0.001595`
- `lag_12__CT_place_ARAMP`: coefficient `-0.001553`, |coef| `0.001553`

## Top 10 utility ridge features

- `lag_00__CT4__utility_total`: coefficient `0.001388` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001225` (raises CT win probability)
- `lag_08__CT1__molly`: coefficient `0.001107` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.001085` (raises CT win probability)
- `lag_02__CT2__molly`: coefficient `0.001052` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000898` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000862` (raises CT win probability)
- `lag_01__CT4__utility_total`: coefficient `0.000753` (raises CT win probability)
- `lag_01__CT_active_infernos`: coefficient `-0.000721` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000692` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HOLE`: coefficient `-0.002544` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002285` (raises CT win probability)
- `lag_05__CT_place_EXTENDEDA`: coefficient `-0.002268` (lowers CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `0.002213` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002154` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002020` (raises CT win probability)
- `lag_10__T_place_BDOORS`: coefficient `0.001964` (raises CT win probability)
- `lag_08__T_place_HOLE`: coefficient `0.001941` (raises CT win probability)
- `lag_04__T_place_HOLE`: coefficient `0.001843` (raises CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `-0.001769` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `104011`, seconds `65.50`, LSTM delta `+0.3081`

Top all feature movements:
- `lag_00__T_place_HOLE`: contribution `+0.065591`
- `lag_04__T_place_HOLE`: contribution `+0.047521`
- `lag_10__T_place_BDOORS`: contribution `+0.024563`
- `lag_13__CT_shots_fired_sum`: contribution `+0.023346`
- `lag_01__T_place_BDOORS`: contribution `+0.018605`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `102091`, seconds `35.50`, LSTM delta `-0.2420`

Top all feature movements:
- `lag_07__CT_place_ARAMP`: contribution `-0.013786`
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.012730`
- `lag_10__CT_place_ARAMP`: contribution `-0.010554`
- `lag_12__CT_place_ARAMP`: contribution `-0.009677`
- `lag_02__CT_place_EXTENDEDA`: contribution `-0.009097`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.003872`
- `lag_00__CT4__molly`: contribution `-0.003017`

### tick `103531`, seconds `58.00`, LSTM delta `+0.1872`

Top all feature movements:
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.012730`
- `lag_09__T_place_MIDDOORS`: contribution `+0.012430`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007939`
- `lag_00__T_place_MIDDOORS`: contribution `+0.006797`
- `lag_01__T_shots_fired_sum`: contribution `+0.005341`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.002854`

### tick `103755`, seconds `61.50`, LSTM delta `+0.1585`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `+0.013486`
- `lag_02__T_place_BDOORS`: contribution `+0.012907`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009527`
- `lag_05__CT1__shots_fired`: contribution `+0.008279`
- `lag_00__kill_diff_last_3s`: contribution `+0.005186`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103499`, seconds `57.50`, LSTM delta `+0.1072`

Top all feature movements:
- `lag_05__CT_place_EXTENDEDA`: contribution `-0.012730`
- `lag_08__T_place_MIDDOORS`: contribution `+0.009585`
- `lag_00__T_shots_fired_sum`: contribution `+0.009565`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007939`
- `lag_04__CT_place_EXTENDEDA`: contribution `+0.004642`

Top utility-only movements:
- No utility movement among the top local contributors.
