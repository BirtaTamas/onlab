# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `39531`, seconds `44.50`, LSTM `0.3036`, delta `-0.1996`
- tick `39275`, seconds `40.50`, LSTM `0.4561`, delta `+0.1874`
- tick `39595`, seconds `45.50`, LSTM `0.0863`, delta `-0.1855`
- tick `39083`, seconds `37.50`, LSTM `0.3463`, delta `-0.1587`
- tick `39627`, seconds `46.00`, LSTM `0.0188`, delta `-0.0675`
- tick `39147`, seconds `38.50`, LSTM `0.3309`, delta `-0.0585`
- tick `39115`, seconds `38.00`, LSTM `0.3895`, delta `+0.0431`
- tick `38539`, seconds `29.00`, LSTM `0.4654`, delta `-0.0345`
- tick `37739`, seconds `16.50`, LSTM `0.4826`, delta `-0.0324`
- tick `39563`, seconds `45.00`, LSTM `0.2718`, delta `-0.0317`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003654`, |coef| `0.003654`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.003240`, |coef| `0.003240`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002535`, |coef| `0.002535`
- `lag_12__T_place_TSIDEUPPER`: coefficient `-0.002111`, |coef| `0.002111`
- `lag_15__T_shots_fired_sum`: coefficient `0.001830`, |coef| `0.001830`
- `lag_13__T_place_TSIDELOWER`: coefficient `0.001766`, |coef| `0.001766`
- `lag_00__T_kills_last_3s`: coefficient `-0.001703`, |coef| `0.001703`
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_12__CT1__is_scoped`: coefficient `0.001509`, |coef| `0.001509`
- `lag_10__T_place_TSIDEUPPER`: coefficient `-0.001454`, |coef| `0.001454`
- `lag_14__CT3__duck_amount`: coefficient `-0.001438`, |coef| `0.001438`
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001369`, |coef| `0.001369`
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.001351`, |coef| `0.001351`
- `lag_11__T_place_TSIDEUPPER`: coefficient `-0.001330`, |coef| `0.001330`
- `lag_15__T5__shots_fired`: coefficient `0.001318`, |coef| `0.001318`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.003240` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.002535` (raises CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.001674` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.001351` (lowers CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.001168` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000855` (raises CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.000825` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000708` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `0.000533` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000524` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003654` (raises CT win probability)
- `lag_12__T_place_TSIDEUPPER`: coefficient `-0.002111` (lowers CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `0.001830` (raises CT win probability)
- `lag_13__T_place_TSIDELOWER`: coefficient `0.001766` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001703` (lowers CT win probability)
- `lag_12__CT1__is_scoped`: coefficient `0.001509` (raises CT win probability)
- `lag_10__T_place_TSIDEUPPER`: coefficient `-0.001454` (lowers CT win probability)
- `lag_14__CT3__duck_amount`: coefficient `-0.001438` (lowers CT win probability)
- `lag_07__T_place_TSIDELOWER`: coefficient `-0.001369` (lowers CT win probability)
- `lag_11__T_place_TSIDEUPPER`: coefficient `-0.001330` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `39531`, seconds `44.50`, LSTM delta `-0.1996`

Top all feature movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.021008`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.013908`
- `lag_15__T_shots_fired_sum`: contribution `-0.013720`
- `lag_15__T5__shots_fired`: contribution `-0.008101`
- `lag_10__T_place_TSIDEUPPER`: contribution `-0.007332`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.021008`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.013908`

### tick `39275`, seconds `40.50`, LSTM delta `+0.1874`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.040657`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.026092`
- `lag_13__T_place_TSIDELOWER`: contribution `+0.019852`
- `lag_13__T_place_RAMP`: contribution `+0.012216`
- `lag_12__CT1__is_scoped`: contribution `+0.006462`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.040657`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.026092`

### tick `39595`, seconds `45.50`, LSTM delta `-0.1855`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.040657`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.026092`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.014656`
- `lag_12__T_place_TSIDEUPPER`: contribution `-0.010649`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.008487`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.040657`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.026092`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.014656`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.008487`

### tick `39083`, seconds `37.50`, LSTM delta `-0.1587`

Top all feature movements:
- `lag_07__T_place_TSIDELOWER`: contribution `-0.015389`
- `lag_07__T_place_RAMP`: contribution `-0.010854`
- `lag_13__T_place_TSIDELOWER`: contribution `-0.006617`
- `lag_12__CT1__is_scoped`: contribution `-0.006462`
- `lag_00__T_kills_last_3s`: contribution `-0.005395`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39627`, seconds `46.00`, LSTM delta `-0.0675`

Top all feature movements:
- `lag_13__T_place_TSIDELOWER`: contribution `-0.013235`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.010730`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.007283`
- `lag_13__T_place_TSIDEUPPER`: contribution `-0.006605`
- `lag_00__T_kills_last_3s`: contribution `-0.005395`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.010730`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.007283`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.003374`
