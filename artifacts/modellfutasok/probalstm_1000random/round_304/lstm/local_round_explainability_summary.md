# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `9`

## Largest probability jumps

- tick `90507`, seconds `99.00`, LSTM `0.1123`, delta `-0.1869`
- tick `90731`, seconds `102.50`, LSTM `0.0488`, delta `-0.1639`
- tick `89707`, seconds `86.50`, LSTM `0.4048`, delta `-0.1574`
- tick `90475`, seconds `98.50`, LSTM `0.2992`, delta `+0.1438`
- tick `89835`, seconds `88.50`, LSTM `0.2608`, delta `-0.1410`
- tick `90667`, seconds `101.50`, LSTM `0.2113`, delta `+0.0729`
- tick `89963`, seconds `90.50`, LSTM `0.2094`, delta `-0.0672`
- tick `88811`, seconds `72.50`, LSTM `0.4754`, delta `-0.0662`
- tick `89643`, seconds `85.50`, LSTM `0.5618`, delta `+0.0610`
- tick `89899`, seconds `89.50`, LSTM `0.2636`, delta `+0.0573`

## Top 15 local ridge features

- `lag_04__CT_place_TROPHY`: coefficient `-0.002479`, |coef| `0.002479`
- `lag_00__kill_diff_last_3s`: coefficient `0.002296`, |coef| `0.002296`
- `lag_00__T_kills_last_3s`: coefficient `-0.002035`, |coef| `0.002035`
- `lag_11__CT_place_CONTROL`: coefficient `0.001998`, |coef| `0.001998`
- `lag_00__damage_diff_last_5s`: coefficient `0.001915`, |coef| `0.001915`
- `lag_05__CT_place_ADMIN`: coefficient `0.001692`, |coef| `0.001692`
- `lag_00__T_place_MINI`: coefficient `-0.001679`, |coef| `0.001679`
- `lag_00__T_damage_last_5s`: coefficient `-0.001625`, |coef| `0.001625`
- `lag_00__T3__duck_amount`: coefficient `0.001533`, |coef| `0.001533`
- `lag_03__CT_place_TROPHY`: coefficient `0.001533`, |coef| `0.001533`
- `lag_01__T_place_MINI`: coefficient `-0.001477`, |coef| `0.001477`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001432`, |coef| `0.001432`
- `lag_15__T_place_SQUEAKY`: coefficient `-0.001273`, |coef| `0.001273`
- `lag_03__CT_place_CONTROL`: coefficient `-0.001264`, |coef| `0.001264`
- `lag_00__CT_place_VENDING`: coefficient `0.001258`, |coef| `0.001258`

## Top 10 utility ridge features

- `lag_15__T3__flash_duration`: coefficient `0.000871` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.000797` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `-0.000732` (lowers CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000687` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000603` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `-0.000574` (lowers CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000570` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000496` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `0.000479` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000451` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_TROPHY`: coefficient `-0.002479` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002296` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002035` (lowers CT win probability)
- `lag_11__CT_place_CONTROL`: coefficient `0.001998` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001915` (raises CT win probability)
- `lag_05__CT_place_ADMIN`: coefficient `0.001692` (raises CT win probability)
- `lag_00__T_place_MINI`: coefficient `-0.001679` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001625` (lowers CT win probability)
- `lag_00__T3__duck_amount`: coefficient `0.001533` (raises CT win probability)
- `lag_03__CT_place_TROPHY`: coefficient `0.001533` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `90507`, seconds `99.00`, LSTM delta `-0.1869`

Top all feature movements:
- `lag_04__CT_place_TROPHY`: contribution `-0.036611`
- `lag_03__T_place_MINI`: contribution `-0.014066`
- `lag_12__CT_place_CONTROL`: contribution `-0.012295`
- `lag_00__CT_shots_fired_sum`: contribution `-0.009950`
- `lag_04__CT_place_CONTROL`: contribution `-0.009671`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `-0.005085`

### tick `90731`, seconds `102.50`, LSTM delta `-0.1639`

Top all feature movements:
- `lag_00__CT_place_VENDING`: contribution `-0.021554`
- `lag_11__CT_place_CONTROL`: contribution `-0.020738`
- `lag_06__CT_place_VENDING`: contribution `-0.015965`
- `lag_11__CT_place_TROPHY`: contribution `-0.014713`
- `lag_06__CT_place_TROPHY`: contribution `-0.013210`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.001426`

### tick `89707`, seconds `86.50`, LSTM delta `-0.1574`

Top all feature movements:
- `lag_05__CT_place_ADMIN`: contribution `-0.011756`
- `lag_15__T_place_SQUEAKY`: contribution `-0.007923`
- `lag_00__T_kills_last_3s`: contribution `-0.006447`
- `lag_02__T_place_SQUEAKY`: contribution `-0.005874`
- `lag_08__CT_place_ADMIN`: contribution `-0.005849`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90475`, seconds `98.50`, LSTM delta `+0.1438`

Top all feature movements:
- `lag_03__CT_place_TROPHY`: contribution `+0.022639`
- `lag_11__CT_place_CONTROL`: contribution `+0.020738`
- `lag_03__CT_place_CONTROL`: contribution `+0.013121`
- `lag_02__T_place_MINI`: contribution `+0.010893`
- `lag_11__T_place_SILO`: contribution `+0.006327`

Top utility-only movements:
- `lag_10__T3__flash_duration`: contribution `+0.002903`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.002027`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.001957`

### tick `89835`, seconds `88.50`, LSTM delta `-0.1410`

Top all feature movements:
- `lag_00__T_place_MINI`: contribution `-0.023358`
- `lag_00__T3__duck_amount`: contribution `-0.005780`
- `lag_00__kill_diff_last_3s`: contribution `-0.005525`
- `lag_12__CT_place_ADMIN`: contribution `-0.005150`
- `lag_03__T5__is_scoped`: contribution `-0.004438`

Top utility-only movements:
- `lag_15__T3__flash_duration`: contribution `-0.003248`
