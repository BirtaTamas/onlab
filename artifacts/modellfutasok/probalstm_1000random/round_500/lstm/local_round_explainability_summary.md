# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `23861`, seconds `5.50`, LSTM `0.1729`, delta `-0.1718`
- tick `23893`, seconds `6.00`, LSTM `0.1176`, delta `-0.0553`
- tick `23829`, seconds `5.00`, LSTM `0.3448`, delta `-0.0450`
- tick `24149`, seconds `10.00`, LSTM `0.0995`, delta `+0.0442`
- tick `24501`, seconds `15.50`, LSTM `0.1058`, delta `-0.0304`
- tick `25493`, seconds `31.00`, LSTM `0.1492`, delta `+0.0299`
- tick `25557`, seconds `32.00`, LSTM `0.1521`, delta `+0.0285`
- tick `23605`, seconds `1.50`, LSTM `0.4146`, delta `-0.0277`
- tick `23925`, seconds `6.50`, LSTM `0.0914`, delta `-0.0262`
- tick `25525`, seconds `31.50`, LSTM `0.1236`, delta `-0.0255`

## Top 15 local ridge features

- `lag_06__T_mollies_last_5s`: coefficient `-0.002091`, |coef| `0.002091`
- `lag_05__T_mollies_last_5s`: coefficient `-0.000893`, |coef| `0.000893`
- `lag_05__CT_place_MIDDOORS`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_02__CT4__is_scoped`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_08__CT_place_UNDERA`: coefficient `-0.000819`, |coef| `0.000819`
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000743`, |coef| `0.000743`
- `lag_05__T_place_OUTSIDETUNNEL`: coefficient `-0.000731`, |coef| `0.000731`
- `lag_07__T_mollies_last_5s`: coefficient `-0.000692`, |coef| `0.000692`
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.000676`, |coef| `0.000676`
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.000669`, |coef| `0.000669`
- `lag_07__T5__is_scoped`: coefficient `-0.000661`, |coef| `0.000661`
- `lag_04__T_place_SHORTSTAIRS`: coefficient `-0.000658`, |coef| `0.000658`
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.000655`, |coef| `0.000655`
- `lag_00__T_place_TSPAWN`: coefficient `0.000654`, |coef| `0.000654`
- `lag_11__CT_place_CTSPAWN`: coefficient `-0.000632`, |coef| `0.000632`

## Top 10 utility ridge features

- `lag_06__T_mollies_last_5s`: coefficient `-0.002091` (lowers CT win probability)
- `lag_05__T_mollies_last_5s`: coefficient `-0.000893` (lowers CT win probability)
- `lag_07__T_mollies_last_5s`: coefficient `-0.000692` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000561` (raises CT win probability)
- `lag_11__CT1__flash`: coefficient `-0.000548` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000455` (raises CT win probability)
- `lag_11__CT1__utility_total`: coefficient `-0.000405` (lowers CT win probability)
- `lag_11__T1__molly`: coefficient `-0.000405` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `0.000400` (raises CT win probability)
- `lag_11__T_flash_inv`: coefficient `-0.000400` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_MIDDOORS`: coefficient `-0.000848` (lowers CT win probability)
- `lag_02__CT4__is_scoped`: coefficient `-0.000829` (lowers CT win probability)
- `lag_08__CT_place_UNDERA`: coefficient `-0.000819` (lowers CT win probability)
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000743` (raises CT win probability)
- `lag_05__T_place_OUTSIDETUNNEL`: coefficient `-0.000731` (lowers CT win probability)
- `lag_03__T_place_SHORTSTAIRS`: coefficient `-0.000676` (lowers CT win probability)
- `lag_00__T_place_SHORTSTAIRS`: coefficient `-0.000669` (lowers CT win probability)
- `lag_07__T5__is_scoped`: coefficient `-0.000661` (lowers CT win probability)
- `lag_04__T_place_SHORTSTAIRS`: coefficient `-0.000658` (lowers CT win probability)
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.000655` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23861`, seconds `5.50`, LSTM delta `-0.1718`

Top all feature movements:
- `lag_06__T_mollies_last_5s`: contribution `-0.042994`
- `lag_08__CT_place_UNDERA`: contribution `-0.005003`
- `lag_05__CT_place_MIDDOORS`: contribution `-0.004897`
- `lag_05__T_place_OUTSIDETUNNEL`: contribution `-0.003653`
- `lag_07__T5__is_scoped`: contribution `-0.003154`

Top utility-only movements:
- `lag_06__T_mollies_last_5s`: contribution `-0.042994`
- `lag_11__CT1__flash`: contribution `-0.001570`
- `lag_00__CT5__smoke`: contribution `-0.001230`

### tick `23893`, seconds `6.00`, LSTM delta `-0.0553`

Top all feature movements:
- `lag_07__T_mollies_last_5s`: contribution `-0.014221`
- `lag_08__CT_place_UNDERA`: contribution `-0.002501`
- `lag_08__T5__is_scoped`: contribution `-0.002197`
- `lag_06__CT_place_MIDDOORS`: contribution `-0.002119`
- `lag_00__T_shots_fired_sum`: contribution `+0.001804`

Top utility-only movements:
- `lag_07__T_mollies_last_5s`: contribution `-0.014221`

### tick `23829`, seconds `5.00`, LSTM delta `-0.0450`

Top all feature movements:
- `lag_05__T_mollies_last_5s`: contribution `-0.018364`
- `lag_07__CT_place_UNDERA`: contribution `-0.003481`
- `lag_02__CT_place_UNDERA`: contribution `-0.001782`
- `lag_01__CT4__is_scoped`: contribution `-0.001632`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `-0.001620`

Top utility-only movements:
- `lag_05__T_mollies_last_5s`: contribution `-0.018364`
- `lag_10__CT1__flash`: contribution `-0.000576`
- `lag_05__T3__molly`: contribution `-0.000425`
- `lag_10__T1__molly`: contribution `-0.000372`

### tick `24149`, seconds `10.00`, LSTM delta `+0.0442`

Top all feature movements:
- `lag_05__T_mollies_last_5s`: contribution `+0.018364`
- `lag_15__T_mollies_last_5s`: contribution `+0.007302`
- `lag_05__T_place_OUTSIDETUNNEL`: contribution `+0.003653`
- `lag_03__CT_place_LONGA`: contribution `-0.001331`
- `lag_03__CT_place_UNDERA`: contribution `-0.001203`

Top utility-only movements:
- `lag_05__T_mollies_last_5s`: contribution `+0.018364`
- `lag_15__T_mollies_last_5s`: contribution `+0.007302`
- `lag_05__CT_flash_duration_sum`: contribution `+0.000720`
- `lag_05__CT5__flash_duration`: contribution `+0.000700`
- `lag_00__CT_B_site_active_infernos`: contribution `+0.000587`

### tick `24501`, seconds `15.50`, LSTM delta `-0.0304`

Top all feature movements:
- `lag_03__CT_place_LONGDOORS`: contribution `-0.002182`
- `lag_08__CT3__duck_amount`: contribution `-0.001300`
- `lag_08__T3__duck_amount`: contribution `-0.001067`
- `lag_03__CT_place_PIT`: contribution `-0.000906`
- `lag_04__CT5__duck_amount`: contribution `-0.000831`

Top utility-only movements:
- `lag_00__CT3__smoke`: contribution `-0.000757`
- `lag_13__CT2__flash_duration`: contribution `-0.000740`
- `lag_13__CT_flash_duration_sum`: contribution `-0.000671`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.000621`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.000587`
