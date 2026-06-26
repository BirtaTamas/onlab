# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `74548`, seconds `90.50`, LSTM `0.8868`, delta `+0.1399`
- tick `72948`, seconds `65.50`, LSTM `0.6916`, delta `+0.0894`
- tick `74580`, seconds `91.00`, LSTM `0.9512`, delta `+0.0644`
- tick `74516`, seconds `90.00`, LSTM `0.7470`, delta `+0.0509`
- tick `68916`, seconds `2.50`, LSTM `0.6498`, delta `-0.0455`
- tick `69876`, seconds `17.50`, LSTM `0.6457`, delta `-0.0424`
- tick `69556`, seconds `12.50`, LSTM `0.6551`, delta `+0.0324`
- tick `70804`, seconds `32.00`, LSTM `0.6355`, delta `+0.0322`
- tick `73332`, seconds `71.50`, LSTM `0.7357`, delta `+0.0298`
- tick `69748`, seconds `15.50`, LSTM `0.6839`, delta `-0.0295`

## Top 15 local ridge features

- `lag_02__CT_place_TSIDELOWER`: coefficient `0.002296`, |coef| `0.002296`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001508`, |coef| `0.001508`
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.001443`, |coef| `0.001443`
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.001274`, |coef| `0.001274`
- `lag_00__CT3__is_scoped`: coefficient `0.001197`, |coef| `0.001197`
- `lag_00__CT_kills_last_3s`: coefficient `0.001136`, |coef| `0.001136`
- `lag_11__T_place_CTSPAWN`: coefficient `0.001078`, |coef| `0.001078`
- `lag_01__CT3__is_scoped`: coefficient `0.001069`, |coef| `0.001069`
- `lag_00__kill_diff_last_3s`: coefficient `0.000971`, |coef| `0.000971`
- `lag_00__CT_damage_last_5s`: coefficient `0.000944`, |coef| `0.000944`
- `lag_00__T_place_HOUSE`: coefficient `-0.000921`, |coef| `0.000921`
- `lag_04__CT_place_TSIDELOWER`: coefficient `0.000899`, |coef| `0.000899`
- `lag_12__CT3__is_scoped`: coefficient `-0.000840`, |coef| `0.000840`
- `lag_00__CT4__is_walking`: coefficient `-0.000820`, |coef| `0.000820`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.000784`, |coef| `0.000784`

## Top 10 utility ridge features

- `lag_02__CT_A_site_active_infernos`: coefficient `0.000782` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_01__T_mollies_last_5s`: coefficient `-0.000643` (lowers CT win probability)
- `lag_05__CT3__molly`: coefficient `-0.000588` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000580` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000553` (raises CT win probability)
- `lag_03__T_smokes_last_5s`: coefficient `-0.000458` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000434` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.000431` (raises CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.000431` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_TSIDELOWER`: coefficient `0.002296` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001508` (raises CT win probability)
- `lag_01__CT_place_TSIDELOWER`: coefficient `0.001443` (raises CT win probability)
- `lag_03__CT_place_TSIDELOWER`: coefficient `0.001274` (raises CT win probability)
- `lag_00__CT3__is_scoped`: coefficient `0.001197` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001136` (raises CT win probability)
- `lag_11__T_place_CTSPAWN`: coefficient `0.001078` (raises CT win probability)
- `lag_01__CT3__is_scoped`: coefficient `0.001069` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000971` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000944` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `74548`, seconds `90.50`, LSTM delta `+0.1399`

Top all feature movements:
- `lag_02__CT_place_TSIDELOWER`: contribution `+0.031194`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005239`
- `lag_11__T_place_CTSPAWN`: contribution `+0.005144`
- `lag_01__CT3__is_scoped`: contribution `+0.004863`
- `lag_00__CT_kills_last_3s`: contribution `+0.003280`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `+0.002758`
- `lag_05__CT3__molly`: contribution `+0.001453`

### tick `72948`, seconds `65.50`, LSTM delta `+0.0894`

Top all feature movements:
- `lag_14__T3__flash_duration`: contribution `+0.005501`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005239`
- `lag_00__CT_kills_last_3s`: contribution `+0.003280`
- `lag_07__T_place_SIDEENTRANCE`: contribution `+0.002867`
- `lag_12__CT_place_SIDEHALL`: contribution `+0.002796`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `+0.005501`
- `lag_01__T_B_site_active_infernos`: contribution `+0.001640`
- `lag_03__T_A_site_active_infernos`: contribution `+0.001242`
- `lag_03__T_B_site_active_infernos`: contribution `+0.001001`

### tick `74580`, seconds `91.00`, LSTM delta `+0.0644`

Top all feature movements:
- `lag_03__CT_place_TSIDELOWER`: contribution `+0.017313`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005239`
- `lag_00__CT_kills_last_3s`: contribution `+0.003280`
- `lag_12__T_place_CTSPAWN`: contribution `+0.002910`
- `lag_00__T_place_CTSPAWN`: contribution `+0.002886`

Top utility-only movements:
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001521`

### tick `74516`, seconds `90.00`, LSTM delta `+0.0509`

Top all feature movements:
- `lag_01__CT_place_TSIDELOWER`: contribution `+0.019598`
- `lag_00__CT3__is_scoped`: contribution `+0.005445`
- `lag_00__T_place_HOUSE`: contribution `+0.004050`
- `lag_12__CT3__is_scoped`: contribution `+0.003822`
- `lag_10__T_place_CTSPAWN`: contribution `+0.003353`

Top utility-only movements:
- `lag_01__CT_A_site_active_infernos`: contribution `+0.001950`
- `lag_04__CT3__molly`: contribution `+0.000977`

### tick `68916`, seconds `2.50`, LSTM delta `-0.0455`

Top all feature movements:
- `lag_01__T_mollies_last_5s`: contribution `-0.013218`
- `lag_03__T_smokes_last_5s`: contribution `-0.006717`
- `lag_00__T_flashes_last_5s`: contribution `-0.002565`
- `lag_05__CT_place_MAINHALL`: contribution `-0.001615`
- `lag_03__CT_place_MAINHALL`: contribution `-0.001462`

Top utility-only movements:
- `lag_01__T_mollies_last_5s`: contribution `-0.013218`
- `lag_03__T_smokes_last_5s`: contribution `-0.006717`
- `lag_00__T_flashes_last_5s`: contribution `-0.002565`
- `lag_05__CT3__molly`: contribution `-0.001152`
- `lag_00__T1__flash`: contribution `-0.000966`
