# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `104922`, seconds `16.50`, LSTM `0.0846`, delta `-0.1265`
- tick `104314`, seconds `7.00`, LSTM `0.3263`, delta `-0.0548`
- tick `104986`, seconds `17.50`, LSTM `0.0298`, delta `-0.0481`
- tick `104218`, seconds `5.50`, LSTM `0.3829`, delta `+0.0421`
- tick `104346`, seconds `7.50`, LSTM `0.2888`, delta `-0.0374`
- tick `104058`, seconds `3.00`, LSTM `0.3414`, delta `-0.0311`
- tick `104410`, seconds `8.50`, LSTM `0.2403`, delta `-0.0287`
- tick `103994`, seconds `2.00`, LSTM `0.3920`, delta `+0.0247`
- tick `103898`, seconds `0.50`, LSTM `0.4013`, delta `-0.0243`
- tick `103962`, seconds `1.50`, LSTM `0.3672`, delta `-0.0242`

## Top 15 local ridge features

- `lag_10__T_flashed_players`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_10__CT_place_APARTMENTS`: coefficient `-0.000847`, |coef| `0.000847`
- `lag_07__CT_place_APARTMENTS`: coefficient `-0.000840`, |coef| `0.000840`
- `lag_10__T4__flash_duration`: coefficient `-0.000736`, |coef| `0.000736`
- `lag_07__CT_place_BALCONY`: coefficient `0.000724`, |coef| `0.000724`
- `lag_11__CT_place_BALCONY`: coefficient `-0.000709`, |coef| `0.000709`
- `lag_10__T_flash_duration_sum`: coefficient `-0.000671`, |coef| `0.000671`
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.000631`, |coef| `0.000631`
- `lag_02__CT_place_APARTMENTS`: coefficient `-0.000617`, |coef| `0.000617`
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000578`, |coef| `0.000578`
- `lag_09__CT_place_APARTMENTS`: coefficient `-0.000568`, |coef| `0.000568`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000568`, |coef| `0.000568`
- `lag_04__CT_place_APARTMENTS`: coefficient `-0.000557`, |coef| `0.000557`
- `lag_10__T_place_TRAMP`: coefficient `-0.000553`, |coef| `0.000553`
- `lag_03__CT_place_APARTMENTS`: coefficient `-0.000553`, |coef| `0.000553`

## Top 10 utility ridge features

- `lag_10__T4__flash_duration`: coefficient `-0.000736` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.000671` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000539` (lowers CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.000483` (lowers CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.000454` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000427` (raises CT win probability)
- `lag_07__T_mollies_last_5s`: coefficient `0.000405` (raises CT win probability)
- `lag_10__T_mollies_last_5s`: coefficient `-0.000391` (lowers CT win probability)
- `lag_13__T_smokes_last_5s`: coefficient `-0.000379` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000375` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_flashed_players`: coefficient `-0.000974` (lowers CT win probability)
- `lag_10__CT_place_APARTMENTS`: coefficient `-0.000847` (lowers CT win probability)
- `lag_07__CT_place_APARTMENTS`: coefficient `-0.000840` (lowers CT win probability)
- `lag_07__CT_place_BALCONY`: coefficient `0.000724` (raises CT win probability)
- `lag_11__CT_place_BALCONY`: coefficient `-0.000709` (lowers CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.000631` (lowers CT win probability)
- `lag_02__CT_place_APARTMENTS`: coefficient `-0.000617` (lowers CT win probability)
- `lag_00__CT_place_CTSPAWN`: coefficient `0.000578` (raises CT win probability)
- `lag_09__CT_place_APARTMENTS`: coefficient `-0.000568` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000568` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `104922`, seconds `16.50`, LSTM delta `-0.1265`

Top all feature movements:
- `lag_10__T_flashed_players`: contribution `-0.009400`
- `lag_10__T4__flash_duration`: contribution `-0.004846`
- `lag_07__CT_place_BALCONY`: contribution `-0.004649`
- `lag_11__CT_place_BALCONY`: contribution `-0.004550`
- `lag_10__T_flash_duration_sum`: contribution `-0.003905`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.004846`
- `lag_10__T_flash_duration_sum`: contribution `-0.003905`
- `lag_09__T1__flash_duration`: contribution `-0.002900`
- `lag_10__T2__flash_duration`: contribution `-0.002033`
- `lag_09__T_flash_duration_sum`: contribution `-0.001417`

### tick `104314`, seconds `7.00`, LSTM delta `-0.0548`

Top all feature movements:
- `lag_00__T_mollies_last_5s`: contribution `-0.009341`
- `lag_10__T_mollies_last_5s`: contribution `-0.008043`
- `lag_13__T_smokes_last_5s`: contribution `-0.005560`
- `lag_03__T_smokes_last_5s`: contribution `-0.004165`
- `lag_13__T_he_last_5s`: contribution `-0.003380`

Top utility-only movements:
- `lag_00__T_mollies_last_5s`: contribution `-0.009341`
- `lag_10__T_mollies_last_5s`: contribution `-0.008043`
- `lag_13__T_smokes_last_5s`: contribution `-0.005560`
- `lag_03__T_smokes_last_5s`: contribution `-0.004165`
- `lag_13__T_he_last_5s`: contribution `-0.003380`

### tick `104986`, seconds `17.50`, LSTM delta `-0.0481`

Top all feature movements:
- `lag_11__CT_place_BALCONY`: contribution `-0.004550`
- `lag_12__T_flashed_players`: contribution `-0.003667`
- `lag_00__T_shots_fired_sum`: contribution `-0.002556`
- `lag_09__CT_place_BALCONY`: contribution `+0.002458`
- `lag_09__CT_place_APARTMENTS`: contribution `-0.002184`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.001621`
- `lag_00__T4__flash_duration`: contribution `-0.001600`
- `lag_12__T_flash_duration_sum`: contribution `-0.001288`
- `lag_11__T_flash_duration_sum`: contribution `-0.000966`
- `lag_11__T1__flash_duration`: contribution `-0.000945`

### tick `104218`, seconds `5.50`, LSTM delta `+0.0421`

Top all feature movements:
- `lag_07__T_mollies_last_5s`: contribution `+0.008330`
- `lag_00__T_he_last_5s`: contribution `+0.004770`
- `lag_07__T_smokes_last_5s`: contribution `+0.003311`
- `lag_06__T_he_last_5s`: contribution `+0.002891`
- `lag_00__T_flashes_last_5s`: contribution `+0.002299`

Top utility-only movements:
- `lag_07__T_mollies_last_5s`: contribution `+0.008330`
- `lag_00__T_he_last_5s`: contribution `+0.004770`
- `lag_07__T_smokes_last_5s`: contribution `+0.003311`
- `lag_06__T_he_last_5s`: contribution `+0.002891`
- `lag_00__T_flashes_last_5s`: contribution `+0.002299`

### tick `104346`, seconds `7.50`, LSTM delta `-0.0374`

Top all feature movements:
- `lag_11__T_mollies_last_5s`: contribution `-0.005026`
- `lag_00__T_he_last_5s`: contribution `+0.004770`
- `lag_14__T_he_last_5s`: contribution `-0.003510`
- `lag_14__T_smokes_last_5s`: contribution `-0.002776`
- `lag_00__T_flashes_last_5s`: contribution `+0.002299`

Top utility-only movements:
- `lag_11__T_mollies_last_5s`: contribution `-0.005026`
- `lag_00__T_he_last_5s`: contribution `+0.004770`
- `lag_14__T_he_last_5s`: contribution `-0.003510`
- `lag_14__T_smokes_last_5s`: contribution `-0.002776`
- `lag_00__T_flashes_last_5s`: contribution `+0.002299`
