# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `30174`, seconds `65.00`, LSTM `0.8318`, delta `+0.1029`
- tick `28510`, seconds `39.00`, LSTM `0.7301`, delta `-0.0576`
- tick `30942`, seconds `77.00`, LSTM `0.9481`, delta `+0.0562`
- tick `28670`, seconds `41.50`, LSTM `0.7494`, delta `+0.0493`
- tick `28702`, seconds `42.00`, LSTM `0.7849`, delta `+0.0354`
- tick `28222`, seconds `34.50`, LSTM `0.7891`, delta `-0.0347`
- tick `30398`, seconds `68.50`, LSTM `0.8813`, delta `+0.0347`
- tick `27966`, seconds `30.50`, LSTM `0.7688`, delta `-0.0339`
- tick `29086`, seconds `48.00`, LSTM `0.6931`, delta `-0.0309`
- tick `28030`, seconds `31.50`, LSTM `0.7807`, delta `+0.0295`

## Top 15 local ridge features

- `lag_00__T_place_TOPOFMID`: coefficient `-0.001285`, |coef| `0.001285`
- `lag_00__T3__is_walking`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001250`, |coef| `0.001250`
- `lag_07__CT5__is_walking`: coefficient `0.001205`, |coef| `0.001205`
- `lag_08__T_place_MIDDLE`: coefficient `-0.001200`, |coef| `0.001200`
- `lag_15__T4__duck_amount`: coefficient `-0.001165`, |coef| `0.001165`
- `lag_03__T_place_MIDDLE`: coefficient `-0.001161`, |coef| `0.001161`
- `lag_03__T4__duck_amount`: coefficient `0.001156`, |coef| `0.001156`
- `lag_00__T_place_UPSTAIRS`: coefficient `0.001155`, |coef| `0.001155`
- `lag_00__CT_kills_last_3s`: coefficient `0.001094`, |coef| `0.001094`
- `lag_14__T_place_APARTMENTS`: coefficient `-0.001087`, |coef| `0.001087`
- `lag_00__damage_diff_last_5s`: coefficient `0.001081`, |coef| `0.001081`
- `lag_00__T1__alive`: coefficient `-0.001076`, |coef| `0.001076`
- `lag_00__T1__hp`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_00__CT_damage_last_5s`: coefficient `0.001050`, |coef| `0.001050`

## Top 10 utility ridge features

- `lag_15__CT3__smoke`: coefficient `0.000693` (raises CT win probability)
- `lag_14__CT_A_site_active_smokes`: coefficient `0.000665` (raises CT win probability)
- `lag_14__CT_active_smokes`: coefficient `0.000646` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000632` (raises CT win probability)
- `lag_13__CT_flashes_last_5s`: coefficient `0.000617` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000597` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000518` (raises CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `0.000494` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `-0.000471` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000426` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_TOPOFMID`: coefficient `-0.001285` (lowers CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.001272` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001250` (raises CT win probability)
- `lag_07__CT5__is_walking`: coefficient `0.001205` (raises CT win probability)
- `lag_08__T_place_MIDDLE`: coefficient `-0.001200` (lowers CT win probability)
- `lag_15__T4__duck_amount`: coefficient `-0.001165` (lowers CT win probability)
- `lag_03__T_place_MIDDLE`: coefficient `-0.001161` (lowers CT win probability)
- `lag_03__T4__duck_amount`: coefficient `0.001156` (raises CT win probability)
- `lag_00__T_place_UPSTAIRS`: coefficient `0.001155` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001094` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `30174`, seconds `65.00`, LSTM delta `+0.1029`

Top all feature movements:
- `lag_15__T4__duck_amount`: contribution `+0.004121`
- `lag_11__CT1__is_scoped`: contribution `+0.004096`
- `lag_08__T_place_TOPOFMID`: contribution `+0.003947`
- `lag_08__T_place_MIDDLE`: contribution `+0.003900`
- `lag_07__CT5__duck_amount`: contribution `+0.003398`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28510`, seconds `39.00`, LSTM delta `-0.0576`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `-0.007247`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006948`
- `lag_14__T_place_BALCONY`: contribution `-0.005347`
- `lag_14__T_place_SECONDMID`: contribution `-0.003148`
- `lag_00__T3__is_walking`: contribution `-0.002955`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30942`, seconds `77.00`, LSTM delta `+0.0562`

Top all feature movements:
- `lag_08__T_place_ARCH`: contribution `+0.007032`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003474`
- `lag_00__CT2__duck_amount`: contribution `+0.003399`
- `lag_00__CT_kills_last_3s`: contribution `+0.003159`
- `lag_00__CT1__is_scoped`: contribution `+0.003045`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.001307`

### tick `28670`, seconds `41.50`, LSTM delta `+0.0493`

Top all feature movements:
- `lag_00__T_place_UPSTAIRS`: contribution `+0.019480`
- `lag_04__T_walking_count`: contribution `+0.002811`
- `lag_15__T5__duck_amount`: contribution `+0.002160`
- `lag_03__T1__is_walking`: contribution `-0.002149`
- `lag_14__T2__is_walking`: contribution `+0.002038`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28702`, seconds `42.00`, LSTM delta `+0.0354`

Top all feature movements:
- `lag_01__T_place_UPSTAIRS`: contribution `+0.016531`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003474`
- `lag_00__T3__is_walking`: contribution `+0.002955`
- `lag_03__T1__is_walking`: contribution `+0.002149`
- `lag_00__CT3__duck_amount`: contribution `+0.001944`

Top utility-only movements:
- No utility movement among the top local contributors.
