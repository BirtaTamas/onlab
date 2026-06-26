# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `32860`, seconds `15.50`, LSTM `0.8442`, delta `+0.1392`
- tick `36252`, seconds `68.50`, LSTM `0.9002`, delta `+0.1105`
- tick `36220`, seconds `68.00`, LSTM `0.7897`, delta `+0.0678`
- tick `36380`, seconds `70.50`, LSTM `0.9547`, delta `+0.0591`
- tick `36028`, seconds `65.00`, LSTM `0.7147`, delta `+0.0534`
- tick `36156`, seconds `67.00`, LSTM `0.7146`, delta `+0.0403`
- tick `32796`, seconds `14.50`, LSTM `0.6809`, delta `+0.0388`
- tick `35484`, seconds `56.50`, LSTM `0.6516`, delta `-0.0379`
- tick `35964`, seconds `64.00`, LSTM `0.6637`, delta `+0.0342`
- tick `33052`, seconds `18.50`, LSTM `0.7880`, delta `-0.0327`

## Top 15 local ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.001133`, |coef| `0.001133`
- `lag_07__T_place_DECON`: coefficient `-0.001044`, |coef| `0.001044`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001016`, |coef| `0.001016`
- `lag_00__CT_kills_last_3s`: coefficient `0.001009`, |coef| `0.001009`
- `lag_05__T_place_OBSERVATION`: coefficient `-0.001004`, |coef| `0.001004`
- `lag_00__CT_place_HUTROOF`: coefficient `0.000951`, |coef| `0.000951`
- `lag_02__CT_place_HEAVEN`: coefficient `-0.000830`, |coef| `0.000830`
- `lag_05__T_place_DECON`: coefficient `-0.000822`, |coef| `0.000822`
- `lag_04__T_place_OBSERVATION`: coefficient `-0.000803`, |coef| `0.000803`
- `lag_14__T_place_DECON`: coefficient `0.000797`, |coef| `0.000797`
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000786`, |coef| `0.000786`
- `lag_01__T4__duck_amount`: coefficient `0.000781`, |coef| `0.000781`
- `lag_15__CT_place_HELL`: coefficient `-0.000768`, |coef| `0.000768`
- `lag_15__CT_place_HEAVEN`: coefficient `0.000767`, |coef| `0.000767`
- `lag_09__CT_B_site_active_infernos`: coefficient `0.000764`, |coef| `0.000764`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_infernos`: coefficient `0.000786` (raises CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `0.000764` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `-0.000701` (lowers CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.000694` (raises CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000670` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000663` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `0.000505` (raises CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.000486` (lowers CT win probability)
- `lag_12__T2__flash_duration`: coefficient `0.000451` (raises CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `0.000441` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.001133` (lowers CT win probability)
- `lag_07__T_place_DECON`: coefficient `-0.001044` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001016` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001009` (raises CT win probability)
- `lag_05__T_place_OBSERVATION`: coefficient `-0.001004` (lowers CT win probability)
- `lag_00__CT_place_HUTROOF`: coefficient `0.000951` (raises CT win probability)
- `lag_02__CT_place_HEAVEN`: coefficient `-0.000830` (lowers CT win probability)
- `lag_05__T_place_DECON`: coefficient `-0.000822` (lowers CT win probability)
- `lag_04__T_place_OBSERVATION`: coefficient `-0.000803` (lowers CT win probability)
- `lag_14__T_place_DECON`: coefficient `0.000797` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `32860`, seconds `15.50`, LSTM delta `+0.1392`

Top all feature movements:
- `lag_12__T_flash_duration_sum`: contribution `+0.005918`
- `lag_12__T_flashed_players`: contribution `+0.005195`
- `lag_01__T2__flash_duration`: contribution `+0.004491`
- `lag_02__CT_place_HEAVEN`: contribution `+0.004484`
- `lag_03__CT_place_MINI`: contribution `+0.004240`

Top utility-only movements:
- `lag_12__T_flash_duration_sum`: contribution `+0.005918`
- `lag_01__T2__flash_duration`: contribution `+0.004491`
- `lag_01__T1__flash_duration`: contribution `+0.004011`
- `lag_01__T_flash_duration_sum`: contribution `+0.003440`
- `lag_12__T2__flash_duration`: contribution `+0.002890`

### tick `36252`, seconds `68.50`, LSTM delta `+0.1105`

Top all feature movements:
- `lag_05__T_place_OBSERVATION`: contribution `+0.016996`
- `lag_03__T_place_DECON`: contribution `+0.008393`
- `lag_07__T_place_OBSERVATION`: contribution `+0.008175`
- `lag_04__T_place_TROPHY`: contribution `+0.004240`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003530`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `+0.002773`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.002624`

### tick `36220`, seconds `68.00`, LSTM delta `+0.0678`

Top all feature movements:
- `lag_04__T_place_OBSERVATION`: contribution `+0.013603`
- `lag_02__T_place_DECON`: contribution `+0.008290`
- `lag_06__T_place_OBSERVATION`: contribution `+0.006867`
- `lag_02__CT_place_ADMIN`: contribution `+0.003390`
- `lag_00__T_place_VENDING`: contribution `+0.002805`

Top utility-only movements:
- `lag_08__CT_A_site_active_infernos`: contribution `+0.001417`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.001342`

### tick `36380`, seconds `70.50`, LSTM delta `+0.0591`

Top all feature movements:
- `lag_07__T_place_DECON`: contribution `+0.016774`
- `lag_07__CT_place_HELL`: contribution `+0.004013`
- `lag_00__CT_kills_last_3s`: contribution `+0.002913`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002824`
- `lag_07__CT_place_ADMIN`: contribution `+0.002613`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36028`, seconds `65.00`, LSTM delta `+0.0534`

Top all feature movements:
- `lag_00__T_place_OBSERVATION`: contribution `+0.019183`
- `lag_01__T4__duck_amount`: contribution `+0.002516`
- `lag_02__CT4__duck_amount`: contribution `-0.001794`
- `lag_04__CT1__duck_amount`: contribution `+0.001784`
- `lag_00__CT4__is_walking`: contribution `-0.001535`

Top utility-only movements:
- `lag_02__CT_A_site_active_infernos`: contribution `+0.001459`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.001379`
- `lag_05__T_A_site_active_infernos`: contribution `+0.001174`
- `lag_05__T_B_site_active_infernos`: contribution `+0.001058`
