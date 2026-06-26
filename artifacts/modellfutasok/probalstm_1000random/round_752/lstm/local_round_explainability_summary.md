# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `75796`, seconds `34.50`, LSTM `0.7069`, delta `-0.1525`
- tick `75732`, seconds `33.50`, LSTM `0.8688`, delta `+0.1525`
- tick `78644`, seconds `79.00`, LSTM `0.9213`, delta `+0.1346`
- tick `78804`, seconds `81.50`, LSTM `0.9659`, delta `+0.1221`
- tick `75476`, seconds `29.50`, LSTM `0.8007`, delta `+0.1125`
- tick `75892`, seconds `36.00`, LSTM `0.7699`, delta `+0.1035`
- tick `78772`, seconds `81.00`, LSTM `0.8437`, delta `-0.0990`
- tick `75412`, seconds `28.50`, LSTM `0.6737`, delta `+0.0922`
- tick `78612`, seconds `78.50`, LSTM `0.7868`, delta `+0.0876`
- tick `75700`, seconds `33.00`, LSTM `0.7163`, delta `-0.0805`

## Top 15 local ridge features

- `lag_13__T_place_OBSERVATION`: coefficient `0.002480`, |coef| `0.002480`
- `lag_02__CT_place_DECON`: coefficient `0.001996`, |coef| `0.001996`
- `lag_00__kill_diff_last_3s`: coefficient `0.001910`, |coef| `0.001910`
- `lag_08__CT_place_DECON`: coefficient `0.001748`, |coef| `0.001748`
- `lag_09__T_place_OBSERVATION`: coefficient `-0.001584`, |coef| `0.001584`
- `lag_00__CT_kills_last_3s`: coefficient `0.001457`, |coef| `0.001457`
- `lag_00__damage_diff_last_5s`: coefficient `0.001355`, |coef| `0.001355`
- `lag_00__CT_damage_last_5s`: coefficient `0.001342`, |coef| `0.001342`
- `lag_01__CT_place_VENTS`: coefficient `-0.001329`, |coef| `0.001329`
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001220`, |coef| `0.001220`
- `lag_06__CT_place_VENTS`: coefficient `-0.001179`, |coef| `0.001179`
- `lag_07__T_place_CONTROL`: coefficient `0.001155`, |coef| `0.001155`
- `lag_08__T_place_VENTS`: coefficient `0.001116`, |coef| `0.001116`
- `lag_03__CT_place_VENTS`: coefficient `0.001098`, |coef| `0.001098`
- `lag_12__T_place_OBSERVATION`: coefficient `0.001041`, |coef| `0.001041`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.000975` (lowers CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.000778` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000437` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000436` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000416` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000368` (raises CT win probability)
- `lag_03__CT1__molly`: coefficient `0.000359` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.000353` (raises CT win probability)
- `lag_08__CT5__molly`: coefficient `0.000348` (raises CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.000340` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_OBSERVATION`: coefficient `0.002480` (raises CT win probability)
- `lag_02__CT_place_DECON`: coefficient `0.001996` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001910` (raises CT win probability)
- `lag_08__CT_place_DECON`: coefficient `0.001748` (raises CT win probability)
- `lag_09__T_place_OBSERVATION`: coefficient `-0.001584` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001457` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001355` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001342` (raises CT win probability)
- `lag_01__CT_place_VENTS`: coefficient `-0.001329` (lowers CT win probability)
- `lag_00__T_place_OBSERVATION`: coefficient `-0.001220` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `75796`, seconds `34.50`, LSTM delta `-0.1525`

Top all feature movements:
- `lag_03__CT_place_VENTS`: contribution `-0.009210`
- `lag_05__CT_place_VENTS`: contribution `-0.008104`
- `lag_06__T_place_TROPHY`: contribution `-0.006178`
- `lag_02__T_place_CONTROL`: contribution `-0.005574`
- `lag_10__T_place_TROPHY`: contribution `-0.004728`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75732`, seconds `33.50`, LSTM delta `+0.1525`

Top all feature movements:
- `lag_01__CT_place_VENTS`: contribution `+0.011155`
- `lag_03__CT_place_VENTS`: contribution `+0.009210`
- `lag_00__T_place_CONTROL`: contribution `+0.006682`
- `lag_02__T_place_CONTROL`: contribution `+0.005574`
- `lag_00__kill_diff_last_3s`: contribution `+0.004596`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `78644`, seconds `79.00`, LSTM delta `+0.1346`

Top all feature movements:
- `lag_13__T_place_OBSERVATION`: contribution `+0.041991`
- `lag_09__T_place_OBSERVATION`: contribution `+0.026818`
- `lag_03__CT_place_DECON`: contribution `+0.005544`
- `lag_00__kill_diff_last_3s`: contribution `+0.004596`
- `lag_00__CT_kills_last_3s`: contribution `+0.004206`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.001180`

### tick `78804`, seconds `81.50`, LSTM delta `+0.1221`

Top all feature movements:
- `lag_08__CT_place_DECON`: contribution `+0.027797`
- `lag_14__T_place_OBSERVATION`: contribution `+0.013714`
- `lag_00__T_flash_alpha_mean`: contribution `+0.005914`
- `lag_00__T_shots_fired_sum`: contribution `+0.005776`
- `lag_03__CT_place_DECON`: contribution `-0.005544`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.005914`

### tick `75476`, seconds `29.50`, LSTM delta `+0.1125`

Top all feature movements:
- `lag_08__T_place_VENTS`: contribution `+0.015052`
- `lag_07__T_place_CONTROL`: contribution `+0.008209`
- `lag_06__T_place_TROPHY`: contribution `+0.006178`
- `lag_02__T_place_CONTROL`: contribution `-0.005574`
- `lag_07__T_place_TROPHY`: contribution `+0.004810`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `+0.004241`
