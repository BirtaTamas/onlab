# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `35214`, seconds `107.50`, LSTM `0.5778`, delta `+0.4227`
- tick `35502`, seconds `112.00`, LSTM `0.8799`, delta `+0.2388`
- tick `34670`, seconds `99.00`, LSTM `0.4312`, delta `-0.1972`
- tick `29166`, seconds `13.00`, LSTM `0.7560`, delta `+0.1502`
- tick `29998`, seconds `26.00`, LSTM `0.7986`, delta `-0.1199`
- tick `29230`, seconds `14.00`, LSTM `0.8975`, delta `+0.1158`
- tick `34990`, seconds `104.00`, LSTM `0.3091`, delta `+0.1099`
- tick `34894`, seconds `102.50`, LSTM `0.2452`, delta `-0.0859`
- tick `33934`, seconds `87.50`, LSTM `0.6575`, delta `-0.0816`
- tick `35022`, seconds `104.50`, LSTM `0.2410`, delta `-0.0681`

## Top 15 local ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.006010`, |coef| `0.006010`
- `lag_02__T_place_OBSERVATION`: coefficient `0.005475`, |coef| `0.005475`
- `lag_00__damage_diff_last_5s`: coefficient `0.003281`, |coef| `0.003281`
- `lag_00__kill_diff_last_3s`: coefficient `0.003275`, |coef| `0.003275`
- `lag_00__T_place_DECON`: coefficient `-0.003260`, |coef| `0.003260`
- `lag_09__T1__duck_amount`: coefficient `-0.003179`, |coef| `0.003179`
- `lag_10__CT4__flash_duration`: coefficient `0.003153`, |coef| `0.003153`
- `lag_01__CT4__flash_duration`: coefficient `-0.002464`, |coef| `0.002464`
- `lag_04__T_place_OBSERVATION`: coefficient `0.002328`, |coef| `0.002328`
- `lag_00__CT_kills_last_3s`: coefficient `0.002275`, |coef| `0.002275`
- `lag_09__T_place_OBSERVATION`: coefficient `-0.002264`, |coef| `0.002264`
- `lag_00__CT4__flash_duration`: coefficient `-0.002247`, |coef| `0.002247`
- `lag_08__T1__is_walking`: coefficient `0.002245`, |coef| `0.002245`
- `lag_11__T_place_OBSERVATION`: coefficient `0.002101`, |coef| `0.002101`
- `lag_09__T_duck_amount_mean`: coefficient `-0.001974`, |coef| `0.001974`

## Top 10 utility ridge features

- `lag_10__CT4__flash_duration`: coefficient `0.003153` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.002464` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.002247` (lowers CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `-0.001860` (lowers CT win probability)
- `lag_10__CT_A_site_active_infernos`: coefficient `-0.001833` (lowers CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `-0.001753` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.001666` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001632` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `0.001577` (raises CT win probability)
- `lag_07__CT5__molly`: coefficient `0.001544` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_OBSERVATION`: coefficient `-0.006010` (lowers CT win probability)
- `lag_02__T_place_OBSERVATION`: coefficient `0.005475` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003281` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003275` (raises CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.003260` (lowers CT win probability)
- `lag_09__T1__duck_amount`: coefficient `-0.003179` (lowers CT win probability)
- `lag_04__T_place_OBSERVATION`: coefficient `0.002328` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002275` (raises CT win probability)
- `lag_09__T_place_OBSERVATION`: coefficient `-0.002264` (lowers CT win probability)
- `lag_08__T1__is_walking`: coefficient `0.002245` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `35214`, seconds `107.50`, LSTM delta `+0.4227`

Top all feature movements:
- `lag_00__T_place_OBSERVATION`: contribution `+0.101768`
- `lag_02__T_place_OBSERVATION`: contribution `+0.092707`
- `lag_06__T_place_DECON`: contribution `+0.020122`
- `lag_10__CT4__flash_duration`: contribution `+0.019630`
- `lag_07__T_place_DECON`: contribution `+0.018701`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.019630`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.006468`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.006390`
- `lag_10__CT_flash_duration_sum`: contribution `+0.004392`

### tick `35502`, seconds `112.00`, LSTM delta `+0.2388`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `+0.052381`
- `lag_09__T_place_OBSERVATION`: contribution `+0.038336`
- `lag_11__T_place_OBSERVATION`: contribution `+0.035575`
- `lag_15__T_place_DECON`: contribution `+0.010939`
- `lag_00__T_flash_alpha_mean`: contribution `+0.009903`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.009903`

### tick `34670`, seconds `99.00`, LSTM delta `-0.1972`

Top all feature movements:
- `lag_02__T_place_DECON`: contribution `-0.030089`
- `lag_09__T1__duck_amount`: contribution `-0.010796`
- `lag_00__kill_diff_last_3s`: contribution `-0.007882`
- `lag_12__T2__duck_amount`: contribution `-0.007043`
- `lag_00__T_shots_fired_sum`: contribution `-0.006319`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.006023`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.005879`
- `lag_07__CT5__molly`: contribution `-0.003831`

### tick `29166`, seconds `13.00`, LSTM delta `+0.1502`

Top all feature movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.008184`
- `lag_00__kill_diff_last_3s`: contribution `+0.007882`
- `lag_00__damage_diff_last_5s`: contribution `+0.007107`
- `lag_00__CT_kills_last_3s`: contribution `+0.006568`
- `lag_07__T_place_SQUEAKY`: contribution `+0.005780`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.008184`
- `lag_06__CT2__flash_duration`: contribution `+0.003651`
- `lag_05__CT_active_infernos`: contribution `+0.003529`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.002798`
- `lag_08__T1__flash_duration`: contribution `+0.002512`

### tick `29998`, seconds `26.00`, LSTM delta `-0.1199`

Top all feature movements:
- `lag_09__T1__duck_amount`: contribution `-0.012446`
- `lag_00__kill_diff_last_3s`: contribution `-0.007882`
- `lag_00__T_shots_fired_sum`: contribution `-0.006319`
- `lag_00__T_kills_last_3s`: contribution `-0.005747`
- `lag_09__T_duck_amount_mean`: contribution `-0.005740`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.004967`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.002804`
