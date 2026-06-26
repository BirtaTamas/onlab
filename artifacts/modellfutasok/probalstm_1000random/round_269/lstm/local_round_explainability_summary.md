# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `73350`, seconds `68.50`, LSTM `0.6123`, delta `+0.2609`
- tick `71398`, seconds `38.00`, LSTM `0.5241`, delta `-0.2102`
- tick `74086`, seconds `80.00`, LSTM `0.9312`, delta `+0.1885`
- tick `70950`, seconds `31.00`, LSTM `0.4589`, delta `-0.1328`
- tick `71334`, seconds `37.00`, LSTM `0.7285`, delta `+0.1327`
- tick `71782`, seconds `44.00`, LSTM `0.4117`, delta `-0.0946`
- tick `73318`, seconds `68.00`, LSTM `0.3514`, delta `+0.0849`
- tick `72262`, seconds `51.50`, LSTM `0.3185`, delta `-0.0652`
- tick `73606`, seconds `72.50`, LSTM `0.7574`, delta `+0.0521`
- tick `71878`, seconds `45.50`, LSTM `0.2904`, delta `-0.0515`

## Top 15 local ridge features

- `lag_00__T_place_LOCKERROOM`: coefficient `-0.006064`, |coef| `0.006064`
- `lag_00__CT_place_DECON`: coefficient `-0.003768`, |coef| `0.003768`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002225`, |coef| `0.002225`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002171`, |coef| `0.002171`
- `lag_00__kill_diff_last_3s`: coefficient `0.002130`, |coef| `0.002130`
- `lag_00__damage_diff_last_5s`: coefficient `0.002023`, |coef| `0.002023`
- `lag_00__CT_kills_last_3s`: coefficient `0.001946`, |coef| `0.001946`
- `lag_08__T4__duck_amount`: coefficient `0.001808`, |coef| `0.001808`
- `lag_00__CT_damage_last_5s`: coefficient `0.001681`, |coef| `0.001681`
- `lag_08__CT2__is_walking`: coefficient `0.001638`, |coef| `0.001638`
- `lag_00__T2__duck_amount`: coefficient `0.001626`, |coef| `0.001626`
- `lag_09__CT4__duck_amount`: coefficient `0.001621`, |coef| `0.001621`
- `lag_12__CT_place_ADMIN`: coefficient `-0.001482`, |coef| `0.001482`
- `lag_01__CT1__shots_fired`: coefficient `0.001419`, |coef| `0.001419`
- `lag_07__CT_place_SQUEAKY`: coefficient `-0.001382`, |coef| `0.001382`

## Top 10 utility ridge features

- `lag_04__CT_B_site_active_infernos`: coefficient `-0.000969` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `-0.000934` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000823` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000806` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000795` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `-0.000782` (lowers CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.000697` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `-0.000614` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000569` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `-0.000565` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_LOCKERROOM`: coefficient `-0.006064` (lowers CT win probability)
- `lag_00__CT_place_DECON`: coefficient `-0.003768` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002225` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002171` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002130` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002023` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001946` (raises CT win probability)
- `lag_08__T4__duck_amount`: coefficient `0.001808` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001681` (raises CT win probability)
- `lag_08__CT2__is_walking`: coefficient `0.001638` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `73350`, seconds `68.50`, LSTM delta `+0.2609`

Top all feature movements:
- `lag_00__T_place_LOCKERROOM`: contribution `+0.224238`
- `lag_00__CT_shots_fired_sum`: contribution `-0.009277`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007541`
- `lag_00__CT_kills_last_3s`: contribution `+0.005618`
- `lag_00__kill_diff_last_3s`: contribution `+0.005127`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71398`, seconds `38.00`, LSTM delta `-0.2102`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.009050`
- `lag_11__T_place_CONTROL`: contribution `-0.008417`
- `lag_10__CT_place_ADMIN`: contribution `-0.008226`
- `lag_11__CT_shots_fired_sum`: contribution `-0.007515`
- `lag_12__T_place_CONTROL`: contribution `-0.006174`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.005414`

### tick `74086`, seconds `80.00`, LSTM delta `+0.1885`

Top all feature movements:
- `lag_00__CT_place_DECON`: contribution `+0.059912`
- `lag_12__CT_place_ADMIN`: contribution `+0.010295`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007730`
- `lag_01__CT_shots_fired_sum`: contribution `+0.006033`
- `lag_00__CT_kills_last_3s`: contribution `+0.005618`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `+0.003331`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.003298`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.002845`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.002828`

### tick `70950`, seconds `31.00`, LSTM delta `-0.1328`

Top all feature movements:
- `lag_07__CT_flashed_players`: contribution `-0.007689`
- `lag_06__T_place_HUT`: contribution `-0.007579`
- `lag_03__T_place_HUT`: contribution `-0.007363`
- `lag_03__T_place_TROPHY`: contribution `-0.005954`
- `lag_00__kill_diff_last_3s`: contribution `-0.005127`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.005075`
- `lag_07__CT_flash_duration_sum`: contribution `-0.004593`
- `lag_00__CT3__flash_duration`: contribution `-0.003878`
- `lag_07__CT3__flash_duration`: contribution `-0.002642`

### tick `71334`, seconds `37.00`, LSTM delta `+0.1327`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007730`
- `lag_03__T_place_TROPHY`: contribution `+0.005954`
- `lag_00__CT_kills_last_3s`: contribution `+0.005618`
- `lag_11__CT_shots_fired_sum`: contribution `+0.005466`
- `lag_02__CT_place_MINI`: contribution `+0.005417`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `+0.003787`
