# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `3`

## Largest probability jumps

- tick `13013`, seconds `26.50`, LSTM `0.9032`, delta `+0.0933`
- tick `17749`, seconds `100.50`, LSTM `0.9590`, delta `+0.0483`
- tick `16981`, seconds `88.50`, LSTM `0.9055`, delta `-0.0416`
- tick `11989`, seconds `10.50`, LSTM `0.8486`, delta `-0.0345`
- tick `12149`, seconds `13.00`, LSTM `0.8638`, delta `+0.0276`
- tick `17717`, seconds `100.00`, LSTM `0.9107`, delta `+0.0272`
- tick `14101`, seconds `43.50`, LSTM `0.9552`, delta `+0.0248`
- tick `12981`, seconds `26.00`, LSTM `0.8098`, delta `+0.0233`
- tick `12917`, seconds `25.00`, LSTM `0.8065`, delta `-0.0227`
- tick `13589`, seconds `35.50`, LSTM `0.9014`, delta `-0.0219`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000827`, |coef| `0.000827`
- `lag_06__T_place_TROPHY`: coefficient `0.000803`, |coef| `0.000803`
- `lag_00__CT_kills_last_3s`: coefficient `0.000762`, |coef| `0.000762`
- `lag_03__T_place_DECON`: coefficient `0.000723`, |coef| `0.000723`
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.000653`, |coef| `0.000653`
- `lag_00__damage_diff_last_5s`: coefficient `0.000648`, |coef| `0.000648`
- `lag_01__T_place_DECON`: coefficient `-0.000624`, |coef| `0.000624`
- `lag_00__T_place_DECON`: coefficient `-0.000616`, |coef| `0.000616`
- `lag_11__CT_place_HELL`: coefficient `-0.000604`, |coef| `0.000604`
- `lag_14__CT_place_CONTROL`: coefficient `0.000589`, |coef| `0.000589`
- `lag_11__CT2__duck_amount`: coefficient `-0.000549`, |coef| `0.000549`
- `lag_06__T_place_VENDING`: coefficient `-0.000545`, |coef| `0.000545`
- `lag_00__CT_damage_last_5s`: coefficient `0.000497`, |coef| `0.000497`
- `lag_03__CT_place_ADMIN`: coefficient `-0.000495`, |coef| `0.000495`
- `lag_07__CT5__duck_amount`: coefficient `0.000493`, |coef| `0.000493`

## Top 10 utility ridge features

- `lag_01__T_smokes_last_5s`: coefficient `-0.000466` (lowers CT win probability)
- `lag_10__CT4__smoke`: coefficient `-0.000409` (lowers CT win probability)
- `lag_10__T_smokes_last_5s`: coefficient `0.000320` (raises CT win probability)
- `lag_02__T_smokes_last_5s`: coefficient `-0.000301` (lowers CT win probability)
- `lag_14__T_smokes_last_5s`: coefficient `0.000301` (raises CT win probability)
- `lag_06__CT_B_site_active_smokes`: coefficient `0.000297` (raises CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000294` (lowers CT win probability)
- `lag_06__CT_A_site_active_smokes`: coefficient `0.000289` (raises CT win probability)
- `lag_13__T_smokes_last_5s`: coefficient `0.000286` (raises CT win probability)
- `lag_05__T_smokes_last_5s`: coefficient `-0.000272` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.000827` (raises CT win probability)
- `lag_06__T_place_TROPHY`: coefficient `0.000803` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000762` (raises CT win probability)
- `lag_03__T_place_DECON`: coefficient `0.000723` (raises CT win probability)
- `lag_00__CT_place_LOCKERROOM`: coefficient `-0.000653` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000648` (raises CT win probability)
- `lag_01__T_place_DECON`: coefficient `-0.000624` (lowers CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.000616` (lowers CT win probability)
- `lag_11__CT_place_HELL`: coefficient `-0.000604` (lowers CT win probability)
- `lag_14__CT_place_CONTROL`: coefficient `0.000589` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `13013`, seconds `26.50`, LSTM delta `+0.0933`

Top all feature movements:
- `lag_06__T_place_TROPHY`: contribution `+0.005090`
- `lag_03__CT_place_ADMIN`: contribution `+0.003440`
- `lag_06__T_place_VENDING`: contribution `+0.002765`
- `lag_14__CT_place_HEAVEN`: contribution `+0.002537`
- `lag_14__CT_place_HELL`: contribution `+0.002386`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17749`, seconds `100.50`, LSTM delta `+0.0483`

Top all feature movements:
- `lag_03__T_place_DECON`: contribution `+0.011611`
- `lag_01__T_place_DECON`: contribution `+0.010027`
- `lag_06__CT_place_LOCKERROOM`: contribution `+0.004002`
- `lag_03__CT_place_ADMIN`: contribution `-0.003440`
- `lag_09__CT_place_LOCKERROOM`: contribution `+0.003233`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `16981`, seconds `88.50`, LSTM delta `-0.0416`

Top all feature movements:
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.008128`
- `lag_14__CT_place_CONTROL`: contribution `-0.006111`
- `lag_08__CT_place_MINI`: contribution `-0.002792`
- `lag_15__T_place_SECRET`: contribution `-0.002012`
- `lag_00__kill_diff_last_3s`: contribution `-0.001992`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `11989`, seconds `10.50`, LSTM delta `-0.0345`

Top all feature movements:
- `lag_11__CT_place_HELL`: contribution `-0.006551`
- `lag_10__T_smokes_last_5s`: contribution `-0.004689`
- `lag_00__CT_place_HEAVEN`: contribution `+0.002280`
- `lag_10__CT_place_HELL`: contribution `-0.002186`
- `lag_08__CT_place_HELL`: contribution `-0.002045`

Top utility-only movements:
- `lag_10__T_smokes_last_5s`: contribution `-0.004689`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.000655`
- `lag_06__CT_A_site_active_infernos`: contribution `-0.000485`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.000475`

### tick `12149`, seconds `13.00`, LSTM delta `+0.0276`

Top all feature movements:
- `lag_10__CT_place_HELL`: contribution `+0.002186`
- `lag_10__CT_place_RAFTERS`: contribution `+0.001652`
- `lag_06__CT_place_RAFTERS`: contribution `+0.001615`
- `lag_15__CT_place_HELL`: contribution `+0.001600`
- `lag_13__CT_place_HELL`: contribution `-0.001444`

Top utility-only movements:
- `lag_15__T_smokes_last_5s`: contribution `+0.000659`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.000655`
