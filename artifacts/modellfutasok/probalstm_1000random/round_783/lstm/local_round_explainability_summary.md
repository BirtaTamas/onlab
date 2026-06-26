# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `17`

## Largest probability jumps

- tick `124839`, seconds `32.50`, LSTM `0.8432`, delta `+0.1866`
- tick `124551`, seconds `28.00`, LSTM `0.6866`, delta `+0.1549`
- tick `125575`, seconds `44.00`, LSTM `0.8021`, delta `+0.0534`
- tick `125447`, seconds `42.00`, LSTM `0.7443`, delta `-0.0518`
- tick `123495`, seconds `11.50`, LSTM `0.4534`, delta `+0.0499`
- tick `125191`, seconds `38.00`, LSTM `0.7582`, delta `-0.0461`
- tick `125031`, seconds `35.50`, LSTM `0.7782`, delta `-0.0422`
- tick `123559`, seconds `12.50`, LSTM `0.5265`, delta `+0.0408`
- tick `124071`, seconds `20.50`, LSTM `0.5147`, delta `+0.0377`
- tick `123623`, seconds `13.50`, LSTM `0.5017`, delta `-0.0372`

## Top 15 local ridge features

- `lag_15__CT_place_LIBRARY`: coefficient `0.002359`, |coef| `0.002359`
- `lag_12__CT5__flash_duration`: coefficient `0.002093`, |coef| `0.002093`
- `lag_00__CT2__duck_amount`: coefficient `0.001655`, |coef| `0.001655`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001598`, |coef| `0.001598`
- `lag_06__CT_place_LIBRARY`: coefficient `0.001545`, |coef| `0.001545`
- `lag_03__CT5__flash_duration`: coefficient `0.001525`, |coef| `0.001525`
- `lag_00__CT_kills_last_3s`: coefficient `0.001487`, |coef| `0.001487`
- `lag_00__damage_diff_last_5s`: coefficient `0.001285`, |coef| `0.001285`
- `lag_00__kill_diff_last_3s`: coefficient `0.001240`, |coef| `0.001240`
- `lag_01__CT_place_RUINS`: coefficient `0.001206`, |coef| `0.001206`
- `lag_00__CT2__shots_fired`: coefficient `0.001200`, |coef| `0.001200`
- `lag_14__CT_place_RUINS`: coefficient `0.001155`, |coef| `0.001155`
- `lag_05__CT_place_LIBRARY`: coefficient `0.001129`, |coef| `0.001129`
- `lag_07__CT_burning_players`: coefficient `0.001127`, |coef| `0.001127`
- `lag_00__CT_damage_last_5s`: coefficient `0.001073`, |coef| `0.001073`

## Top 10 utility ridge features

- `lag_12__CT5__flash_duration`: coefficient `0.002093` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.001525` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.001024` (raises CT win probability)
- `lag_10__CT2__flash_duration`: coefficient `0.000881` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `0.000875` (raises CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `-0.000823` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000817` (lowers CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.000797` (raises CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `0.000780` (raises CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.000767` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_LIBRARY`: coefficient `0.002359` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.001655` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001598` (raises CT win probability)
- `lag_06__CT_place_LIBRARY`: coefficient `0.001545` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001487` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001285` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001240` (raises CT win probability)
- `lag_01__CT_place_RUINS`: coefficient `0.001206` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.001200` (raises CT win probability)
- `lag_14__CT_place_RUINS`: coefficient `0.001155` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `124839`, seconds `32.50`, LSTM delta `+0.1866`

Top all feature movements:
- `lag_12__CT5__flash_duration`: contribution `+0.015611`
- `lag_15__CT_place_LIBRARY`: contribution `+0.015122`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006661`
- `lag_00__CT_kills_last_3s`: contribution `+0.004294`
- `lag_09__T_flashed_players`: contribution `+0.003746`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.015611`

### tick `124551`, seconds `28.00`, LSTM delta `+0.1549`

Top all feature movements:
- `lag_03__CT5__flash_duration`: contribution `+0.011373`
- `lag_06__CT_place_LIBRARY`: contribution `+0.009904`
- `lag_00__CT2__duck_amount`: contribution `+0.006103`
- `lag_10__T1__flash_duration`: contribution `+0.005609`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005551`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `+0.011373`
- `lag_10__T1__flash_duration`: contribution `+0.005609`
- `lag_10__CT2__flash_duration`: contribution `+0.004913`
- `lag_10__T2__flash_duration`: contribution `+0.004881`
- `lag_10__T_flash_duration_sum`: contribution `+0.003559`

### tick `125575`, seconds `44.00`, LSTM delta `+0.0534`

Top all feature movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.005054`
- `lag_01__CT_place_RUINS`: contribution `+0.004212`
- `lag_10__T_shots_fired_sum`: contribution `+0.003362`
- `lag_00__T1__is_scoped`: contribution `+0.003319`
- `lag_10__CT_shots_fired_sum`: contribution `+0.003177`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.005054`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.001488`
- `lag_08__T_utility_damage_last_5s`: contribution `+0.001351`

### tick `125447`, seconds `42.00`, LSTM delta `-0.0518`

Top all feature movements:
- `lag_01__CT_place_RUINS`: contribution `-0.004212`
- `lag_00__CT_place_BALCONY`: contribution `-0.003954`
- `lag_15__T_utility_damage_last_5s`: contribution `-0.003943`
- `lag_06__T_shots_fired_sum`: contribution `-0.003758`
- `lag_12__CT_place_BALCONY`: contribution `+0.003501`

Top utility-only movements:
- `lag_15__T_utility_damage_last_5s`: contribution `-0.003943`
- `lag_05__T_utility_damage_last_5s`: contribution `-0.002705`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.001290`

### tick `123495`, seconds `11.50`, LSTM delta `+0.0499`

Top all feature movements:
- `lag_14__CT_place_RUINS`: contribution `+0.004036`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.004017`
- `lag_14__T_place_LOWERMID`: contribution `+0.003786`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003290`
- `lag_12__CT_place_LIBRARY`: contribution `-0.003272`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.004017`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003290`
