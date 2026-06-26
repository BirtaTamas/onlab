# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `18`

## Largest probability jumps

- tick `125848`, seconds `20.00`, LSTM `0.1026`, delta `-0.3708`
- tick `125432`, seconds `13.50`, LSTM `0.4386`, delta `-0.1708`
- tick `125592`, seconds `16.00`, LSTM `0.6569`, delta `+0.1618`
- tick `125688`, seconds `17.50`, LSTM `0.4325`, delta `-0.1216`
- tick `125976`, seconds `22.00`, LSTM `0.0180`, delta `-0.0767`
- tick `125656`, seconds `17.00`, LSTM `0.5541`, delta `-0.0741`
- tick `125944`, seconds `21.50`, LSTM `0.0947`, delta `-0.0738`
- tick `125912`, seconds `21.00`, LSTM `0.1685`, delta `+0.0573`
- tick `125304`, seconds `11.50`, LSTM `0.6078`, delta `+0.0567`
- tick `125464`, seconds `14.00`, LSTM `0.4873`, delta `+0.0487`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002531`, |coef| `0.002531`
- `lag_00__CT1__shots_fired`: coefficient `0.001701`, |coef| `0.001701`
- `lag_01__CT2__flash_duration`: coefficient `-0.001599`, |coef| `0.001599`
- `lag_12__CT2__flash_duration`: coefficient `-0.001559`, |coef| `0.001559`
- `lag_14__CT2__flash_duration`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_09__CT2__flash_duration`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_05__CT2__flash_duration`: coefficient `-0.001303`, |coef| `0.001303`
- `lag_15__CT2__flash_duration`: coefficient `-0.001291`, |coef| `0.001291`
- `lag_01__CT_place_TSIDEUPPER`: coefficient `0.001285`, |coef| `0.001285`
- `lag_06__CT5__flash_duration`: coefficient `0.001220`, |coef| `0.001220`
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.001170`, |coef| `0.001170`
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.001161`, |coef| `0.001161`
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001149`, |coef| `0.001149`
- `lag_00__kill_diff_last_3s`: coefficient `0.001129`, |coef| `0.001129`
- `lag_00__T_kills_last_3s`: coefficient `-0.001114`, |coef| `0.001114`

## Top 10 utility ridge features

- `lag_01__CT2__flash_duration`: coefficient `-0.001599` (lowers CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.001559` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `-0.001436` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.001378` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001303` (lowers CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `-0.001291` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001220` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.001170` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.001161` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `-0.000939` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002531` (raises CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `0.001701` (raises CT win probability)
- `lag_01__CT_place_TSIDEUPPER`: coefficient `0.001285` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `-0.001149` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001129` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001114` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001098` (lowers CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.001030` (raises CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `0.001029` (raises CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `0.001025` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `125848`, seconds `20.00`, LSTM delta `-0.3708`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.038689`
- `lag_00__CT1__shots_fired`: contribution `-0.019771`
- `lag_12__T_shots_fired_sum`: contribution `-0.011524`
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.010988`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.010462`

Top utility-only movements:
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.010988`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.010462`
- `lag_12__CT2__flash_duration`: contribution `-0.009894`
- `lag_14__CT2__flash_duration`: contribution `-0.008895`
- `lag_06__CT5__flash_duration`: contribution `-0.005595`

### tick `125432`, seconds `13.50`, LSTM delta `-0.1708`

Top all feature movements:
- `lag_01__CT2__flash_duration`: contribution `-0.009902`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.008420`
- `lag_03__T_flashed_players`: contribution `-0.005250`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.005058`
- `lag_02__T_shots_fired_sum`: contribution `-0.004115`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.009902`
- `lag_04__CT_utility_damage_last_5s`: contribution `-0.008420`
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.005058`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.003225`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002681`

### tick `125592`, seconds `16.00`, LSTM delta `+0.1618`

Top all feature movements:
- `lag_01__CT_place_TSIDEUPPER`: contribution `+0.009660`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.008795`
- `lag_05__CT2__flash_duration`: contribution `+0.008070`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.007287`
- `lag_06__CT5__flash_duration`: contribution `+0.005595`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.008795`
- `lag_05__CT2__flash_duration`: contribution `+0.008070`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.007287`
- `lag_06__CT5__flash_duration`: contribution `+0.005595`
- `lag_06__CT_flash_duration_sum`: contribution `+0.003899`

### tick `125688`, seconds `17.50`, LSTM delta `-0.1216`

Top all feature movements:
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.009660`
- `lag_09__CT2__flash_duration`: contribution `-0.008532`
- `lag_07__T_shots_fired_sum`: contribution `-0.007046`
- `lag_08__CT2__flash_duration`: contribution `+0.004808`
- `lag_07__CT2__flash_duration`: contribution `-0.004418`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.008532`
- `lag_08__CT2__flash_duration`: contribution `+0.004808`
- `lag_07__CT2__flash_duration`: contribution `-0.004418`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.003779`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003007`

### tick `125976`, seconds `22.00`, LSTM delta `-0.0767`

Top all feature movements:
- `lag_04__T5__shots_fired`: contribution `-0.006942`
- `lag_00__T_kills_last_3s`: contribution `-0.003530`
- `lag_02__CT_shots_fired_sum`: contribution `-0.003194`
- `lag_00__kill_diff_last_3s`: contribution `-0.002717`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.002591`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `+0.002545`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.002080`
- `lag_09__utility_damage_diff_last_5s`: contribution `-0.002033`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.001695`
