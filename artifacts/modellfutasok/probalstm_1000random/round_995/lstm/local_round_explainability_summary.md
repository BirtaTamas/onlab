# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `44254`, seconds `70.50`, LSTM `0.8499`, delta `+0.1251`
- tick `44478`, seconds `74.00`, LSTM `0.8865`, delta `+0.1100`
- tick `44510`, seconds `74.50`, LSTM `0.9518`, delta `+0.0653`
- tick `44318`, seconds `71.50`, LSTM `0.8136`, delta `-0.0538`
- tick `44446`, seconds `73.50`, LSTM `0.7765`, delta `-0.0408`
- tick `44126`, seconds `68.50`, LSTM `0.7167`, delta `+0.0327`
- tick `42686`, seconds `46.00`, LSTM `0.6830`, delta `-0.0266`
- tick `39774`, seconds `0.50`, LSTM `0.7833`, delta `+0.0262`
- tick `40478`, seconds `11.50`, LSTM `0.7251`, delta `+0.0260`
- tick `41726`, seconds `31.00`, LSTM `0.7046`, delta `-0.0249`

## Top 15 local ridge features

- `lag_03__T_place_QUAD`: coefficient `0.001552`, |coef| `0.001552`
- `lag_04__T_flashed_players`: coefficient `0.000981`, |coef| `0.000981`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000917`, |coef| `0.000917`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000876`, |coef| `0.000876`
- `lag_00__T_place_QUAD`: coefficient `-0.000846`, |coef| `0.000846`
- `lag_01__T_place_QUAD`: coefficient `0.000798`, |coef| `0.000798`
- `lag_10__T4__is_walking`: coefficient `-0.000793`, |coef| `0.000793`
- `lag_00__CT2__flash_duration`: coefficient `0.000790`, |coef| `0.000790`
- `lag_10__T_place_QUAD`: coefficient `0.000778`, |coef| `0.000778`
- `lag_00__T_place_LIBRARY`: coefficient `0.000707`, |coef| `0.000707`
- `lag_00__kill_diff_last_3s`: coefficient `0.000705`, |coef| `0.000705`
- `lag_04__T_place_QUAD`: coefficient `0.000700`, |coef| `0.000700`
- `lag_00__CT_kills_last_3s`: coefficient `0.000683`, |coef| `0.000683`
- `lag_06__T_flashed_players`: coefficient `-0.000648`, |coef| `0.000648`
- `lag_02__CT_place_RUINS`: coefficient `0.000617`, |coef| `0.000617`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.000790` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.000568` (raises CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `0.000489` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000456` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.000453` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000443` (raises CT win probability)
- `lag_07__T5__flash`: coefficient `-0.000431` (lowers CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.000413` (raises CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.000411` (raises CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.000403` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_QUAD`: coefficient `0.001552` (raises CT win probability)
- `lag_04__T_flashed_players`: coefficient `0.000981` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000917` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000876` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `-0.000846` (lowers CT win probability)
- `lag_01__T_place_QUAD`: coefficient `0.000798` (raises CT win probability)
- `lag_10__T4__is_walking`: coefficient `-0.000793` (lowers CT win probability)
- `lag_10__T_place_QUAD`: coefficient `0.000778` (raises CT win probability)
- `lag_00__T_place_LIBRARY`: coefficient `0.000707` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000705` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `44254`, seconds `70.50`, LSTM delta `+0.1251`

Top all feature movements:
- `lag_03__T_place_QUAD`: contribution `+0.037380`
- `lag_00__T_place_QUAD`: contribution `+0.020385`
- `lag_04__T_flashed_players`: contribution `+0.005681`
- `lag_04__CT2__flash_duration`: contribution `+0.002749`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002436`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.002749`
- `lag_04__T_flash_duration_sum`: contribution `+0.001296`
- `lag_04__T2__flash_duration`: contribution `+0.001145`
- `lag_04__T3__flash_duration`: contribution `+0.001099`

### tick `44478`, seconds `74.00`, LSTM delta `+0.1100`

Top all feature movements:
- `lag_01__T_place_QUAD`: contribution `+0.019213`
- `lag_10__T_place_QUAD`: contribution `+0.018738`
- `lag_07__T_place_QUAD`: contribution `+0.009791`
- `lag_06__T_utility_damage_last_5s`: contribution `+0.003843`
- `lag_06__T_flashed_players`: contribution `+0.002500`

Top utility-only movements:
- `lag_06__T_utility_damage_last_5s`: contribution `+0.003843`
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.001221`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.001127`
- `lag_11__CT2__flash_duration`: contribution `+0.001065`

### tick `44510`, seconds `74.50`, LSTM delta `+0.0653`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `+0.020385`
- `lag_02__T_place_QUAD`: contribution `+0.008984`
- `lag_11__T_place_QUAD`: contribution `+0.004366`
- `lag_00__kill_diff_last_3s`: contribution `+0.003394`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003045`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `+0.002710`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.001130`
- `lag_06__CT2__flash_duration`: contribution `+0.000993`

### tick `44318`, seconds `71.50`, LSTM delta `-0.0538`

Top all feature movements:
- `lag_02__T_place_QUAD`: contribution `-0.008984`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005480`
- `lag_05__T_place_QUAD`: contribution `-0.004330`
- `lag_00__CT2__flash_duration`: contribution `-0.003823`
- `lag_06__T_flashed_players`: contribution `-0.003749`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.003823`
- `lag_01__T2__flash_duration`: contribution `-0.001261`
- `lag_01__T_utility_damage_last_5s`: contribution `-0.001223`
- `lag_01__T3__flash_duration`: contribution `-0.001209`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.001111`

### tick `44446`, seconds `73.50`, LSTM delta `-0.0408`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `-0.020385`
- `lag_06__T_place_QUAD`: contribution `-0.014805`
- `lag_04__CT2__flash_duration`: contribution `-0.002749`
- `lag_05__T_flashed_players`: contribution `-0.002082`
- `lag_00__CT_kills_last_3s`: contribution `-0.001972`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `-0.002749`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.001917`
- `lag_10__CT2__flash_duration`: contribution `+0.000738`
