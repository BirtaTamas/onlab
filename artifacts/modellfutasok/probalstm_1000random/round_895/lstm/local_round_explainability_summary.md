# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `33`

## Largest probability jumps

- tick `287701`, seconds `86.50`, LSTM `0.2016`, delta `-0.2583`
- tick `283157`, seconds `15.50`, LSTM `0.4681`, delta `-0.1508`
- tick `283189`, seconds `16.00`, LSTM `0.3757`, delta `-0.0924`
- tick `288661`, seconds `101.50`, LSTM `0.0205`, delta `-0.0907`
- tick `288437`, seconds `98.00`, LSTM `0.1169`, delta `+0.0720`
- tick `283477`, seconds `20.50`, LSTM `0.3854`, delta `+0.0596`
- tick `282805`, seconds `10.00`, LSTM `0.5725`, delta `-0.0586`
- tick `287221`, seconds `79.00`, LSTM `0.3795`, delta `-0.0543`
- tick `285333`, seconds `49.50`, LSTM `0.5767`, delta `+0.0529`
- tick `285493`, seconds `52.00`, LSTM `0.5459`, delta `-0.0471`

## Top 15 local ridge features

- `lag_14__CT_place_BDOORS`: coefficient `0.001893`, |coef| `0.001893`
- `lag_15__T_place_EXTENDEDA`: coefficient `-0.001782`, |coef| `0.001782`
- `lag_13__T5__flash_duration`: coefficient `0.001673`, |coef| `0.001673`
- `lag_00__T_kills_last_3s`: coefficient `-0.001516`, |coef| `0.001516`
- `lag_13__T_place_EXTENDEDA`: coefficient `-0.001509`, |coef| `0.001509`
- `lag_02__T_place_EXTENDEDA`: coefficient `-0.001491`, |coef| `0.001491`
- `lag_14__T4__flash_duration`: coefficient `0.001487`, |coef| `0.001487`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001429`, |coef| `0.001429`
- `lag_10__CT_place_BDOORS`: coefficient `0.001428`, |coef| `0.001428`
- `lag_00__T_damage_last_5s`: coefficient `-0.001427`, |coef| `0.001427`
- `lag_00__damage_diff_last_5s`: coefficient `0.001403`, |coef| `0.001403`
- `lag_14__T_place_EXTENDEDA`: coefficient `-0.001363`, |coef| `0.001363`
- `lag_00__kill_diff_last_3s`: coefficient `0.001310`, |coef| `0.001310`
- `lag_02__CT2__flash_duration`: coefficient `-0.001178`, |coef| `0.001178`
- `lag_00__CT4__duck_amount`: coefficient `0.001110`, |coef| `0.001110`

## Top 10 utility ridge features

- `lag_13__T5__flash_duration`: coefficient `0.001673` (raises CT win probability)
- `lag_14__T4__flash_duration`: coefficient `0.001487` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `-0.001178` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.001094` (lowers CT win probability)
- `lag_03__T_he_last_5s`: coefficient `0.000932` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000885` (raises CT win probability)
- `lag_14__T_he_last_5s`: coefficient `0.000841` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.000831` (lowers CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `-0.000801` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.000788` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_BDOORS`: coefficient `0.001893` (raises CT win probability)
- `lag_15__T_place_EXTENDEDA`: coefficient `-0.001782` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001516` (lowers CT win probability)
- `lag_13__T_place_EXTENDEDA`: coefficient `-0.001509` (lowers CT win probability)
- `lag_02__T_place_EXTENDEDA`: coefficient `-0.001491` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001429` (lowers CT win probability)
- `lag_10__CT_place_BDOORS`: coefficient `0.001428` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001427` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001403` (raises CT win probability)
- `lag_14__T_place_EXTENDEDA`: coefficient `-0.001363` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `287701`, seconds `86.50`, LSTM delta `-0.2583`

Top all feature movements:
- `lag_13__T5__flash_duration`: contribution `-0.012610`
- `lag_14__T4__flash_duration`: contribution `-0.010565`
- `lag_14__CT_place_BDOORS`: contribution `-0.009105`
- `lag_15__T_place_EXTENDEDA`: contribution `-0.008834`
- `lag_00__T_shots_fired_sum`: contribution `-0.007501`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.012610`
- `lag_14__T4__flash_duration`: contribution `-0.010565`
- `lag_06__T2__flash_duration`: contribution `-0.006118`
- `lag_15__T1__flash_duration`: contribution `-0.005643`
- `lag_15__T_flash_duration_sum`: contribution `-0.003317`

### tick `283157`, seconds `15.50`, LSTM delta `-0.1508`

Top all feature movements:
- `lag_11__CT_place_HOLE`: contribution `-0.011532`
- `lag_14__T_he_last_5s`: contribution `-0.010972`
- `lag_13__CT_place_BDOORS`: contribution `-0.010477`
- `lag_02__CT2__flash_duration`: contribution `-0.009509`
- `lag_13__CT_place_HOLE`: contribution `-0.007577`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `-0.010972`
- `lag_02__CT2__flash_duration`: contribution `-0.009509`
- `lag_05__T3__flash_duration`: contribution `-0.001795`
- `lag_02__CT_flash_duration_sum`: contribution `-0.001733`

### tick `283189`, seconds `16.00`, LSTM delta `-0.0924`

Top all feature movements:
- `lag_14__CT_place_BDOORS`: contribution `-0.018209`
- `lag_15__T_he_last_5s`: contribution `-0.009676`
- `lag_03__CT2__flash_duration`: contribution `-0.006235`
- `lag_00__T_shots_fired_sum`: contribution `-0.005358`
- `lag_12__CT_place_HOLE`: contribution `-0.005135`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `-0.009676`
- `lag_03__CT2__flash_duration`: contribution `-0.006235`

### tick `288661`, seconds `101.50`, LSTM delta `-0.0907`

Top all feature movements:
- `lag_14__T_place_EXTENDEDA`: contribution `-0.006757`
- `lag_00__T_kills_last_3s`: contribution `-0.004803`
- `lag_06__CT_shots_fired_sum`: contribution `-0.004550`
- `lag_15__T1__flash_duration`: contribution `+0.004192`
- `lag_12__T_bomb_zone_count`: contribution `-0.004096`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.004192`
- `lag_15__T_flash_duration_sum`: contribution `+0.003461`
- `lag_15__T2__flash_duration`: contribution `+0.002206`

### tick `288437`, seconds `98.00`, LSTM delta `+0.0720`

Top all feature movements:
- `lag_13__T_place_EXTENDEDA`: contribution `+0.007483`
- `lag_02__T_place_EXTENDEDA`: contribution `+0.007392`
- `lag_08__CT4__flash_duration`: contribution `+0.004345`
- `lag_12__T_bomb_zone_count`: contribution `+0.004096`
- `lag_02__T_place_SHORTSTAIRS`: contribution `+0.004061`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `+0.004345`
- `lag_08__T2__flash_duration`: contribution `-0.002008`
- `lag_05__CT2__flash_duration`: contribution `+0.001206`
