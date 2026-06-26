# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `10`

## Largest probability jumps

- tick `54256`, seconds `48.50`, LSTM `0.6230`, delta `+0.1636`
- tick `54416`, seconds `51.00`, LSTM `0.8387`, delta `+0.1234`
- tick `52464`, seconds `20.50`, LSTM `0.5887`, delta `-0.1187`
- tick `55536`, seconds `68.50`, LSTM `0.8622`, delta `-0.0857`
- tick `55376`, seconds `66.00`, LSTM `0.9473`, delta `+0.0778`
- tick `54544`, seconds `53.00`, LSTM `0.9520`, delta `+0.0746`
- tick `54576`, seconds `53.50`, LSTM `0.8915`, delta `-0.0605`
- tick `53968`, seconds `44.00`, LSTM `0.5582`, delta `+0.0600`
- tick `53904`, seconds `43.00`, LSTM `0.5101`, delta `+0.0592`
- tick `54896`, seconds `58.50`, LSTM `0.8447`, delta `+0.0567`

## Top 15 local ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.001374`, |coef| `0.001374`
- `lag_00__kill_diff_last_3s`: coefficient `0.001358`, |coef| `0.001358`
- `lag_15__T_place_FOUNTAIN`: coefficient `-0.001180`, |coef| `0.001180`
- `lag_00__damage_diff_last_5s`: coefficient `0.001173`, |coef| `0.001173`
- `lag_00__CT_place_BACKOFA`: coefficient `0.001148`, |coef| `0.001148`
- `lag_04__T_place_CONSTRUCTION`: coefficient `0.001066`, |coef| `0.001066`
- `lag_00__CT1__flash_duration`: coefficient `0.001050`, |coef| `0.001050`
- `lag_07__T_place_CONSTRUCTION`: coefficient `0.001034`, |coef| `0.001034`
- `lag_00__CT_kills_last_3s`: coefficient `0.001024`, |coef| `0.001024`
- `lag_06__T_place_CONSTRUCTION`: coefficient `0.000986`, |coef| `0.000986`
- `lag_00__T5__flash_duration`: coefficient `-0.000982`, |coef| `0.000982`
- `lag_04__T_place_WATER`: coefficient `-0.000981`, |coef| `0.000981`
- `lag_15__CT_place_LOBBY`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_06__T_place_WATER`: coefficient `-0.000890`, |coef| `0.000890`
- `lag_01__T_place_CONSTRUCTION`: coefficient `0.000858`, |coef| `0.000858`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `0.001050` (raises CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000982` (lowers CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `0.000842` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000824` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000807` (raises CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `0.000796` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.000771` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.000745` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000717` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.000699` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_FOUNTAIN`: coefficient `0.001374` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001358` (raises CT win probability)
- `lag_15__T_place_FOUNTAIN`: coefficient `-0.001180` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001173` (raises CT win probability)
- `lag_00__CT_place_BACKOFA`: coefficient `0.001148` (raises CT win probability)
- `lag_04__T_place_CONSTRUCTION`: coefficient `0.001066` (raises CT win probability)
- `lag_07__T_place_CONSTRUCTION`: coefficient `0.001034` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001024` (raises CT win probability)
- `lag_06__T_place_CONSTRUCTION`: coefficient `0.000986` (raises CT win probability)
- `lag_04__T_place_WATER`: coefficient `-0.000981` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `54256`, seconds `48.50`, LSTM delta `+0.1636`

Top all feature movements:
- `lag_04__T_place_CONSTRUCTION`: contribution `+0.013253`
- `lag_00__CT_place_BACKOFA`: contribution `+0.011084`
- `lag_15__CT_place_LOBBY`: contribution `+0.007345`
- `lag_06__T5__flash_duration`: contribution `+0.005987`
- `lag_09__CT1__flash_duration`: contribution `+0.005818`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.005987`
- `lag_09__CT1__flash_duration`: contribution `+0.005818`
- `lag_09__CT_flash_duration_sum`: contribution `+0.004867`
- `lag_00__CT1__flash_duration`: contribution `+0.003942`
- `lag_00__T1__flash_duration`: contribution `+0.002253`

### tick `54416`, seconds `51.00`, LSTM delta `+0.1234`

Top all feature movements:
- `lag_00__T5__flash_duration`: contribution `+0.007286`
- `lag_11__T5__flash_duration`: contribution `+0.006115`
- `lag_05__CT_place_BACKOFA`: contribution `+0.005802`
- `lag_14__CT_flash_duration_sum`: contribution `+0.004048`
- `lag_05__T1__flash_duration`: contribution `+0.003561`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `+0.007286`
- `lag_11__T5__flash_duration`: contribution `+0.006115`
- `lag_14__CT_flash_duration_sum`: contribution `+0.004048`
- `lag_05__T1__flash_duration`: contribution `+0.003561`
- `lag_14__CT1__flash_duration`: contribution `+0.003457`

### tick `52464`, seconds `20.50`, LSTM delta `-0.1187`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.014456`
- `lag_04__T_place_PIPE`: contribution `-0.007579`
- `lag_09__T_utility_damage_last_5s`: contribution `-0.006971`
- `lag_04__CT_place_UPPERPARK`: contribution `-0.005503`
- `lag_10__CT_place_UPPERPARK`: contribution `-0.004837`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `-0.006971`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.003102`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002420`
- `lag_09__utility_damage_diff_last_5s`: contribution `-0.002376`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.002187`

### tick `55536`, seconds `68.50`, LSTM delta `-0.0857`

Top all feature movements:
- `lag_09__CT_place_BRIDGE`: contribution `-0.008738`
- `lag_12__CT_place_BRIDGE`: contribution `-0.007512`
- `lag_14__T_place_CONSTRUCTION`: contribution `-0.005478`
- `lag_11__T_place_PIPE`: contribution `-0.004946`
- `lag_04__CT3__is_scoped`: contribution `-0.003389`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `-0.003328`
- `lag_10__CT3__flash_duration`: contribution `-0.001653`
- `lag_15__CT3__flash_duration`: contribution `-0.001609`

### tick `55376`, seconds `66.00`, LSTM delta `+0.0778`

Top all feature movements:
- `lag_07__CT_place_BRIDGE`: contribution `+0.009136`
- `lag_06__T_place_WATER`: contribution `+0.005077`
- `lag_04__CT_place_BRIDGE`: contribution `+0.004313`
- `lag_02__T_place_PIPE`: contribution `+0.004302`
- `lag_10__CT_place_WATER`: contribution `+0.004176`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.001653`
