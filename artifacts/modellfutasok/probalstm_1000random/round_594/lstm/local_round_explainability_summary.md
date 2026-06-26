# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `11`

## Largest probability jumps

- tick `98869`, seconds `84.50`, LSTM `0.7831`, delta `+0.0978`
- tick `97141`, seconds `57.50`, LSTM `0.6715`, delta `+0.0711`
- tick `97525`, seconds `63.50`, LSTM `0.7210`, delta `+0.0676`
- tick `99125`, seconds `88.50`, LSTM `0.8998`, delta `+0.0558`
- tick `97941`, seconds `70.00`, LSTM `0.7180`, delta `-0.0539`
- tick `98069`, seconds `72.00`, LSTM `0.6519`, delta `-0.0514`
- tick `97333`, seconds `60.50`, LSTM `0.6359`, delta `-0.0505`
- tick `98101`, seconds `72.50`, LSTM `0.6982`, delta `+0.0463`
- tick `99381`, seconds `92.50`, LSTM `0.9669`, delta `+0.0404`
- tick `97429`, seconds `62.00`, LSTM `0.6859`, delta `+0.0384`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001260`, |coef| `0.001260`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001239`, |coef| `0.001239`
- `lag_00__T_macro_B`: coefficient `-0.001239`, |coef| `0.001239`
- `lag_00__CT_place_SECONDMID`: coefficient `0.001191`, |coef| `0.001191`
- `lag_04__CT_place_TRAMP`: coefficient `0.001110`, |coef| `0.001110`
- `lag_00__kill_diff_last_3s`: coefficient `0.001103`, |coef| `0.001103`
- `lag_12__CT_place_TRAMP`: coefficient `0.001060`, |coef| `0.001060`
- `lag_00__T_flashed_players`: coefficient `-0.001054`, |coef| `0.001054`
- `lag_10__CT_place_TRAMP`: coefficient `0.001005`, |coef| `0.001005`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000966`, |coef| `0.000966`
- `lag_14__T5__duck_amount`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_15__CT3__flash_duration`: coefficient `0.000900`, |coef| `0.000900`
- `lag_03__CT3__is_walking`: coefficient `-0.000861`, |coef| `0.000861`
- `lag_01__CT3__flash_duration`: coefficient `-0.000856`, |coef| `0.000856`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000833`, |coef| `0.000833`

## Top 10 utility ridge features

- `lag_15__CT3__flash_duration`: coefficient `0.000900` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `-0.000856` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000698` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000550` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000546` (lowers CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `0.000524` (raises CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.000512` (lowers CT win probability)
- `lag_07__CT3__flash_duration`: coefficient `-0.000507` (lowers CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.000498` (lowers CT win probability)
- `lag_14__CT_utility_damage_last_5s`: coefficient `-0.000497` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001260` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001239` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001239` (lowers CT win probability)
- `lag_00__CT_place_SECONDMID`: coefficient `0.001191` (raises CT win probability)
- `lag_04__CT_place_TRAMP`: coefficient `0.001110` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001103` (raises CT win probability)
- `lag_12__CT_place_TRAMP`: coefficient `0.001060` (raises CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001054` (lowers CT win probability)
- `lag_10__CT_place_TRAMP`: coefficient `0.001005` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000966` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `98869`, seconds `84.50`, LSTM delta `+0.0978`

Top all feature movements:
- `lag_04__CT_place_TRAMP`: contribution `+0.014956`
- `lag_15__CT3__flash_duration`: contribution `+0.007428`
- `lag_01__CT3__flash_duration`: contribution `+0.007066`
- `lag_13__CT_place_UNDERPASS`: contribution `+0.003887`
- `lag_00__CT_kills_last_3s`: contribution `+0.003639`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `+0.007428`
- `lag_01__CT3__flash_duration`: contribution `+0.007066`
- `lag_15__CT_flash_duration_sum`: contribution `+0.002851`
- `lag_01__CT_flash_duration_sum`: contribution `+0.001628`

### tick `97141`, seconds `57.50`, LSTM delta `+0.0711`

Top all feature movements:
- `lag_00__T_flashed_players`: contribution `+0.004067`
- `lag_00__CT_kills_last_3s`: contribution `+0.003639`
- `lag_14__T5__duck_amount`: contribution `+0.003464`
- `lag_02__CT_place_ARCH`: contribution `+0.002841`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002685`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.002164`
- `lag_13__CT3__smoke`: contribution `+0.001096`
- `lag_00__T_flash_duration_sum`: contribution `+0.001017`
- `lag_00__T4__smoke`: contribution `+0.001014`

### tick `97525`, seconds `63.50`, LSTM delta `+0.0676`

Top all feature movements:
- `lag_00__CT_place_SECONDMID`: contribution `+0.024411`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007852`
- `lag_00__CT_place_BALCONY`: contribution `+0.005343`
- `lag_01__CT2__shots_fired`: contribution `+0.002906`
- `lag_06__CT_shots_fired_sum`: contribution `+0.002856`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99125`, seconds `88.50`, LSTM delta `+0.0558`

Top all feature movements:
- `lag_12__CT_place_TRAMP`: contribution `+0.014280`
- `lag_00__CT_place_TRAMP`: contribution `+0.007496`
- `lag_09__CT3__flash_duration`: contribution `+0.004112`
- `lag_00__CT_kills_last_3s`: contribution `+0.003639`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002685`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.004112`
- `lag_09__CT_flash_duration_sum`: contribution `+0.000895`

### tick `97941`, seconds `70.00`, LSTM delta `-0.0539`

Top all feature movements:
- `lag_00__CT_place_SECONDMID`: contribution `-0.024411`
- `lag_13__CT_place_SECONDMID`: contribution `-0.009007`
- `lag_14__CT_shots_fired_sum`: contribution `-0.006712`
- `lag_06__CT_place_BALCONY`: contribution `-0.001987`
- `lag_12__CT_place_RUINS`: contribution `-0.001867`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.001807`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.001171`
