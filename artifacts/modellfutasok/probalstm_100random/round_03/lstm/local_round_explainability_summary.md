# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `44330`, seconds `35.50`, LSTM `0.2249`, delta `-0.4330`
- tick `44234`, seconds `34.00`, LSTM `0.6416`, delta `-0.1450`
- tick `44106`, seconds `32.00`, LSTM `0.7093`, delta `+0.1434`
- tick `43626`, seconds `24.50`, LSTM `0.6647`, delta `+0.1263`
- tick `44362`, seconds `36.00`, LSTM `0.1033`, delta `-0.1216`
- tick `43658`, seconds `25.00`, LSTM `0.5594`, delta `-0.1053`
- tick `44394`, seconds `36.50`, LSTM `0.0265`, delta `-0.0767`
- tick `43114`, seconds `16.50`, LSTM `0.4463`, delta `+0.0726`
- tick `43146`, seconds `17.00`, LSTM `0.5062`, delta `+0.0599`
- tick `42922`, seconds `13.50`, LSTM `0.4348`, delta `-0.0469`

## Top 15 local ridge features

- `lag_11__T_utility_damage_last_5s`: coefficient `0.004536`, |coef| `0.004536`
- `lag_00__kill_diff_last_3s`: coefficient `0.003839`, |coef| `0.003839`
- `lag_00__T_kills_last_3s`: coefficient `-0.002781`, |coef| `0.002781`
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.002523`, |coef| `0.002523`
- `lag_03__CT_shots_fired_sum`: coefficient `0.002373`, |coef| `0.002373`
- `lag_15__CT_place_PIT`: coefficient `-0.002166`, |coef| `0.002166`
- `lag_10__T2__duck_amount`: coefficient `0.002164`, |coef| `0.002164`
- `lag_00__CT_place_BANANA`: coefficient `0.002148`, |coef| `0.002148`
- `lag_14__T1__duck_amount`: coefficient `0.002083`, |coef| `0.002083`
- `lag_05__CT2__duck_amount`: coefficient `-0.002071`, |coef| `0.002071`
- `lag_00__CT_kills_last_3s`: coefficient `0.002071`, |coef| `0.002071`
- `lag_12__T_utility_damage_last_5s`: coefficient `0.001986`, |coef| `0.001986`
- `lag_07__T2__shots_fired`: coefficient `-0.001827`, |coef| `0.001827`
- `lag_04__CT2__shots_fired`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_03__T_shots_fired_sum`: coefficient `0.001726`, |coef| `0.001726`

## Top 10 utility ridge features

- `lag_11__T_utility_damage_last_5s`: coefficient `0.004536` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.002523` (lowers CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `0.001986` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001388` (lowers CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001366` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.001304` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001165` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.001129` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.001128` (raises CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.001109` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003839` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002781` (lowers CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.002373` (raises CT win probability)
- `lag_15__CT_place_PIT`: coefficient `-0.002166` (lowers CT win probability)
- `lag_10__T2__duck_amount`: coefficient `0.002164` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.002148` (raises CT win probability)
- `lag_14__T1__duck_amount`: coefficient `0.002083` (raises CT win probability)
- `lag_05__CT2__duck_amount`: coefficient `-0.002071` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002071` (raises CT win probability)
- `lag_07__T2__shots_fired`: coefficient `-0.001827` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44330`, seconds `35.50`, LSTM delta `-0.4330`

Top all feature movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.045335`
- `lag_00__kill_diff_last_3s`: contribution `-0.018482`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.014124`
- `lag_03__CT_shots_fired_sum`: contribution `-0.009893`
- `lag_15__CT_place_PIT`: contribution `-0.009327`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.045335`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.014124`

### tick `44234`, seconds `34.00`, LSTM delta `-0.1450`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.009893`
- `lag_08__T_utility_damage_last_5s`: contribution `-0.009387`
- `lag_00__kill_diff_last_3s`: contribution `-0.009241`
- `lag_00__T_kills_last_3s`: contribution `-0.008811`
- `lag_00__CT_place_BANANA`: contribution `-0.006358`

Top utility-only movements:
- `lag_08__T_utility_damage_last_5s`: contribution `-0.009387`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.003060`

### tick `44106`, seconds `32.00`, LSTM delta `+0.1434`

Top all feature movements:
- `lag_04__T_utility_damage_last_5s`: contribution `+0.011647`
- `lag_00__kill_diff_last_3s`: contribution `+0.009241`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.008005`
- `lag_14__CT3__flash_duration`: contribution `+0.006667`
- `lag_00__CT_kills_last_3s`: contribution `+0.005979`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `+0.011647`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.008005`
- `lag_14__CT3__flash_duration`: contribution `+0.006667`
- `lag_15__CT1__flash_duration`: contribution `+0.004969`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.003408`

### tick `43626`, seconds `24.50`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009647`
- `lag_13__CT_place_BALCONY`: contribution `+0.009311`
- `lag_00__kill_diff_last_3s`: contribution `+0.009241`
- `lag_12__CT3__flash_duration`: contribution `+0.008393`
- `lag_01__CT_place_BALCONY`: contribution `+0.006129`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009647`
- `lag_12__CT3__flash_duration`: contribution `+0.008393`
- `lag_12__CT_flash_duration_sum`: contribution `+0.002699`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001829`

### tick `44362`, seconds `36.00`, LSTM delta `-0.1216`

Top all feature movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.019851`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.006208`
- `lag_15__T1__duck_amount`: contribution `-0.005885`
- `lag_01__kill_diff_last_3s`: contribution `-0.004808`
- `lag_07__CT_shots_fired_sum`: contribution `+0.004041`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.019851`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.006208`
