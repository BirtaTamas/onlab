# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `122392`, seconds `43.50`, LSTM `0.2367`, delta `-0.3149`
- tick `122584`, seconds `46.50`, LSTM `0.3947`, delta `+0.1883`
- tick `126520`, seconds `108.00`, LSTM `0.0311`, delta `-0.1506`
- tick `126456`, seconds `107.00`, LSTM `0.2874`, delta `-0.1367`
- tick `126488`, seconds `107.50`, LSTM `0.1818`, delta `-0.1056`
- tick `122616`, seconds `47.00`, LSTM `0.4914`, delta `+0.0967`
- tick `126200`, seconds `103.00`, LSTM `0.4254`, delta `-0.0901`
- tick `121048`, seconds `22.50`, LSTM `0.5650`, delta `-0.0489`
- tick `122552`, seconds `46.00`, LSTM `0.2064`, delta `-0.0474`
- tick `122072`, seconds `38.50`, LSTM `0.6058`, delta `+0.0432`

## Top 15 local ridge features

- `lag_00__T_place_GRAVEYARD`: coefficient `-0.003220`, |coef| `0.003220`
- `lag_13__T_place_DECK`: coefficient `0.003189`, |coef| `0.003189`
- `lag_02__T_place_GRAVEYARD`: coefficient `-0.003056`, |coef| `0.003056`
- `lag_01__T_place_GRAVEYARD`: coefficient `-0.002632`, |coef| `0.002632`
- `lag_00__kill_diff_last_3s`: coefficient `0.001670`, |coef| `0.001670`
- `lag_11__CT4__flash_duration`: coefficient `-0.001598`, |coef| `0.001598`
- `lag_00__T_kills_last_3s`: coefficient `-0.001597`, |coef| `0.001597`
- `lag_00__T_place_PIT`: coefficient `-0.001565`, |coef| `0.001565`
- `lag_01__T_place_PIT`: coefficient `-0.001361`, |coef| `0.001361`
- `lag_04__T5__duck_amount`: coefficient `-0.001289`, |coef| `0.001289`
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001282`, |coef| `0.001282`
- `lag_00__damage_diff_last_5s`: coefficient `0.001236`, |coef| `0.001236`
- `lag_01__T_place_BALCONY`: coefficient `-0.001189`, |coef| `0.001189`
- `lag_15__T_place_QUAD`: coefficient `0.001162`, |coef| `0.001162`
- `lag_10__T_flashed_players`: coefficient `-0.001160`, |coef| `0.001160`

## Top 10 utility ridge features

- `lag_11__CT4__flash_duration`: coefficient `-0.001598` (lowers CT win probability)
- `lag_03__T_utility_damage_last_5s`: coefficient `-0.001282` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.001158` (lowers CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `0.001133` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001093` (lowers CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.000989` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000971` (lowers CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `-0.000967` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000948` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000900` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_GRAVEYARD`: coefficient `-0.003220` (lowers CT win probability)
- `lag_13__T_place_DECK`: coefficient `0.003189` (raises CT win probability)
- `lag_02__T_place_GRAVEYARD`: coefficient `-0.003056` (lowers CT win probability)
- `lag_01__T_place_GRAVEYARD`: coefficient `-0.002632` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001670` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001597` (lowers CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.001565` (lowers CT win probability)
- `lag_01__T_place_PIT`: coefficient `-0.001361` (lowers CT win probability)
- `lag_04__T5__duck_amount`: coefficient `-0.001289` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001236` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `122392`, seconds `43.50`, LSTM delta `-0.3149`

Top all feature movements:
- `lag_13__T_place_DECK`: contribution `-0.077350`
- `lag_01__T_place_BALCONY`: contribution `-0.016357`
- `lag_10__T_place_BALCONY`: contribution `-0.011933`
- `lag_11__CT4__flash_duration`: contribution `-0.009489`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.008603`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `-0.009489`
- `lag_03__T_utility_damage_last_5s`: contribution `-0.008603`
- `lag_05__T5__flash_duration`: contribution `-0.007804`
- `lag_05__T3__flash_duration`: contribution `-0.007351`
- `lag_05__T_flash_duration_sum`: contribution `-0.005385`

### tick `122584`, seconds `46.50`, LSTM delta `+0.1883`

Top all feature movements:
- `lag_00__T_place_PIT`: contribution `+0.009873`
- `lag_00__kill_diff_last_3s`: contribution `+0.008038`
- `lag_09__T_utility_damage_last_5s`: contribution `+0.007605`
- `lag_07__T_place_BALCONY`: contribution `+0.007226`
- `lag_11__T5__flash_duration`: contribution `+0.007060`

Top utility-only movements:
- `lag_09__T_utility_damage_last_5s`: contribution `+0.007605`
- `lag_11__T5__flash_duration`: contribution `+0.007060`
- `lag_00__T5__flash_duration`: contribution `+0.006426`
- `lag_15__T4__flash_duration`: contribution `+0.006162`
- `lag_11__CT4__flash_duration`: contribution `+0.005405`

### tick `126520`, seconds `108.00`, LSTM delta `-0.1506`

Top all feature movements:
- `lag_02__T_place_GRAVEYARD`: contribution `-0.060082`
- `lag_01__T_place_PIT`: contribution `-0.008585`
- `lag_09__T_place_ARCH`: contribution `-0.007537`
- `lag_00__T_kills_last_3s`: contribution `-0.005058`
- `lag_12__T_bomb_zone_count`: contribution `-0.004630`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.002667`

### tick `126456`, seconds `107.00`, LSTM delta `-0.1367`

Top all feature movements:
- `lag_00__T_place_GRAVEYARD`: contribution `-0.063307`
- `lag_14__T_place_QUAD`: contribution `-0.015879`
- `lag_15__T_place_ARCH`: contribution `-0.009989`
- `lag_03__CT_place_LIBRARY`: contribution `-0.005008`
- `lag_10__T_bomb_zone_count`: contribution `-0.004682`

Top utility-only movements:
- `lag_10__T4__flash_duration`: contribution `-0.003112`
- `lag_10__T_flash_duration_sum`: contribution `-0.001800`

### tick `126488`, seconds `107.50`, LSTM delta `-0.1056`

Top all feature movements:
- `lag_01__T_place_GRAVEYARD`: contribution `-0.051736`
- `lag_15__T_place_QUAD`: contribution `-0.027977`
- `lag_00__T_place_PIT`: contribution `-0.009873`
- `lag_11__T_bomb_zone_count`: contribution `-0.004162`
- `lag_04__CT_place_LIBRARY`: contribution `-0.003665`

Top utility-only movements:
- `lag_11__T4__flash_duration`: contribution `-0.003416`
- `lag_00__T4__flash_duration`: contribution `-0.001695`
- `lag_11__T_flash_duration_sum`: contribution `+0.001463`
