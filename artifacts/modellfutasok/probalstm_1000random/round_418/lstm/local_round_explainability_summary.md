# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `9`

## Largest probability jumps

- tick `63929`, seconds `67.00`, LSTM `0.2204`, delta `-0.3030`
- tick `66585`, seconds `108.50`, LSTM `0.6333`, delta `+0.2446`
- tick `63833`, seconds `65.50`, LSTM `0.5403`, delta `-0.2083`
- tick `66393`, seconds `105.50`, LSTM `0.2183`, delta `+0.1288`
- tick `63641`, seconds `62.50`, LSTM `0.7215`, delta `+0.1215`
- tick `61433`, seconds `28.00`, LSTM `0.6875`, delta `+0.1121`
- tick `61497`, seconds `29.00`, LSTM `0.5837`, delta `-0.1023`
- tick `63961`, seconds `67.50`, LSTM `0.1463`, delta `-0.0741`
- tick `64281`, seconds `72.50`, LSTM `0.0680`, delta `-0.0694`
- tick `66553`, seconds `108.00`, LSTM `0.3888`, delta `+0.0529`

## Top 15 local ridge features

- `lag_12__T_place_GRAVEYARD`: coefficient `-0.003616`, |coef| `0.003616`
- `lag_06__T_place_GRAVEYARD`: coefficient `-0.002775`, |coef| `0.002775`
- `lag_04__T_place_QUAD`: coefficient `0.002599`, |coef| `0.002599`
- `lag_10__T_place_QUAD`: coefficient `-0.002534`, |coef| `0.002534`
- `lag_03__T_place_QUAD`: coefficient `0.002370`, |coef| `0.002370`
- `lag_05__CT_place_LIBRARY`: coefficient `-0.002346`, |coef| `0.002346`
- `lag_00__T_place_QUAD`: coefficient `0.002246`, |coef| `0.002246`
- `lag_06__T_macro_A`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_06__T_place_BOMBSITEA`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_15__T_place_PIT`: coefficient `-0.002188`, |coef| `0.002188`
- `lag_09__T_place_QUAD`: coefficient `-0.001980`, |coef| `0.001980`
- `lag_00__kill_diff_last_3s`: coefficient `0.001978`, |coef| `0.001978`
- `lag_02__CT_place_ARCH`: coefficient `0.001764`, |coef| `0.001764`
- `lag_11__T5__is_scoped`: coefficient `0.001732`, |coef| `0.001732`
- `lag_01__T_place_QUAD`: coefficient `0.001686`, |coef| `0.001686`

## Top 10 utility ridge features

- `lag_14__T2__flash_duration`: coefficient `-0.001108` (lowers CT win probability)
- `lag_07__T4__molly`: coefficient `-0.001075` (lowers CT win probability)
- `lag_13__CT_utility_damage_last_5s`: coefficient `-0.001063` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000912` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `0.000902` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `-0.000852` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `-0.000842` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `-0.000694` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.000660` (lowers CT win probability)
- `lag_01__T4__molly`: coefficient `-0.000652` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_GRAVEYARD`: coefficient `-0.003616` (lowers CT win probability)
- `lag_06__T_place_GRAVEYARD`: coefficient `-0.002775` (lowers CT win probability)
- `lag_04__T_place_QUAD`: coefficient `0.002599` (raises CT win probability)
- `lag_10__T_place_QUAD`: coefficient `-0.002534` (lowers CT win probability)
- `lag_03__T_place_QUAD`: coefficient `0.002370` (raises CT win probability)
- `lag_05__CT_place_LIBRARY`: coefficient `-0.002346` (lowers CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.002246` (raises CT win probability)
- `lag_06__T_macro_A`: coefficient `-0.002245` (lowers CT win probability)
- `lag_06__T_place_BOMBSITEA`: coefficient `-0.002245` (lowers CT win probability)
- `lag_15__T_place_PIT`: coefficient `-0.002188` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `63929`, seconds `67.00`, LSTM delta `-0.3030`

Top all feature movements:
- `lag_10__T_place_QUAD`: contribution `-0.061024`
- `lag_09__T_place_QUAD`: contribution `-0.047697`
- `lag_06__T_place_QUAD`: contribution `-0.013266`
- `lag_00__T_shots_fired_sum`: contribution `-0.012601`
- `lag_07__T_place_QUAD`: contribution `-0.008040`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.004169`

### tick `66585`, seconds `108.50`, LSTM delta `+0.2446`

Top all feature movements:
- `lag_12__T_place_GRAVEYARD`: contribution `+0.071084`
- `lag_05__CT_place_LIBRARY`: contribution `+0.015043`
- `lag_15__T_place_PIT`: contribution `+0.013810`
- `lag_11__T5__is_scoped`: contribution `+0.008263`
- `lag_06__T_macro_A`: contribution `+0.007483`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63833`, seconds `65.50`, LSTM delta `-0.2083`

Top all feature movements:
- `lag_04__T_place_QUAD`: contribution `-0.062597`
- `lag_03__T_place_QUAD`: contribution `-0.057096`
- `lag_06__T_place_QUAD`: contribution `+0.013266`
- `lag_00__kill_diff_last_3s`: contribution `-0.009520`
- `lag_07__T_place_QUAD`: contribution `+0.008040`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.002519`

### tick `66393`, seconds `105.50`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_06__T_place_GRAVEYARD`: contribution `+0.054547`
- `lag_09__T_place_PIT`: contribution `+0.007314`
- `lag_04__T5__is_scoped`: contribution `+0.006040`
- `lag_00__T_place_BOMBSITEA`: contribution `+0.004949`
- `lag_00__T_macro_A`: contribution `+0.004949`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63641`, seconds `62.50`, LSTM delta `+0.1215`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `+0.054087`
- `lag_01__T_place_QUAD`: contribution `+0.040619`
- `lag_00__kill_diff_last_3s`: contribution `+0.004760`
- `lag_00__CT_kills_last_3s`: contribution `+0.003993`
- `lag_00__T_shots_fired_sum`: contribution `+0.002520`

Top utility-only movements:
- No utility movement among the top local contributors.
