# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `5`

## Largest probability jumps

- tick `30463`, seconds `104.00`, LSTM `0.3090`, delta `-0.2642`
- tick `30079`, seconds `98.00`, LSTM `0.5332`, delta `-0.1013`
- tick `30367`, seconds `102.50`, LSTM `0.5077`, delta `-0.0931`
- tick `30399`, seconds `103.00`, LSTM `0.5747`, delta `+0.0670`
- tick `30719`, seconds `108.00`, LSTM `0.1390`, delta `-0.0601`
- tick `30271`, seconds `101.00`, LSTM `0.6065`, delta `+0.0510`
- tick `30623`, seconds `106.50`, LSTM `0.1755`, delta `-0.0479`
- tick `31583`, seconds `121.50`, LSTM `0.2031`, delta `+0.0459`
- tick `31007`, seconds `112.50`, LSTM `0.1264`, delta `-0.0418`
- tick `31807`, seconds `125.00`, LSTM `0.1633`, delta `-0.0368`

## Top 15 local ridge features

- `lag_03__T_place_QUAD`: coefficient `0.002323`, |coef| `0.002323`
- `lag_00__T_place_QUAD`: coefficient `0.001928`, |coef| `0.001928`
- `lag_04__T1__duck_amount`: coefficient `0.001530`, |coef| `0.001530`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001271`, |coef| `0.001271`
- `lag_00__CT_A_site_active_infernos`: coefficient `0.001230`, |coef| `0.001230`
- `lag_10__T2__duck_amount`: coefficient `-0.001125`, |coef| `0.001125`
- `lag_14__CT1__shots_fired`: coefficient `-0.001112`, |coef| `0.001112`
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_06__T1__duck_amount`: coefficient `0.001082`, |coef| `0.001082`
- `lag_02__T1__duck_amount`: coefficient `0.001080`, |coef| `0.001080`
- `lag_09__T_place_PIT`: coefficient `-0.001045`, |coef| `0.001045`
- `lag_00__CT2__duck_amount`: coefficient `0.001007`, |coef| `0.001007`
- `lag_13__CT1__shots_fired`: coefficient `-0.000999`, |coef| `0.000999`
- `lag_02__CT1__shots_fired`: coefficient `-0.000997`, |coef| `0.000997`
- `lag_10__CT1__shots_fired`: coefficient `-0.000980`, |coef| `0.000980`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `0.001230` (raises CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.000932` (raises CT win probability)
- `lag_03__T5__flash_duration`: coefficient `0.000916` (raises CT win probability)
- `lag_03__T3__flash_duration`: coefficient `0.000908` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000888` (raises CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.000820` (raises CT win probability)
- `lag_04__CT_active_infernos`: coefficient `0.000775` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000764` (lowers CT win probability)
- `lag_12__CT3__molly`: coefficient `0.000716` (raises CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `0.000683` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_QUAD`: coefficient `0.002323` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.001928` (raises CT win probability)
- `lag_04__T1__duck_amount`: coefficient `0.001530` (raises CT win probability)
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001271` (raises CT win probability)
- `lag_10__T2__duck_amount`: coefficient `-0.001125` (lowers CT win probability)
- `lag_14__CT1__shots_fired`: coefficient `-0.001112` (lowers CT win probability)
- `lag_11__CT_place_LIBRARY`: coefficient `-0.001099` (lowers CT win probability)
- `lag_06__T1__duck_amount`: coefficient `0.001082` (raises CT win probability)
- `lag_02__T1__duck_amount`: coefficient `0.001080` (raises CT win probability)
- `lag_09__T_place_PIT`: coefficient `-0.001045` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `30463`, seconds `104.00`, LSTM delta `-0.2642`

Top all feature movements:
- `lag_03__T_place_QUAD`: contribution `-0.111915`
- `lag_06__T_place_QUAD`: contribution `-0.021966`
- `lag_07__T_place_QUAD`: contribution `-0.017086`
- `lag_03__T5__flash_duration`: contribution `-0.006951`
- `lag_12__CT_shots_fired_sum`: contribution `-0.006829`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.006951`
- `lag_03__T3__flash_duration`: contribution `-0.005675`
- `lag_02__T2__flash_duration`: contribution `-0.004596`
- `lag_04__CT2__flash_duration`: contribution `-0.003318`
- `lag_03__T_flash_duration_sum`: contribution `-0.003195`

### tick `30079`, seconds `98.00`, LSTM delta `-0.1013`

Top all feature movements:
- `lag_04__T_flash_duration_sum`: contribution `-0.008549`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008045`
- `lag_04__T_flashed_players`: contribution `-0.005381`
- `lag_00__CT_place_TOPOFMID`: contribution `-0.004614`
- `lag_04__T2__flash_duration`: contribution `-0.004137`

Top utility-only movements:
- `lag_04__T_flash_duration_sum`: contribution `-0.008549`
- `lag_04__T2__flash_duration`: contribution `-0.004137`
- `lag_04__T3__flash_duration`: contribution `-0.004085`
- `lag_04__T5__flash_duration`: contribution `-0.003862`
- `lag_03__CT2__flash_duration`: contribution `-0.002680`

### tick `30367`, seconds `102.50`, LSTM delta `-0.0931`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `-0.092870`
- `lag_03__T_place_QUAD`: contribution `+0.055957`
- `lag_09__CT_shots_fired_sum`: contribution `-0.004144`
- `lag_00__T5__flash_duration`: contribution `-0.004057`
- `lag_04__T_place_QUAD`: contribution `+0.003926`

Top utility-only movements:
- `lag_00__T5__flash_duration`: contribution `-0.004057`
- `lag_13__T_flash_duration_sum`: contribution `-0.002774`
- `lag_00__T3__flash_duration`: contribution `-0.002464`
- `lag_13__T3__flash_duration`: contribution `-0.002141`
- `lag_13__T5__flash_duration`: contribution `-0.001864`

### tick `30399`, seconds `103.00`, LSTM delta `+0.0670`

Top all feature movements:
- `lag_01__T_place_QUAD`: contribution `+0.021062`
- `lag_05__T_place_QUAD`: contribution `+0.020455`
- `lag_10__CT_shots_fired_sum`: contribution `+0.005102`
- `lag_04__T_place_QUAD`: contribution `+0.003926`
- `lag_09__T_place_ARCH`: contribution `+0.003457`

Top utility-only movements:
- `lag_14__T_flash_duration_sum`: contribution `+0.002826`
- `lag_14__T5__flash_duration`: contribution `+0.002368`
- `lag_01__T5__flash_duration`: contribution `-0.002341`
- `lag_14__T3__flash_duration`: contribution `+0.002158`
- `lag_01__T3__flash_duration`: contribution `-0.002036`

### tick `30719`, seconds `108.00`, LSTM delta `-0.0601`

Top all feature movements:
- `lag_15__T_place_QUAD`: contribution `-0.015129`
- `lag_00__T_place_PIT`: contribution `-0.005410`
- `lag_10__T2__flash_duration`: contribution `-0.003592`
- `lag_00__CT5__is_scoped`: contribution `-0.003171`
- `lag_00__T_bomb_zone_count`: contribution `+0.002922`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `-0.003592`
- `lag_11__T5__flash_duration`: contribution `-0.002572`
- `lag_11__T3__flash_duration`: contribution `-0.002116`
- `lag_11__T_flash_duration_sum`: contribution `-0.001847`
