# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `4`

## Largest probability jumps

- tick `33237`, seconds `83.00`, LSTM `0.0297`, delta `-0.0832`
- tick `27957`, seconds `0.50`, LSTM `0.0809`, delta `-0.0506`
- tick `32565`, seconds `72.50`, LSTM `0.1326`, delta `+0.0444`
- tick `32853`, seconds `77.00`, LSTM `0.1112`, delta `-0.0274`
- tick `32789`, seconds `76.00`, LSTM `0.1426`, delta `-0.0230`
- tick `32629`, seconds `73.50`, LSTM `0.1687`, delta `+0.0206`
- tick `32917`, seconds `78.00`, LSTM `0.1085`, delta `-0.0189`
- tick `32885`, seconds `77.50`, LSTM `0.1274`, delta `+0.0162`
- tick `32597`, seconds `73.00`, LSTM `0.1482`, delta `+0.0156`
- tick `32949`, seconds `78.50`, LSTM `0.0939`, delta `-0.0146`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000724`, |coef| `0.000724`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000645`, |coef| `0.000645`
- `lag_00__CT1__is_walking`: coefficient `0.000525`, |coef| `0.000525`
- `lag_00__CT_he_last_5s`: coefficient `-0.000525`, |coef| `0.000525`
- `lag_00__T_place_MAIN`: coefficient `-0.000489`, |coef| `0.000489`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000477`, |coef| `0.000477`
- `lag_06__T_place_MAIN`: coefficient `-0.000475`, |coef| `0.000475`
- `lag_00__CT_place_BRICKS`: coefficient `0.000474`, |coef| `0.000474`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `0.000469`, |coef| `0.000469`
- `lag_12__T_place_MAIN`: coefficient `-0.000442`, |coef| `0.000442`
- `lag_10__T_place_MAIN`: coefficient `-0.000441`, |coef| `0.000441`
- `lag_00__CT5__duck_amount`: coefficient `0.000440`, |coef| `0.000440`
- `lag_00__T_place_BRIDGE`: coefficient `-0.000434`, |coef| `0.000434`
- `lag_14__CT_place_OUTSIDELONG`: coefficient `0.000431`, |coef| `0.000431`
- `lag_01__T_place_MIDDLE`: coefficient `0.000417`, |coef| `0.000417`

## Top 10 utility ridge features

- `lag_00__CT_he_last_5s`: coefficient `-0.000525` (lowers CT win probability)
- `lag_14__CT_he_last_5s`: coefficient `0.000364` (raises CT win probability)
- `lag_12__CT_he_last_5s`: coefficient `0.000361` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000356` (raises CT win probability)
- `lag_01__CT3__smoke`: coefficient `-0.000331` (lowers CT win probability)
- `lag_02__T3__smoke`: coefficient `-0.000292` (lowers CT win probability)
- `lag_13__CT_he_last_5s`: coefficient `0.000282` (raises CT win probability)
- `lag_12__T2__flash_duration`: coefficient `-0.000279` (lowers CT win probability)
- `lag_06__T_active_smokes`: coefficient `-0.000278` (lowers CT win probability)
- `lag_12__CT5__smoke`: coefficient `-0.000274` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000724` (lowers CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000645` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.000525` (raises CT win probability)
- `lag_00__T_place_MAIN`: coefficient `-0.000489` (lowers CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000477` (lowers CT win probability)
- `lag_06__T_place_MAIN`: coefficient `-0.000475` (lowers CT win probability)
- `lag_00__CT_place_BRICKS`: coefficient `0.000474` (raises CT win probability)
- `lag_02__CT_place_OUTSIDELONG`: coefficient `0.000469` (raises CT win probability)
- `lag_12__T_place_MAIN`: coefficient `-0.000442` (lowers CT win probability)
- `lag_10__T_place_MAIN`: coefficient `-0.000441` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `33237`, seconds `83.00`, LSTM delta `-0.0832`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.009106`
- `lag_02__CT_place_BRICKS`: contribution `-0.006678`
- `lag_11__CT_place_BRICKS`: contribution `-0.006037`
- `lag_03__T_place_WALKWAY`: contribution `-0.004483`
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.004371`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `27957`, seconds `0.50`, LSTM delta `-0.0506`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.018659`
- `lag_00__CT_he_last_5s`: contribution `-0.009626`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.002107`
- `lag_01__T_place_TSPAWN`: contribution `-0.000708`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000629`

Top utility-only movements:
- `lag_00__CT_he_last_5s`: contribution `-0.009626`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.002107`
- `lag_01__CT3__smoke`: contribution `-0.000522`
- `lag_01__T2__smoke`: contribution `-0.000401`
- `lag_01__T3__smoke`: contribution `-0.000279`

### tick `32565`, seconds `72.50`, LSTM delta `+0.0444`

Top all feature movements:
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.006545`
- `lag_01__T_place_MIDDOORS`: contribution `+0.002026`
- `lag_00__CT1__is_walking`: contribution `+0.001226`
- `lag_12__T2__flash_duration`: contribution `+0.000985`
- `lag_13__T2__duck_amount`: contribution `+0.000963`

Top utility-only movements:
- `lag_12__T2__flash_duration`: contribution `+0.000985`
- `lag_01__CT3__smoke`: contribution `+0.000732`
- `lag_02__T3__smoke`: contribution `+0.000635`
- `lag_12__CT5__smoke`: contribution `+0.000601`

### tick `32853`, seconds `77.00`, LSTM delta `-0.0274`

Top all feature movements:
- `lag_02__CT_place_OUTSIDELONG`: contribution `-0.004759`
- `lag_00__T_place_MAIN`: contribution `-0.003161`
- `lag_09__CT_place_OUTSIDELONG`: contribution `-0.002035`
- `lag_07__T_place_BRIDGE`: contribution `-0.001181`
- `lag_10__T_place_MIDDOORS`: contribution `-0.001119`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32789`, seconds `76.00`, LSTM delta `-0.0230`

Top all feature movements:
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.006545`
- `lag_08__T_place_MIDDOORS`: contribution `-0.001280`
- `lag_07__CT_place_OUTSIDELONG`: contribution `-0.001154`
- `lag_14__CT4__is_walking`: contribution `-0.000951`
- `lag_01__CT4__is_walking`: contribution `+0.000916`

Top utility-only movements:
- No utility movement among the top local contributors.
