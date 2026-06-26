# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `1`

## Largest probability jumps

- tick `4338`, seconds `48.00`, LSTM `0.0547`, delta `-0.1365`
- tick `3154`, seconds `29.50`, LSTM `0.3694`, delta `-0.1167`
- tick `3410`, seconds `33.50`, LSTM `0.3160`, delta `+0.0790`
- tick `3378`, seconds `33.00`, LSTM `0.2370`, delta `+0.0652`
- tick `3282`, seconds `31.50`, LSTM `0.2494`, delta `-0.0571`
- tick `3218`, seconds `30.50`, LSTM `0.3056`, delta `-0.0470`
- tick `3122`, seconds `29.00`, LSTM `0.4861`, delta `+0.0448`
- tick `3346`, seconds `32.50`, LSTM `0.1717`, delta `-0.0435`
- tick `4594`, seconds `52.00`, LSTM `0.0222`, delta `-0.0390`
- tick `3058`, seconds `28.00`, LSTM `0.4254`, delta `+0.0381`

## Top 15 local ridge features

- `lag_10__CT_place_BRIDGE`: coefficient `-0.003192`, |coef| `0.003192`
- `lag_15__CT3__duck_amount`: coefficient `0.001631`, |coef| `0.001631`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001466`, |coef| `0.001466`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001379`, |coef| `0.001379`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_00__T_macro_B`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001357`, |coef| `0.001357`
- `lag_08__CT3__duck_amount`: coefficient `0.001321`, |coef| `0.001321`
- `lag_09__CT_place_ALLEY`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_10__CT_place_MIDDLE`: coefficient `0.001126`, |coef| `0.001126`
- `lag_00__T_kills_last_3s`: coefficient `-0.001113`, |coef| `0.001113`
- `lag_08__T_place_OUTSIDELONG`: coefficient `-0.001100`, |coef| `0.001100`
- `lag_08__CT_place_BRIDGE`: coefficient `-0.001074`, |coef| `0.001074`
- `lag_11__CT_place_BRIDGE`: coefficient `-0.001071`, |coef| `0.001071`
- `lag_08__CT_place_WALKWAY`: coefficient `-0.001058`, |coef| `0.001058`

## Top 10 utility ridge features

- `lag_08__CT5__flash_duration`: coefficient `-0.000825` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.000822` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000667` (lowers CT win probability)
- `lag_08__CT2__smoke`: coefficient `0.000636` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000617` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.000518` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.000514` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000507` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.000491` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.000412` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_BRIDGE`: coefficient `-0.003192` (lowers CT win probability)
- `lag_15__CT3__duck_amount`: coefficient `0.001631` (raises CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001466` (raises CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001379` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001368` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001368` (lowers CT win probability)
- `lag_14__T_place_CONNECTOR`: coefficient `-0.001357` (lowers CT win probability)
- `lag_08__CT3__duck_amount`: coefficient `0.001321` (raises CT win probability)
- `lag_09__CT_place_ALLEY`: coefficient `-0.001127` (lowers CT win probability)
- `lag_10__CT_place_MIDDLE`: coefficient `0.001126` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `4338`, seconds `48.00`, LSTM delta `-0.1365`

Top all feature movements:
- `lag_10__CT_place_BRIDGE`: contribution `-0.036583`
- `lag_02__T_place_CONNECTOR`: contribution `-0.006677`
- `lag_15__CT3__duck_amount`: contribution `-0.006068`
- `lag_09__CT_place_SNIPERSNEST`: contribution `-0.004900`
- `lag_02__CT2__duck_amount`: contribution `-0.003885`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3154`, seconds `29.50`, LSTM delta `-0.1167`

Top all feature movements:
- `lag_00__T_place_BOMBSITEB`: contribution `-0.006404`
- `lag_00__T_macro_B`: contribution `-0.006404`
- `lag_08__CT_place_WALKWAY`: contribution `-0.005193`
- `lag_15__CT3__duck_amount`: contribution `-0.004885`
- `lag_08__T4__flash_duration`: contribution `-0.004809`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `-0.004809`
- `lag_08__CT5__flash_duration`: contribution `-0.004789`

### tick `3410`, seconds `33.50`, LSTM delta `+0.0790`

Top all feature movements:
- `lag_14__T_place_CONNECTOR`: contribution `+0.006570`
- `lag_08__T_place_OUTSIDELONG`: contribution `+0.005409`
- `lag_15__T_place_CONNECTOR`: contribution `+0.005020`
- `lag_02__T_bomb_zone_count`: contribution `+0.004665`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.004075`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.003029`
- `lag_06__CT5__flash_duration`: contribution `+0.002394`

### tick `3378`, seconds `33.00`, LSTM delta `+0.0652`

Top all feature movements:
- `lag_00__CT_place_CTSIDEUPPER`: contribution `+0.007571`
- `lag_14__T_place_CONNECTOR`: contribution `+0.006570`
- `lag_01__T_bomb_zone_count`: contribution `+0.005535`
- `lag_08__CT_place_WALKWAY`: contribution `+0.005193`
- `lag_15__T_place_CONNECTOR`: contribution `-0.005020`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `+0.002984`

### tick `3282`, seconds `31.50`, LSTM delta `-0.0571`

Top all feature movements:
- `lag_14__T_place_CONNECTOR`: contribution `-0.006570`
- `lag_15__T_place_CONNECTOR`: contribution `-0.005020`
- `lag_12__T4__flash_duration`: contribution `-0.003906`
- `lag_12__CT5__flash_duration`: contribution `-0.003582`
- `lag_05__CT_place_WALKWAY`: contribution `-0.002534`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.003906`
- `lag_12__CT5__flash_duration`: contribution `-0.003582`
