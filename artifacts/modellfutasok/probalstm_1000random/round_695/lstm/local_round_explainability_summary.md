# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `24`

## Largest probability jumps

- tick `213452`, seconds `105.00`, LSTM `0.6511`, delta `-0.2099`
- tick `211436`, seconds `73.50`, LSTM `0.8445`, delta `+0.1594`
- tick `212620`, seconds `92.00`, LSTM `0.8223`, delta `-0.1037`
- tick `211692`, seconds `77.50`, LSTM `0.9379`, delta `+0.0760`
- tick `211468`, seconds `74.00`, LSTM `0.9002`, delta `+0.0557`
- tick `213004`, seconds `98.00`, LSTM `0.8495`, delta `-0.0483`
- tick `213580`, seconds `107.00`, LSTM `0.5913`, delta `-0.0469`
- tick `211660`, seconds `77.00`, LSTM `0.8619`, delta `-0.0451`
- tick `212588`, seconds `91.50`, LSTM `0.9260`, delta `-0.0417`
- tick `212716`, seconds `93.50`, LSTM `0.8198`, delta `-0.0407`

## Top 15 local ridge features

- `lag_00__CT_place_CANAL`: coefficient `0.003945`, |coef| `0.003945`
- `lag_13__T_duck_amount_mean`: coefficient `-0.003885`, |coef| `0.003885`
- `lag_00__T_kills_last_3s`: coefficient `-0.003384`, |coef| `0.003384`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002967`, |coef| `0.002967`
- `lag_00__kill_diff_last_3s`: coefficient `0.002845`, |coef| `0.002845`
- `lag_14__T_bomb_zone_count`: coefficient `-0.002793`, |coef| `0.002793`
- `lag_00__CT_place_CONSTRUCTION`: coefficient `0.002637`, |coef| `0.002637`
- `lag_13__T3__duck_amount`: coefficient `-0.002484`, |coef| `0.002484`
- `lag_00__CT1__alive`: coefficient `0.002413`, |coef| `0.002413`
- `lag_07__CT3__is_walking`: coefficient `0.002302`, |coef| `0.002302`
- `lag_11__T_duck_amount_mean`: coefficient `0.002293`, |coef| `0.002293`
- `lag_00__CT1__has_defuser`: coefficient `0.002140`, |coef| `0.002140`
- `lag_00__CT1__armor`: coefficient `0.002025`, |coef| `0.002025`
- `lag_00__CT1__has_helmet`: coefficient `0.001979`, |coef| `0.001979`
- `lag_00__T3__duck_amount`: coefficient `-0.001903`, |coef| `0.001903`

## Top 10 utility ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001586` (raises CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001469` (lowers CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.001320` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.001177` (lowers CT win probability)
- `lag_04__T3__flash_duration`: coefficient `0.000907` (raises CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.000708` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.000632` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `-0.000623` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000622` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.000595` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CANAL`: coefficient `0.003945` (raises CT win probability)
- `lag_13__T_duck_amount_mean`: coefficient `-0.003885` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003384` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002967` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002845` (raises CT win probability)
- `lag_14__T_bomb_zone_count`: coefficient `-0.002793` (lowers CT win probability)
- `lag_00__CT_place_CONSTRUCTION`: coefficient `0.002637` (raises CT win probability)
- `lag_13__T3__duck_amount`: coefficient `-0.002484` (lowers CT win probability)
- `lag_00__CT1__alive`: coefficient `0.002413` (raises CT win probability)
- `lag_07__CT3__is_walking`: coefficient `0.002302` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `213452`, seconds `105.00`, LSTM delta `-0.2099`

Top all feature movements:
- `lag_00__CT_place_CANAL`: contribution `-0.023979`
- `lag_13__T_duck_amount_mean`: contribution `-0.022595`
- `lag_14__T_bomb_zone_count`: contribution `-0.016259`
- `lag_00__T_duck_amount_mean`: contribution `-0.014216`
- `lag_11__T_duck_amount_mean`: contribution `-0.012213`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.002793`

### tick `211436`, seconds `73.50`, LSTM delta `+0.1594`

Top all feature movements:
- `lag_03__T_place_CONSTRUCTION`: contribution `+0.033520`
- `lag_13__T_place_PIPE`: contribution `+0.012259`
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.010997`
- `lag_13__T_place_CONSTRUCTION`: contribution `+0.009396`
- `lag_00__kill_diff_last_3s`: contribution `+0.006847`

Top utility-only movements:
- `lag_03__T3__flash_duration`: contribution `+0.002917`
- `lag_04__T4__flash_duration`: contribution `+0.002395`
- `lag_03__T2__flash_duration`: contribution `+0.002298`
- `lag_03__T_flash_duration_sum`: contribution `+0.002212`

### tick `212620`, seconds `92.00`, LSTM delta `-0.1037`

Top all feature movements:
- `lag_00__T_duck_amount_mean`: contribution `+0.011109`
- `lag_00__T_kills_last_3s`: contribution `-0.010722`
- `lag_02__CT_place_BRIDGE`: contribution `-0.010109`
- `lag_01__CT_place_BRIDGE`: contribution `-0.008867`
- `lag_04__T3__flash_duration`: contribution `-0.007127`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `-0.007127`

### tick `211692`, seconds `77.50`, LSTM delta `+0.0760`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `+0.010109`
- `lag_11__T_place_CONSTRUCTION`: contribution `-0.009964`
- `lag_01__CT_place_CANAL`: contribution `-0.009540`
- `lag_00__T_shots_fired_sum`: contribution `+0.009248`
- `lag_03__CT_place_RESTROOM`: contribution `+0.007212`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `+0.001832`

### tick `211468`, seconds `74.00`, LSTM delta `+0.0557`

Top all feature movements:
- `lag_08__T_place_CONSTRUCTION`: contribution `+0.010997`
- `lag_04__T_place_CONSTRUCTION`: contribution `-0.009099`
- `lag_14__T_place_CONSTRUCTION`: contribution `+0.007109`
- `lag_00__kill_diff_last_3s`: contribution `+0.006847`
- `lag_07__CT3__is_walking`: contribution `-0.005495`

Top utility-only movements:
- `lag_04__T3__flash_duration`: contribution `+0.005040`
- `lag_04__T_flash_duration_sum`: contribution `+0.002901`
- `lag_05__CT5__flash_duration`: contribution `+0.002114`
- `lag_05__T4__flash_duration`: contribution `+0.001791`
