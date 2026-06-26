# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `33513`, seconds `57.50`, LSTM `0.7142`, delta `+0.2882`
- tick `33417`, seconds `56.00`, LSTM `0.3748`, delta `+0.2184`
- tick `33993`, seconds `65.00`, LSTM `0.8299`, delta `+0.2145`
- tick `33193`, seconds `52.50`, LSTM `0.1203`, delta `-0.1609`
- tick `34633`, seconds `75.00`, LSTM `0.9537`, delta `+0.1426`
- tick `33001`, seconds `49.50`, LSTM `0.3409`, delta `-0.0974`
- tick `33545`, seconds `58.00`, LSTM `0.6199`, delta `-0.0942`
- tick `33961`, seconds `64.50`, LSTM `0.6154`, delta `+0.0709`
- tick `33129`, seconds `51.50`, LSTM `0.2628`, delta `+0.0648`
- tick `33033`, seconds `50.00`, LSTM `0.2785`, delta `-0.0624`

## Top 15 local ridge features

- `lag_12__T_shots_fired_sum`: coefficient `-0.003142`, |coef| `0.003142`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002696`, |coef| `0.002696`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002673`, |coef| `0.002673`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002591`, |coef| `0.002591`
- `lag_09__T_shots_fired_sum`: coefficient `-0.002483`, |coef| `0.002483`
- `lag_00__CT_kills_last_3s`: coefficient `0.002158`, |coef| `0.002158`
- `lag_00__kill_diff_last_3s`: coefficient `0.002112`, |coef| `0.002112`
- `lag_12__T4__shots_fired`: coefficient `-0.001985`, |coef| `0.001985`
- `lag_00__damage_diff_last_5s`: coefficient `0.001873`, |coef| `0.001873`
- `lag_00__T4__duck_amount`: coefficient `0.001700`, |coef| `0.001700`
- `lag_09__T4__shots_fired`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_06__T_shots_fired_sum`: coefficient `-0.001654`, |coef| `0.001654`
- `lag_00__CT_damage_last_5s`: coefficient `0.001583`, |coef| `0.001583`
- `lag_13__T_shots_fired_sum`: coefficient `0.001569`, |coef| `0.001569`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001664` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.001521` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.001510` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001500` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `0.001410` (raises CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.001066` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001058` (lowers CT win probability)
- `lag_15__T5__utility_total`: coefficient `-0.001037` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001018` (raises CT win probability)
- `lag_02__T_utility_damage_last_5s`: coefficient `-0.001014` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_shots_fired_sum`: coefficient `-0.003142` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002696` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002673` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002591` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `-0.002483` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002158` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002112` (raises CT win probability)
- `lag_12__T4__shots_fired`: coefficient `-0.001985` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001873` (raises CT win probability)
- `lag_00__T4__duck_amount`: coefficient `0.001700` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `33513`, seconds `57.50`, LSTM delta `+0.2882`

Top all feature movements:
- `lag_12__T_shots_fired_sum`: contribution `+0.068310`
- `lag_12__T4__shots_fired`: contribution `+0.035563`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.013159`
- `lag_07__T_utility_damage_last_5s`: contribution `+0.009056`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009001`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `+0.009056`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.004041`

### tick `33417`, seconds `56.00`, LSTM delta `+0.2184`

Top all feature movements:
- `lag_09__T_shots_fired_sum`: contribution `+0.053988`
- `lag_09__T4__shots_fired`: contribution `+0.029884`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.013159`
- `lag_13__T_shots_fired_sum`: contribution `+0.011760`
- `lag_04__T_utility_damage_last_5s`: contribution `+0.006347`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `+0.006347`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.003762`
- `lag_14__T_utility_damage_last_5s`: contribution `+0.003195`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.002588`

### tick `33993`, seconds `65.00`, LSTM delta `+0.2145`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.018002`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.013159`
- `lag_15__T_place_SIDEENTRANCE`: contribution `+0.007451`
- `lag_00__CT_kills_last_3s`: contribution `+0.006231`
- `lag_07__CT1__is_scoped`: contribution `+0.005396`

Top utility-only movements:
- `lag_15__T5__utility_total`: contribution `+0.003321`

### tick `33193`, seconds `52.50`, LSTM delta `-0.1609`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.012400`
- `lag_02__T4__shots_fired`: contribution `-0.011453`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.009056`
- `lag_05__CT1__is_scoped`: contribution `-0.006459`
- `lag_04__T1__is_scoped`: contribution `-0.005510`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.009056`
- `lag_07__utility_damage_diff_last_5s`: contribution `-0.004041`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.002851`

### tick `34633`, seconds `75.00`, LSTM delta `+0.1426`

Top all feature movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009160`
- `lag_11__CT1__flash_duration`: contribution `+0.008611`
- `lag_05__CT1__is_scoped`: contribution `+0.006459`
- `lag_00__CT_kills_last_3s`: contribution `+0.006231`
- `lag_00__kill_diff_last_3s`: contribution `+0.005084`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.009160`
- `lag_11__CT1__flash_duration`: contribution `+0.008611`
- `lag_04__T1__flash_duration`: contribution `+0.003651`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001912`
