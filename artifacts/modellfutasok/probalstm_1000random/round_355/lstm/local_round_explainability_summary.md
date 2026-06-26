# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `102670`, seconds `37.50`, LSTM `0.1818`, delta `-0.3073`
- tick `102318`, seconds `32.00`, LSTM `0.3955`, delta `-0.1970`
- tick `103118`, seconds `44.50`, LSTM `0.0635`, delta `-0.1761`
- tick `102894`, seconds `41.00`, LSTM `0.1827`, delta `+0.1170`
- tick `102446`, seconds `34.00`, LSTM `0.3931`, delta `+0.1117`
- tick `102926`, seconds `41.50`, LSTM `0.2863`, delta `+0.1035`
- tick `102702`, seconds `38.00`, LSTM `0.1036`, delta `-0.0782`
- tick `102350`, seconds `32.50`, LSTM `0.3294`, delta `-0.0661`
- tick `102958`, seconds `42.00`, LSTM `0.2341`, delta `-0.0522`
- tick `102382`, seconds `33.00`, LSTM `0.2779`, delta `-0.0516`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002242`, |coef| `0.002242`
- `lag_00__kill_diff_last_3s`: coefficient `0.002019`, |coef| `0.002019`
- `lag_06__CT2__duck_amount`: coefficient `0.001794`, |coef| `0.001794`
- `lag_02__CT2__flash_duration`: coefficient `-0.001726`, |coef| `0.001726`
- `lag_00__CT_place_QUAD`: coefficient `0.001696`, |coef| `0.001696`
- `lag_00__damage_diff_last_5s`: coefficient `0.001691`, |coef| `0.001691`
- `lag_07__CT4__flash_duration`: coefficient `0.001642`, |coef| `0.001642`
- `lag_06__CT_place_QUAD`: coefficient `0.001559`, |coef| `0.001559`
- `lag_00__T_damage_last_5s`: coefficient `-0.001518`, |coef| `0.001518`
- `lag_09__CT4__flash_duration`: coefficient `-0.001507`, |coef| `0.001507`
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_03__CT_place_QUAD`: coefficient `-0.001487`, |coef| `0.001487`
- `lag_01__T5__duck_amount`: coefficient `0.001474`, |coef| `0.001474`
- `lag_00__CT1__flash_duration`: coefficient `0.001441`, |coef| `0.001441`
- `lag_01__T1__duck_amount`: coefficient `0.001419`, |coef| `0.001419`

## Top 10 utility ridge features

- `lag_02__CT2__flash_duration`: coefficient `-0.001726` (lowers CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001642` (raises CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `-0.001507` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `-0.001500` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.001441` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001355` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001288` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001078` (lowers CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.001046` (lowers CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001029` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002242` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002019` (raises CT win probability)
- `lag_06__CT2__duck_amount`: coefficient `0.001794` (raises CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.001696` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001691` (raises CT win probability)
- `lag_06__CT_place_QUAD`: coefficient `0.001559` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001518` (lowers CT win probability)
- `lag_03__CT_place_QUAD`: coefficient `-0.001487` (lowers CT win probability)
- `lag_01__T5__duck_amount`: coefficient `0.001474` (raises CT win probability)
- `lag_01__T1__duck_amount`: coefficient `0.001419` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `102670`, seconds `37.50`, LSTM delta `-0.3073`

Top all feature movements:
- `lag_00__CT_place_QUAD`: contribution `-0.013366`
- `lag_07__CT4__flash_duration`: contribution `-0.012436`
- `lag_03__CT_place_QUAD`: contribution `-0.011722`
- `lag_00__CT1__flash_duration`: contribution `-0.011320`
- `lag_02__CT1__flash_duration`: contribution `-0.010643`

Top utility-only movements:
- `lag_07__CT4__flash_duration`: contribution `-0.012436`
- `lag_00__CT1__flash_duration`: contribution `-0.011320`
- `lag_02__CT1__flash_duration`: contribution `-0.010643`
- `lag_02__CT2__flash_duration`: contribution `-0.010604`
- `lag_02__CT_flash_duration_sum`: contribution `-0.008137`

### tick `102318`, seconds `32.00`, LSTM delta `-0.1970`

Top all feature movements:
- `lag_06__CT_place_QUAD`: contribution `-0.012285`
- `lag_09__CT4__flash_duration`: contribution `-0.011413`
- `lag_00__T_kills_last_3s`: contribution `-0.007103`
- `lag_06__CT2__duck_amount`: contribution `-0.006834`
- `lag_15__CT1__flash_duration`: contribution `-0.006167`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.011413`
- `lag_15__CT1__flash_duration`: contribution `-0.006167`

### tick `103118`, seconds `44.50`, LSTM delta `-0.1761`

Top all feature movements:
- `lag_08__T_place_QUAD`: contribution `-0.015280`
- `lag_10__T_place_QUAD`: contribution `-0.014979`
- `lag_00__T_kills_last_3s`: contribution `-0.007103`
- `lag_06__CT2__duck_amount`: contribution `-0.006181`
- `lag_14__CT1__flash_duration`: contribution `-0.006027`

Top utility-only movements:
- `lag_14__CT1__flash_duration`: contribution `-0.006027`
- `lag_00__CT2__flash_duration`: contribution `-0.003064`

### tick `102894`, seconds `41.00`, LSTM delta `+0.1170`

Top all feature movements:
- `lag_01__T_place_QUAD`: contribution `+0.017430`
- `lag_03__T_place_QUAD`: contribution `+0.007782`
- `lag_01__T5__duck_amount`: contribution `+0.005597`
- `lag_14__CT4__flash_duration`: contribution `+0.005489`
- `lag_12__T1__is_scoped`: contribution `+0.005229`

Top utility-only movements:
- `lag_14__CT4__flash_duration`: contribution `+0.005489`
- `lag_09__CT1__flash_duration`: contribution `+0.004753`
- `lag_07__CT1__flash_duration`: contribution `+0.002878`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.002560`

### tick `102446`, seconds `34.00`, LSTM delta `+0.1117`

Top all feature movements:
- `lag_00__CT4__flash_duration`: contribution `+0.007766`
- `lag_01__T5__duck_amount`: contribution `+0.005597`
- `lag_11__CT_place_ARCH`: contribution `+0.004842`
- `lag_04__T1__is_scoped`: contribution `+0.004698`
- `lag_13__CT2__duck_amount`: contribution `+0.004452`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.007766`
- `lag_13__CT4__flash_duration`: contribution `+0.004437`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003237`
