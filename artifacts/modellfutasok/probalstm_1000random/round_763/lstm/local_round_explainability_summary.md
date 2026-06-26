# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `9796`, seconds `33.50`, LSTM `0.1901`, delta `-0.0897`
- tick `12676`, seconds `78.50`, LSTM `0.0538`, delta `-0.0783`
- tick `9892`, seconds `35.00`, LSTM `0.1798`, delta `-0.0661`
- tick `9860`, seconds `34.50`, LSTM `0.2459`, delta `+0.0543`
- tick `9732`, seconds `32.50`, LSTM `0.2712`, delta `+0.0408`
- tick `8964`, seconds `20.50`, LSTM `0.1974`, delta `-0.0379`
- tick `9668`, seconds `31.50`, LSTM `0.2511`, delta `+0.0363`
- tick `11108`, seconds `54.00`, LSTM `0.2194`, delta `+0.0348`
- tick `11588`, seconds `61.50`, LSTM `0.1923`, delta `-0.0324`
- tick `11076`, seconds `53.50`, LSTM `0.1846`, delta `+0.0311`

## Top 15 local ridge features

- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.001515`, |coef| `0.001515`
- `lag_00__CT_place_TRUCK`: coefficient `0.001469`, |coef| `0.001469`
- `lag_02__T3__is_walking`: coefficient `0.001207`, |coef| `0.001207`
- `lag_00__CT1__is_walking`: coefficient `0.001176`, |coef| `0.001176`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001173`, |coef| `0.001173`
- `lag_00__damage_diff_last_5s`: coefficient `0.001129`, |coef| `0.001129`
- `lag_01__CT_place_SHOP`: coefficient `-0.001115`, |coef| `0.001115`
- `lag_00__CT5__is_walking`: coefficient `-0.001105`, |coef| `0.001105`
- `lag_05__T5__duck_amount`: coefficient `-0.001044`, |coef| `0.001044`
- `lag_00__T1__duck_amount`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_00__CT2__is_walking`: coefficient `-0.000965`, |coef| `0.000965`
- `lag_15__CT1__duck_amount`: coefficient `0.000936`, |coef| `0.000936`
- `lag_07__T3__duck_amount`: coefficient `0.000923`, |coef| `0.000923`
- `lag_00__CT2__is_scoped`: coefficient `0.000815`, |coef| `0.000815`
- `lag_00__CT_place_SNIPERSNEST`: coefficient `-0.000787`, |coef| `0.000787`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001173` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000742` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.000723` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.000609` (raises CT win probability)
- `lag_09__T_utility_damage_last_5s`: coefficient `-0.000602` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.000507` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000482` (lowers CT win probability)
- `lag_11__CT3__smoke`: coefficient `0.000482` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.000466` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000461` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.001515` (raises CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.001469` (raises CT win probability)
- `lag_02__T3__is_walking`: coefficient `0.001207` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.001176` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001129` (raises CT win probability)
- `lag_01__CT_place_SHOP`: coefficient `-0.001115` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.001105` (lowers CT win probability)
- `lag_05__T5__duck_amount`: coefficient `-0.001044` (lowers CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.000974` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.000965` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9796`, seconds `33.50`, LSTM delta `-0.0897`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.009474`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.007372`
- `lag_01__CT_place_SHOP`: contribution `-0.005593`
- `lag_02__CT_place_TRUCK`: contribution `-0.004933`
- `lag_15__CT1__duck_amount`: contribution `-0.003570`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.007372`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002948`
- `lag_09__T5__flash_duration`: contribution `-0.001538`

### tick `12676`, seconds `78.50`, LSTM delta `-0.0783`

Top all feature movements:
- `lag_15__CT_place_SNIPERSNEST`: contribution `-0.008112`
- `lag_05__T5__duck_amount`: contribution `-0.003964`
- `lag_01__CT2__is_scoped`: contribution `-0.003775`
- `lag_02__T3__is_walking`: contribution `-0.002803`
- `lag_00__CT1__is_walking`: contribution `-0.002746`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9892`, seconds `35.00`, LSTM delta `-0.0661`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `-0.009474`
- `lag_00__CT2__is_scoped`: contribution `-0.004989`
- `lag_02__CT_place_TRUCK`: contribution `-0.004933`
- `lag_14__CT_place_TRUCK`: contribution `-0.004074`
- `lag_01__CT2__is_scoped`: contribution `-0.003775`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9860`, seconds `34.50`, LSTM delta `+0.0543`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `+0.004989`
- `lag_02__CT_place_TRUCK`: contribution `+0.004933`
- `lag_05__T5__duck_amount`: contribution `+0.003964`
- `lag_10__CT_place_TRUCK`: contribution `+0.002890`
- `lag_08__CT_place_TRUCK`: contribution `+0.002801`

Top utility-only movements:
- `lag_02__T_utility_damage_last_5s`: contribution `+0.002485`

### tick `9732`, seconds `32.50`, LSTM delta `+0.0408`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `+0.009474`
- `lag_02__CT_place_TRUCK`: contribution `+0.004933`
- `lag_15__CT1__duck_amount`: contribution `+0.003570`
- `lag_06__T1__duck_amount`: contribution `+0.002878`
- `lag_08__CT_place_TRUCK`: contribution `+0.002801`

Top utility-only movements:
- No utility movement among the top local contributors.
