# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `7`

## Largest probability jumps

- tick `47295`, seconds `73.50`, LSTM `0.8593`, delta `+0.2952`
- tick `47359`, seconds `74.50`, LSTM `0.6239`, delta `-0.2740`
- tick `47103`, seconds `70.50`, LSTM `0.6495`, delta `-0.2193`
- tick `47871`, seconds `82.50`, LSTM `0.4800`, delta `-0.2082`
- tick `47615`, seconds `78.50`, LSTM `0.6182`, delta `-0.1275`
- tick `46879`, seconds `67.00`, LSTM `0.9075`, delta `+0.1193`
- tick `47551`, seconds `77.50`, LSTM `0.7358`, delta `+0.1188`
- tick `48063`, seconds `85.50`, LSTM `0.2978`, delta `-0.0798`
- tick `47231`, seconds `72.50`, LSTM `0.5469`, delta `-0.0699`
- tick `48095`, seconds `86.00`, LSTM `0.2323`, delta `-0.0655`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004055`, |coef| `0.004055`
- `lag_00__T_kills_last_3s`: coefficient `-0.003323`, |coef| `0.003323`
- `lag_00__damage_diff_last_5s`: coefficient `0.002456`, |coef| `0.002456`
- `lag_08__T_place_ELECTRICALBOX`: coefficient `0.002213`, |coef| `0.002213`
- `lag_02__T_place_ELECTRICALBOX`: coefficient `-0.002022`, |coef| `0.002022`
- `lag_00__CT_place_LONGDOG`: coefficient `0.002020`, |coef| `0.002020`
- `lag_00__T2__duck_amount`: coefficient `-0.001988`, |coef| `0.001988`
- `lag_00__T_place_ELECTRICALBOX`: coefficient `0.001951`, |coef| `0.001951`
- `lag_07__T3__duck_amount`: coefficient `0.001851`, |coef| `0.001851`
- `lag_00__CT_kills_last_3s`: coefficient `0.001836`, |coef| `0.001836`
- `lag_12__T_kills_last_3s`: coefficient `0.001811`, |coef| `0.001811`
- `lag_10__CT_place_BACKOFB`: coefficient `-0.001762`, |coef| `0.001762`
- `lag_10__T_place_LONGDOG`: coefficient `0.001725`, |coef| `0.001725`
- `lag_05__T_place_ELECTRICALBOX`: coefficient `-0.001704`, |coef| `0.001704`
- `lag_04__CT4__is_walking`: coefficient `0.001662`, |coef| `0.001662`

## Top 10 utility ridge features

- `lag_13__CT_B_site_active_infernos`: coefficient `-0.001266` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.001202` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001012` (raises CT win probability)
- `lag_00__CT5__molly`: coefficient `0.000955` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.000926` (raises CT win probability)
- `lag_05__CT1__smoke`: coefficient `-0.000898` (lowers CT win probability)
- `lag_14__T_active_smokes`: coefficient `0.000858` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.000841` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000840` (raises CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000810` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004055` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003323` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002456` (raises CT win probability)
- `lag_08__T_place_ELECTRICALBOX`: coefficient `0.002213` (raises CT win probability)
- `lag_02__T_place_ELECTRICALBOX`: coefficient `-0.002022` (lowers CT win probability)
- `lag_00__CT_place_LONGDOG`: coefficient `0.002020` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001988` (lowers CT win probability)
- `lag_00__T_place_ELECTRICALBOX`: coefficient `0.001951` (raises CT win probability)
- `lag_07__T3__duck_amount`: coefficient `0.001851` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001836` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `47295`, seconds `73.50`, LSTM delta `+0.2952`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `+0.053069`
- `lag_11__T_place_ELECTRICALBOX`: contribution `+0.028606`
- `lag_00__kill_diff_last_3s`: contribution `+0.019522`
- `lag_00__T_kills_last_3s`: contribution `+0.010529`
- `lag_14__T_place_LONGDOG`: contribution `+0.007630`

Top utility-only movements:
- `lag_13__CT_B_site_active_infernos`: contribution `+0.004349`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.004241`

### tick `47359`, seconds `74.50`, LSTM delta `-0.2740`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `-0.016095`
- `lag_13__T_place_ELECTRICALBOX`: contribution `-0.014616`
- `lag_00__T_kills_last_3s`: contribution `-0.010529`
- `lag_10__CT_place_BACKOFB`: contribution `-0.010060`
- `lag_00__kill_diff_last_3s`: contribution `-0.009761`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47103`, seconds `70.50`, LSTM delta `-0.2193`

Top all feature movements:
- `lag_05__T_place_ELECTRICALBOX`: contribution `-0.044719`
- `lag_00__T_kills_last_3s`: contribution `-0.010529`
- `lag_00__kill_diff_last_3s`: contribution `-0.009761`
- `lag_10__T_place_LONGDOG`: contribution `-0.008028`
- `lag_15__CT4__duck_amount`: contribution `-0.006092`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002663`

### tick `47871`, seconds `82.50`, LSTM delta `-0.2082`

Top all feature movements:
- `lag_08__T_place_ELECTRICALBOX`: contribution `-0.058085`
- `lag_10__T_place_ELECTRICALBOX`: contribution `-0.034591`
- `lag_00__T_kills_last_3s`: contribution `-0.010529`
- `lag_00__kill_diff_last_3s`: contribution `-0.009761`
- `lag_12__CT_kills_last_3s`: contribution `-0.004760`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47615`, seconds `78.50`, LSTM delta `-0.1275`

Top all feature movements:
- `lag_02__T_place_ELECTRICALBOX`: contribution `-0.053069`
- `lag_00__T_place_ELECTRICALBOX`: contribution `-0.051213`
- `lag_12__T_place_ELECTRICALBOX`: contribution `+0.007993`
- `lag_09__CT_place_LONGDOG`: contribution `+0.004509`
- `lag_00__damage_diff_last_5s`: contribution `-0.003767`

Top utility-only movements:
- No utility movement among the top local contributors.
