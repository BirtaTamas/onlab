# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `129105`, seconds `58.00`, LSTM `0.5541`, delta `+0.3573`
- tick `132401`, seconds `109.50`, LSTM `0.3761`, delta `-0.2641`
- tick `129137`, seconds `58.50`, LSTM `0.8157`, delta `+0.2616`
- tick `129009`, seconds `56.50`, LSTM `0.2438`, delta `-0.2588`
- tick `129841`, seconds `69.50`, LSTM `0.6447`, delta `-0.1087`
- tick `126353`, seconds `15.00`, LSTM `0.6837`, delta `+0.0931`
- tick `132465`, seconds `110.50`, LSTM `0.2091`, delta `-0.0839`
- tick `132433`, seconds `110.00`, LSTM `0.2930`, delta `-0.0831`
- tick `128209`, seconds `44.00`, LSTM `0.6338`, delta `-0.0823`
- tick `132369`, seconds `109.00`, LSTM `0.6402`, delta `+0.0739`

## Top 15 local ridge features

- `lag_10__CT_place_ELECTRICALBOX`: coefficient `-0.006177`, |coef| `0.006177`
- `lag_00__kill_diff_last_3s`: coefficient `0.004728`, |coef| `0.004728`
- `lag_00__T_kills_last_3s`: coefficient `-0.004026`, |coef| `0.004026`
- `lag_00__damage_diff_last_5s`: coefficient `0.003640`, |coef| `0.003640`
- `lag_13__CT_place_ELECTRICALBOX`: coefficient `0.003363`, |coef| `0.003363`
- `lag_03__CT_place_LONGDOG`: coefficient `-0.003313`, |coef| `0.003313`
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.003214`, |coef| `0.003214`
- `lag_00__T_place_LONGDOG`: coefficient `-0.003166`, |coef| `0.003166`
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `-0.002675`, |coef| `0.002675`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002581`, |coef| `0.002581`
- `lag_01__kill_diff_last_3s`: coefficient `0.002542`, |coef| `0.002542`
- `lag_00__T_damage_last_5s`: coefficient `-0.002411`, |coef| `0.002411`
- `lag_12__T_place_IVY`: coefficient `-0.002298`, |coef| `0.002298`
- `lag_01__T_kills_last_3s`: coefficient `-0.002280`, |coef| `0.002280`
- `lag_00__CT4__alive`: coefficient `0.002271`, |coef| `0.002271`

## Top 10 utility ridge features

- `lag_03__T1__molly`: coefficient `0.001879` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001757` (lowers CT win probability)
- `lag_04__T1__molly`: coefficient `0.001491` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001370` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.001212` (lowers CT win probability)
- `lag_05__T1__molly`: coefficient `0.001164` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001139` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001105` (lowers CT win probability)
- `lag_02__T1__molly`: coefficient `0.001081` (raises CT win probability)
- `lag_01__T1__molly`: coefficient `0.000993` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_ELECTRICALBOX`: coefficient `-0.006177` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004728` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004026` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003640` (raises CT win probability)
- `lag_13__CT_place_ELECTRICALBOX`: coefficient `0.003363` (raises CT win probability)
- `lag_03__CT_place_LONGDOG`: coefficient `-0.003313` (lowers CT win probability)
- `lag_14__CT_place_ELECTRICALBOX`: coefficient `0.003214` (raises CT win probability)
- `lag_00__T_place_LONGDOG`: coefficient `-0.003166` (lowers CT win probability)
- `lag_11__CT_place_ELECTRICALBOX`: coefficient `-0.002675` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002581` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `129105`, seconds `58.00`, LSTM delta `+0.3573`

Top all feature movements:
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.071812`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `+0.039092`
- `lag_03__CT_place_LONGDOG`: contribution `+0.021613`
- `lag_00__T_place_LONGDOG`: contribution `+0.014732`
- `lag_09__CT_place_LONGDOG`: contribution `+0.014116`

Top utility-only movements:
- `lag_08__CT_B_site_active_infernos`: contribution `+0.003912`

### tick `132401`, seconds `109.50`, LSTM delta `-0.2641`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.012753`
- `lag_00__kill_diff_last_3s`: contribution `-0.011380`
- `lag_12__T_place_CONNECTOR`: contribution `-0.008895`
- `lag_02__T_bomb_zone_count`: contribution `-0.007960`
- `lag_08__T_bomb_zone_count`: contribution `-0.006293`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.005230`
- `lag_03__T1__molly`: contribution `-0.004162`

### tick `129137`, seconds `58.50`, LSTM delta `+0.2616`

Top all feature movements:
- `lag_14__CT_place_ELECTRICALBOX`: contribution `+0.037358`
- `lag_11__CT_place_ELECTRICALBOX`: contribution `+0.031102`
- `lag_00__T_place_LONGDOG`: contribution `+0.014732`
- `lag_04__CT_place_LONGDOG`: contribution `+0.011813`
- `lag_10__CT_place_LONGDOG`: contribution `+0.011612`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.003211`

### tick `129009`, seconds `56.50`, LSTM delta `-0.2588`

Top all feature movements:
- `lag_10__CT_place_ELECTRICALBOX`: contribution `-0.071812`
- `lag_02__CT_place_LONGDOG`: contribution `-0.014687`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `-0.014055`
- `lag_00__T_kills_last_3s`: contribution `-0.012753`
- `lag_00__kill_diff_last_3s`: contribution `-0.011380`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `129841`, seconds `69.50`, LSTM delta `-0.1087`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.012753`
- `lag_00__kill_diff_last_3s`: contribution `-0.011380`
- `lag_06__CT_place_BACKOFB`: contribution `-0.008915`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.008329`
- `lag_11__CT_place_LONGDOG`: contribution `-0.007449`

Top utility-only movements:
- No utility movement among the top local contributors.
