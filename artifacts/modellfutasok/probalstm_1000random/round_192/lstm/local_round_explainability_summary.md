# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `9`

## Largest probability jumps

- tick `72001`, seconds `78.50`, LSTM `0.5632`, delta `-0.2802`
- tick `71681`, seconds `73.50`, LSTM `0.6021`, delta `-0.2513`
- tick `72513`, seconds `86.50`, LSTM `0.8118`, delta `+0.2360`
- tick `71873`, seconds `76.50`, LSTM `0.7548`, delta `+0.1793`
- tick `72481`, seconds `86.00`, LSTM `0.5758`, delta `+0.1575`
- tick `68321`, seconds `21.00`, LSTM `0.8201`, delta `+0.1484`
- tick `71713`, seconds `74.00`, LSTM `0.6911`, delta `+0.0890`
- tick `67809`, seconds `13.00`, LSTM `0.6954`, delta `-0.0864`
- tick `72321`, seconds `83.50`, LSTM `0.4446`, delta `+0.0745`
- tick `72129`, seconds `80.50`, LSTM `0.3462`, delta `-0.0738`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004042`, |coef| `0.004042`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003295`, |coef| `0.003295`
- `lag_00__damage_diff_last_5s`: coefficient `0.003015`, |coef| `0.003015`
- `lag_00__T_kills_last_3s`: coefficient `-0.002565`, |coef| `0.002565`
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `-0.002518`, |coef| `0.002518`
- `lag_00__CT_kills_last_3s`: coefficient `0.002511`, |coef| `0.002511`
- `lag_10__CT_place_ELECTRICALBOX`: coefficient `0.002275`, |coef| `0.002275`
- `lag_00__T2__duck_amount`: coefficient `-0.002241`, |coef| `0.002241`
- `lag_00__T4__is_walking`: coefficient `-0.002193`, |coef| `0.002193`
- `lag_00__T_damage_last_5s`: coefficient `-0.002132`, |coef| `0.002132`
- `lag_13__CT_place_ELECTRICALBOX`: coefficient `0.002069`, |coef| `0.002069`
- `lag_03__T2__is_walking`: coefficient `-0.001979`, |coef| `0.001979`
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001975`, |coef| `0.001975`
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001916`, |coef| `0.001916`
- `lag_01__T_B_site_active_infernos`: coefficient `-0.001878`, |coef| `0.001878`

## Top 10 utility ridge features

- `lag_01__T_A_site_active_infernos`: coefficient `-0.001975` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001916` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.001878` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001820` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.001774` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.001685` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001598` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.001519` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.001393` (lowers CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.001341` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004042` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003295` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003015` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002565` (lowers CT win probability)
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `-0.002518` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002511` (raises CT win probability)
- `lag_10__CT_place_ELECTRICALBOX`: coefficient `0.002275` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.002241` (lowers CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.002193` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002132` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72001`, seconds `78.50`, LSTM delta `-0.2802`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `-0.029944`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `-0.026441`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.024048`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `-0.010503`
- `lag_00__kill_diff_last_3s`: contribution `-0.009728`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.005704`
- `lag_00__T_B_site_active_infernos`: contribution `-0.005147`

### tick `71681`, seconds `73.50`, LSTM delta `-0.2513`

Top all feature movements:
- `lag_04__CT_place_ELECTRICALBOX`: contribution `-0.029270`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `-0.021413`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.011904`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.011305`
- `lag_00__kill_diff_last_3s`: contribution `-0.009728`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72513`, seconds `86.50`, LSTM delta `+0.2360`

Top all feature movements:
- `lag_06__T_place_ELECTRICALBOX`: contribution `+0.046424`
- `lag_00__T_bomb_zone_count`: contribution `+0.019180`
- `lag_00__kill_diff_last_3s`: contribution `+0.009728`
- `lag_07__T_A_site_active_infernos`: contribution `+0.009513`
- `lag_07__T_B_site_active_infernos`: contribution `+0.008589`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `+0.009513`
- `lag_07__T_B_site_active_infernos`: contribution `+0.008589`
- `lag_00__T_A_site_active_infernos`: contribution `+0.005704`
- `lag_00__T_B_site_active_infernos`: contribution `+0.005147`
- `lag_07__T_active_infernos`: contribution `+0.004683`

### tick `71873`, seconds `76.50`, LSTM delta `+0.1793`

Top all feature movements:
- `lag_00__T_place_ELECTRICALBOX`: contribution `+0.038452`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.026441`
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.018067`
- `lag_00__kill_diff_last_3s`: contribution `+0.009728`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.008293`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72481`, seconds `86.00`, LSTM delta `+0.1575`

Top all feature movements:
- `lag_05__T_place_ELECTRICALBOX`: contribution `+0.024347`
- `lag_06__T_A_site_active_infernos`: contribution `+0.010560`
- `lag_00__kill_diff_last_3s`: contribution `+0.009728`
- `lag_06__T_B_site_active_infernos`: contribution `+0.009526`
- `lag_00__CT_kills_last_3s`: contribution `+0.007248`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.010560`
- `lag_06__T_B_site_active_infernos`: contribution `+0.009526`
- `lag_06__T_active_infernos`: contribution `+0.005159`
- `lag_06__active_infernos_total`: contribution `+0.003715`
- `lag_15__T_A_site_active_infernos`: contribution `+0.002861`
