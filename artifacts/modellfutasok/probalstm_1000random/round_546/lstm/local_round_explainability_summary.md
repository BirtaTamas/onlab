# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `22`

## Largest probability jumps

- tick `185070`, seconds `113.50`, LSTM `0.5508`, delta `+0.4708`
- tick `182926`, seconds `80.00`, LSTM `0.6035`, delta `+0.4180`
- tick `179918`, seconds `33.00`, LSTM `0.3907`, delta `-0.2994`
- tick `178830`, seconds `16.00`, LSTM `0.7086`, delta `+0.2988`
- tick `185294`, seconds `117.00`, LSTM `0.8212`, delta `+0.2985`
- tick `184558`, seconds `105.50`, LSTM `0.3471`, delta `-0.2848`
- tick `183022`, seconds `81.50`, LSTM `0.8973`, delta `+0.1783`
- tick `183598`, seconds `90.50`, LSTM `0.7686`, delta `-0.1668`
- tick `179950`, seconds `33.50`, LSTM `0.2383`, delta `-0.1525`
- tick `178734`, seconds `14.50`, LSTM `0.5579`, delta `+0.1173`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.012513`, |coef| `0.012513`
- `lag_00__kill_diff_last_3s`: coefficient `0.011943`, |coef| `0.011943`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.011401`, |coef| `0.011401`
- `lag_00__CT_defusing_count`: coefficient `0.011386`, |coef| `0.011386`
- `lag_00__CT_kills_last_3s`: coefficient `0.009795`, |coef| `0.009795`
- `lag_00__CT_damage_last_5s`: coefficient `0.007680`, |coef| `0.007680`
- `lag_07__T_flash_alpha_mean`: coefficient `-0.007112`, |coef| `0.007112`
- `lag_09__CT_place_CONNECTOR`: coefficient `-0.005839`, |coef| `0.005839`
- `lag_13__T_velocity_mean`: coefficient `-0.005756`, |coef| `0.005756`
- `lag_13__T_place_IVY`: coefficient `-0.005725`, |coef| `0.005725`
- `lag_00__CT5__is_scoped`: coefficient `-0.005686`, |coef| `0.005686`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.005660`, |coef| `0.005660`
- `lag_00__T_macro_A`: coefficient `-0.005660`, |coef| `0.005660`
- `lag_00__hp_diff`: coefficient `0.005188`, |coef| `0.005188`
- `lag_15__CT_duck_amount_mean`: coefficient `-0.005027`, |coef| `0.005027`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.011401` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.007112` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.002925` (lowers CT win probability)
- `lag_00__T5__flash`: coefficient `-0.002735` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.002603` (raises CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.002563` (raises CT win probability)
- `lag_00__CT4__molly`: coefficient `0.002393` (raises CT win probability)
- `lag_00__utility_inv_diff`: coefficient `0.002325` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.002197` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.002121` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.012513` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.011943` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.011386` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.009795` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.007680` (raises CT win probability)
- `lag_09__CT_place_CONNECTOR`: coefficient `-0.005839` (lowers CT win probability)
- `lag_13__T_velocity_mean`: coefficient `-0.005756` (lowers CT win probability)
- `lag_13__T_place_IVY`: coefficient `-0.005725` (lowers CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `-0.005686` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.005660` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `185070`, seconds `113.50`, LSTM delta `+0.4708`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.069170`
- `lag_15__CT_duck_amount_mean`: contribution `+0.030107`
- `lag_00__kill_diff_last_3s`: contribution `+0.028747`
- `lag_00__CT_kills_last_3s`: contribution `+0.028278`
- `lag_00__damage_diff_last_5s`: contribution `+0.028228`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.069170`

### tick `182926`, seconds `80.00`, LSTM delta `+0.4180`

Top all feature movements:
- `lag_13__T_place_IVY`: contribution `+0.030590`
- `lag_00__kill_diff_last_3s`: contribution `+0.028747`
- `lag_00__CT_kills_last_3s`: contribution `+0.028278`
- `lag_00__damage_diff_last_5s`: contribution `+0.028228`
- `lag_00__CT_damage_last_5s`: contribution `+0.016741`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `179918`, seconds `33.00`, LSTM delta `-0.2994`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.028747`
- `lag_00__damage_diff_last_5s`: contribution `-0.028228`
- `lag_11__T_place_IVY`: contribution `-0.021574`
- `lag_00__T5__is_scoped`: contribution `-0.020665`
- `lag_00__T_kills_last_3s`: contribution `-0.015753`

Top utility-only movements:
- `lag_00__CT4__utility_total`: contribution `-0.007263`
- `lag_00__CT4__molly`: contribution `-0.005895`

### tick `178830`, seconds `16.00`, LSTM delta `+0.2988`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.028747`
- `lag_00__CT_kills_last_3s`: contribution `+0.028278`
- `lag_00__damage_diff_last_5s`: contribution `+0.024276`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.024125`
- `lag_09__CT_place_ELECTRICALBOX`: contribution `+0.021745`

Top utility-only movements:
- `lag_10__T_A_site_active_infernos`: contribution `+0.004602`

### tick `185294`, seconds `117.00`, LSTM delta `+0.2985`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.110375`
- `lag_07__T_flash_alpha_mean`: contribution `+0.043153`
- `lag_00__CT_velocity_mean`: contribution `+0.015888`
- `lag_13__T_velocity_mean`: contribution `+0.011291`
- `lag_01__CT_velocity_mean`: contribution `+0.008736`

Top utility-only movements:
- `lag_07__T_flash_alpha_mean`: contribution `+0.043153`
