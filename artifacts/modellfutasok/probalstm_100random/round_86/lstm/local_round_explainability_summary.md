# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `121513`, seconds `86.50`, LSTM `0.8059`, delta `-0.1220`
- tick `121577`, seconds `87.50`, LSTM `0.9506`, delta `+0.1138`
- tick `119497`, seconds `55.00`, LSTM `0.9165`, delta `+0.0884`
- tick `116777`, seconds `12.50`, LSTM `0.8369`, delta `+0.0441`
- tick `120745`, seconds `74.50`, LSTM `0.9295`, delta `+0.0431`
- tick `121385`, seconds `84.50`, LSTM `0.8957`, delta `-0.0429`
- tick `121545`, seconds `87.00`, LSTM `0.8367`, delta `+0.0308`
- tick `117353`, seconds `21.50`, LSTM `0.8611`, delta `+0.0303`
- tick `119337`, seconds `52.50`, LSTM `0.8326`, delta `+0.0302`
- tick `117737`, seconds `27.50`, LSTM `0.8217`, delta `-0.0298`

## Top 15 local ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.001086`, |coef| `0.001086`
- `lag_00__kill_diff_last_3s`: coefficient `0.000874`, |coef| `0.000874`
- `lag_00__CT_kills_last_3s`: coefficient `0.000847`, |coef| `0.000847`
- `lag_02__T_place_SCAFFOLDING`: coefficient `0.000842`, |coef| `0.000842`
- `lag_00__CT_walking_count`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_04__T_place_SCAFFOLDING`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_05__T_place_SCAFFOLDING`: coefficient `-0.000758`, |coef| `0.000758`
- `lag_00__CT5__is_walking`: coefficient `-0.000732`, |coef| `0.000732`
- `lag_06__CT5__duck_amount`: coefficient `0.000708`, |coef| `0.000708`
- `lag_01__CT_place_TRUCK`: coefficient `-0.000706`, |coef| `0.000706`
- `lag_01__T_place_SCAFFOLDING`: coefficient `-0.000666`, |coef| `0.000666`
- `lag_03__T_place_STAIRS`: coefficient `-0.000665`, |coef| `0.000665`
- `lag_04__CT5__is_walking`: coefficient `0.000660`, |coef| `0.000660`
- `lag_07__CT1__duck_amount`: coefficient `-0.000658`, |coef| `0.000658`
- `lag_04__T_place_SIDEALLEY`: coefficient `-0.000640`, |coef| `0.000640`

## Top 10 utility ridge features

- `lag_05__CT_utility_damage_last_5s`: coefficient `0.000355` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000318` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000308` (lowers CT win probability)
- `lag_07__CT4__molly`: coefficient `0.000304` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `0.000294` (raises CT win probability)
- `lag_08__T_A_site_active_smokes`: coefficient `0.000288` (raises CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.000259` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.000257` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000248` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000246` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT3__is_walking`: coefficient `-0.001086` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000874` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000847` (raises CT win probability)
- `lag_02__T_place_SCAFFOLDING`: coefficient `0.000842` (raises CT win probability)
- `lag_00__CT_walking_count`: coefficient `-0.000773` (lowers CT win probability)
- `lag_04__T_place_SCAFFOLDING`: coefficient `-0.000761` (lowers CT win probability)
- `lag_05__T_place_SCAFFOLDING`: coefficient `-0.000758` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000732` (lowers CT win probability)
- `lag_06__CT5__duck_amount`: coefficient `0.000708` (raises CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `-0.000706` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `121513`, seconds `86.50`, LSTM delta `-0.1220`

Top all feature movements:
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.028659`
- `lag_05__T_place_SCAFFOLDING`: contribution `-0.025822`
- `lag_03__T_place_STAIRS`: contribution `-0.012727`
- `lag_03__T_place_LADDER`: contribution `-0.006569`
- `lag_12__CT_place_LADDER`: contribution `-0.006431`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121577`, seconds `87.50`, LSTM delta `+0.1138`

Top all feature movements:
- `lag_04__T_place_SCAFFOLDING`: contribution `+0.025925`
- `lag_07__T_place_SCAFFOLDING`: contribution `+0.020767`
- `lag_00__T_place_STAIRS`: contribution `+0.010213`
- `lag_05__T_place_LADDER`: contribution `+0.008991`
- `lag_05__T_place_STAIRS`: contribution `+0.008426`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `119497`, seconds `55.00`, LSTM delta `+0.0884`

Top all feature movements:
- `lag_01__CT_place_TRUCK`: contribution `+0.004555`
- `lag_00__CT3__is_walking`: contribution `+0.002594`
- `lag_06__CT5__duck_amount`: contribution `+0.002538`
- `lag_07__CT1__duck_amount`: contribution `+0.002511`
- `lag_00__CT_kills_last_3s`: contribution `+0.002446`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116777`, seconds `12.50`, LSTM delta `+0.0441`

Top all feature movements:
- `lag_01__CT_place_TRUCK`: contribution `-0.004555`
- `lag_09__CT_place_SHOP`: contribution `+0.002351`
- `lag_14__CT_place_SHOP`: contribution `+0.002188`
- `lag_10__CT_place_SHOP`: contribution `+0.002022`
- `lag_00__CT_place_CATWALK`: contribution `+0.001957`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `+0.001094`
- `lag_07__CT4__molly`: contribution `+0.000749`

### tick `120745`, seconds `74.50`, LSTM delta `+0.0431`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.002446`
- `lag_00__T_place_CONNECTOR`: contribution `+0.002116`
- `lag_00__kill_diff_last_3s`: contribution `+0.002103`
- `lag_11__T_place_CONNECTOR`: contribution `+0.001984`
- `lag_04__CT5__duck_amount`: contribution `+0.001881`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.000743`
