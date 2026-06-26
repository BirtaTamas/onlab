# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `38417`, seconds `96.50`, LSTM `0.4462`, delta `+0.2619`
- tick `38481`, seconds `97.50`, LSTM `0.2238`, delta `-0.2424`
- tick `35601`, seconds `52.50`, LSTM `0.1782`, delta `-0.2289`
- tick `36241`, seconds `62.50`, LSTM `0.0643`, delta `-0.1413`
- tick `35345`, seconds `48.50`, LSTM `0.2950`, delta `+0.1400`
- tick `35313`, seconds `48.00`, LSTM `0.1550`, delta `+0.1187`
- tick `38385`, seconds `96.00`, LSTM `0.1844`, delta `+0.0999`
- tick `35377`, seconds `49.00`, LSTM `0.3853`, delta `+0.0903`
- tick `38833`, seconds `103.00`, LSTM `0.3616`, delta `+0.0842`
- tick `38513`, seconds `98.00`, LSTM `0.3061`, delta `+0.0823`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004279`, |coef| `0.004279`
- `lag_03__T_bomb_zone_count`: coefficient `0.004117`, |coef| `0.004117`
- `lag_02__CT_place_PALACEINTERIOR`: coefficient `-0.003428`, |coef| `0.003428`
- `lag_00__damage_diff_last_5s`: coefficient `0.003217`, |coef| `0.003217`
- `lag_01__T_bomb_zone_count`: coefficient `-0.003176`, |coef| `0.003176`
- `lag_06__T_duck_amount_mean`: coefficient `0.003148`, |coef| `0.003148`
- `lag_00__T_damage_last_5s`: coefficient `-0.002953`, |coef| `0.002953`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002905`, |coef| `0.002905`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002877`, |coef| `0.002877`
- `lag_00__CT_kills_last_3s`: coefficient `0.002801`, |coef| `0.002801`
- `lag_03__CT3__duck_amount`: coefficient `0.002642`, |coef| `0.002642`
- `lag_00__T_velocity_mean`: coefficient `-0.002628`, |coef| `0.002628`
- `lag_04__T_bomb_zone_count`: coefficient `0.002576`, |coef| `0.002576`
- `lag_00__T_kills_last_3s`: coefficient `-0.002558`, |coef| `0.002558`
- `lag_14__T3__duck_amount`: coefficient `0.002471`, |coef| `0.002471`

## Top 10 utility ridge features

- `lag_11__T_A_site_active_infernos`: coefficient `-0.002363` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001666` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `-0.001334` (lowers CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `0.001260` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001227` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.001111` (lowers CT win probability)
- `lag_11__active_infernos_total`: coefficient `-0.001083` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.001073` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000949` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `-0.000939` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004279` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `0.004117` (raises CT win probability)
- `lag_02__CT_place_PALACEINTERIOR`: coefficient `-0.003428` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003217` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `-0.003176` (lowers CT win probability)
- `lag_06__T_duck_amount_mean`: coefficient `0.003148` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002953` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.002905` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.002877` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002801` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `38417`, seconds `96.50`, LSTM delta `+0.2619`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `+0.018488`
- `lag_06__T_duck_amount_mean`: contribution `+0.018308`
- `lag_04__T_bomb_zone_count`: contribution `+0.014997`
- `lag_02__CT_place_PALACEINTERIOR`: contribution `+0.013969`
- `lag_00__kill_diff_last_3s`: contribution `+0.010298`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `+0.007032`

### tick `38481`, seconds `97.50`, LSTM delta `-0.2424`

Top all feature movements:
- `lag_03__T_bomb_zone_count`: contribution `-0.023966`
- `lag_00__T_duck_amount_mean`: contribution `-0.016893`
- `lag_00__T_velocity_mean`: contribution `-0.011625`
- `lag_08__T_duck_amount_mean`: contribution `-0.010798`
- `lag_00__kill_diff_last_3s`: contribution `-0.010298`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35601`, seconds `52.50`, LSTM delta `-0.2289`

Top all feature movements:
- `lag_14__CT_place_SCAFFOLDING`: contribution `-0.033949`
- `lag_08__CT_place_SCAFFOLDING`: contribution `-0.018093`
- `lag_09__CT_place_SCAFFOLDING`: contribution `-0.017299`
- `lag_00__T_duck_amount_mean`: contribution `+0.011262`
- `lag_00__kill_diff_last_3s`: contribution `-0.010298`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36241`, seconds `62.50`, LSTM delta `-0.1413`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.010298`
- `lag_00__CT_place_JUNGLE`: contribution `-0.010029`
- `lag_10__T_place_PALACEINTERIOR`: contribution `-0.008164`
- `lag_00__T_kills_last_3s`: contribution `-0.008104`
- `lag_00__T_damage_last_5s`: contribution `-0.007081`

Top utility-only movements:
- `lag_14__CT_A_site_active_infernos`: contribution `-0.004447`
- `lag_03__T_A_site_active_infernos`: contribution `-0.003651`

### tick `35345`, seconds `48.50`, LSTM delta `+0.1400`

Top all feature movements:
- `lag_06__CT_place_SCAFFOLDING`: contribution `+0.026600`
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.011286`
- `lag_01__CT_place_SCAFFOLDING`: contribution `+0.010556`
- `lag_01__CT_place_PALACEINTERIOR`: contribution `+0.008699`
- `lag_02__T1__duck_amount`: contribution `+0.008365`

Top utility-only movements:
- No utility movement among the top local contributors.
