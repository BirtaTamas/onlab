# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `7`

## Largest probability jumps

- tick `66043`, seconds `13.00`, LSTM `0.8178`, delta `+0.1263`
- tick `68571`, seconds `52.50`, LSTM `0.9231`, delta `+0.0886`
- tick `67355`, seconds `33.50`, LSTM `0.9122`, delta `+0.0788`
- tick `66587`, seconds `21.50`, LSTM `0.8141`, delta `+0.0509`
- tick `67963`, seconds `43.00`, LSTM `0.8978`, delta `-0.0464`
- tick `66011`, seconds `12.50`, LSTM `0.6915`, delta `+0.0428`
- tick `65915`, seconds `11.00`, LSTM `0.6806`, delta `-0.0422`
- tick `66331`, seconds `17.50`, LSTM `0.7868`, delta `-0.0372`
- tick `68379`, seconds `49.50`, LSTM `0.8681`, delta `+0.0362`
- tick `68443`, seconds `50.50`, LSTM `0.8327`, delta `-0.0336`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001192`, |coef| `0.001192`
- `lag_00__kill_diff_last_3s`: coefficient `0.001106`, |coef| `0.001106`
- `lag_15__CT_place_SHOP`: coefficient `0.000985`, |coef| `0.000985`
- `lag_00__T3__is_scoped`: coefficient `0.000980`, |coef| `0.000980`
- `lag_00__damage_diff_last_5s`: coefficient `0.000947`, |coef| `0.000947`
- `lag_00__CT_damage_last_5s`: coefficient `0.000798`, |coef| `0.000798`
- `lag_08__T1__is_walking`: coefficient `-0.000788`, |coef| `0.000788`
- `lag_12__CT_place_CONNECTOR`: coefficient `0.000779`, |coef| `0.000779`
- `lag_14__CT1__duck_amount`: coefficient `-0.000755`, |coef| `0.000755`
- `lag_10__CT1__duck_amount`: coefficient `-0.000732`, |coef| `0.000732`
- `lag_13__T1__duck_amount`: coefficient `-0.000716`, |coef| `0.000716`
- `lag_15__CT4__duck_amount`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_02__CT_shots_fired_sum`: coefficient `0.000706`, |coef| `0.000706`
- `lag_14__CT1__is_walking`: coefficient `0.000691`, |coef| `0.000691`
- `lag_00__CT4__is_walking`: coefficient `-0.000672`, |coef| `0.000672`

## Top 10 utility ridge features

- `lag_13__CT_utility_damage_last_5s`: coefficient `0.000622` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000554` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.000546` (raises CT win probability)
- `lag_13__utility_damage_diff_last_5s`: coefficient `0.000517` (raises CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.000455` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000455` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000454` (lowers CT win probability)
- `lag_04__CT_A_site_active_infernos`: coefficient `0.000451` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.000449` (lowers CT win probability)
- `lag_00__T_flash_inv`: coefficient `-0.000448` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001192` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001106` (raises CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `0.000985` (raises CT win probability)
- `lag_00__T3__is_scoped`: coefficient `0.000980` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000947` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000798` (raises CT win probability)
- `lag_08__T1__is_walking`: coefficient `-0.000788` (lowers CT win probability)
- `lag_12__CT_place_CONNECTOR`: coefficient `0.000779` (raises CT win probability)
- `lag_14__CT1__duck_amount`: coefficient `-0.000755` (lowers CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `-0.000732` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `66043`, seconds `13.00`, LSTM delta `+0.1263`

Top all feature movements:
- `lag_15__CT_place_SHOP`: contribution `+0.009886`
- `lag_00__T3__is_scoped`: contribution `-0.006284`
- `lag_08__CT_place_JUNGLE`: contribution `+0.004294`
- `lag_00__CT_kills_last_3s`: contribution `+0.003442`
- `lag_12__CT5__flash_duration`: contribution `+0.003319`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `+0.003319`
- `lag_01__CT5__flash_duration`: contribution `+0.002728`
- `lag_09__T3__flash_duration`: contribution `+0.001999`
- `lag_09__T4__flash_duration`: contribution `+0.001646`

### tick `68571`, seconds `52.50`, LSTM delta `+0.0886`

Top all feature movements:
- `lag_03__CT_place_TRUCK`: contribution `+0.004238`
- `lag_00__CT_kills_last_3s`: contribution `+0.003442`
- `lag_06__T3__is_scoped`: contribution `+0.003334`
- `lag_10__T3__is_scoped`: contribution `+0.002904`
- `lag_10__CT1__duck_amount`: contribution `+0.002794`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `67355`, seconds `33.50`, LSTM delta `+0.0788`

Top all feature movements:
- `lag_11__CT_place_UNDERPASS`: contribution `+0.003732`
- `lag_00__CT_kills_last_3s`: contribution `+0.003442`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.003288`
- `lag_14__CT1__duck_amount`: contribution `+0.002882`
- `lag_12__CT_place_CONNECTOR`: contribution `+0.002786`

Top utility-only movements:
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.003288`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.002352`
- `lag_13__utility_damage_diff_last_5s`: contribution `+0.002242`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001592`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.001552`

### tick `66587`, seconds `21.50`, LSTM delta `+0.0509`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `+0.006284`
- `lag_11__T3__is_scoped`: contribution `+0.003562`
- `lag_15__CT_place_JUNGLE`: contribution `+0.003439`
- `lag_08__T3__is_scoped`: contribution `+0.002572`
- `lag_14__CT_place_TRUCK`: contribution `+0.001935`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.000956`

### tick `67963`, seconds `43.00`, LSTM delta `-0.0464`

Top all feature movements:
- `lag_00__T3__is_scoped`: contribution `-0.006284`
- `lag_08__CT_place_JUNGLE`: contribution `-0.004294`
- `lag_00__kill_diff_last_3s`: contribution `-0.002662`
- `lag_00__damage_diff_last_5s`: contribution `-0.002137`
- `lag_15__T_place_PALACEALLEY`: contribution `-0.001410`

Top utility-only movements:
- `lag_12__CT_A_site_active_infernos`: contribution `-0.001001`
