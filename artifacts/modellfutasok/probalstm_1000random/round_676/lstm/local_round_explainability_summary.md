# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `70982`, seconds `12.00`, LSTM `0.6195`, delta `+0.2009`
- tick `72486`, seconds `35.50`, LSTM `0.9330`, delta `+0.1097`
- tick `72198`, seconds `31.00`, LSTM `0.7337`, delta `+0.0859`
- tick `70950`, seconds `11.50`, LSTM `0.4186`, delta `-0.0632`
- tick `70918`, seconds `11.00`, LSTM `0.4818`, delta `+0.0395`
- tick `72358`, seconds `33.50`, LSTM `0.7759`, delta `+0.0379`
- tick `71046`, seconds `13.00`, LSTM `0.5758`, delta `-0.0320`
- tick `72454`, seconds `35.00`, LSTM `0.8233`, delta `+0.0320`
- tick `73830`, seconds `56.50`, LSTM `0.9692`, delta `+0.0315`
- tick `71142`, seconds `14.50`, LSTM `0.5723`, delta `-0.0267`

## Top 15 local ridge features

- `lag_07__CT_place_RAFTERS`: coefficient `0.001419`, |coef| `0.001419`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001119`, |coef| `0.001119`
- `lag_00__CT_place_VENTS`: coefficient `0.001046`, |coef| `0.001046`
- `lag_00__CT_kills_last_3s`: coefficient `0.001037`, |coef| `0.001037`
- `lag_11__CT_place_ADMIN`: coefficient `0.000987`, |coef| `0.000987`
- `lag_09__CT_place_VENTS`: coefficient `0.000950`, |coef| `0.000950`
- `lag_07__CT_place_HEAVEN`: coefficient `-0.000914`, |coef| `0.000914`
- `lag_01__CT_place_CONTROL`: coefficient `0.000912`, |coef| `0.000912`
- `lag_00__T_place_SILO`: coefficient `-0.000904`, |coef| `0.000904`
- `lag_01__CT_place_HUT`: coefficient `0.000900`, |coef| `0.000900`
- `lag_00__kill_diff_last_3s`: coefficient `0.000865`, |coef| `0.000865`
- `lag_00__CT_damage_last_5s`: coefficient `0.000834`, |coef| `0.000834`
- `lag_07__CT_place_ADMIN`: coefficient `-0.000825`, |coef| `0.000825`
- `lag_02__T_place_TROPHY`: coefficient `0.000788`, |coef| `0.000788`
- `lag_00__T3__shots_fired`: coefficient `0.000786`, |coef| `0.000786`

## Top 10 utility ridge features

- `lag_04__CT1__flash_duration`: coefficient `0.000784` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000531` (raises CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000434` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000421` (lowers CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.000404` (raises CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000401` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `0.000391` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.000382` (raises CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000381` (lowers CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.000379` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_RAFTERS`: coefficient `0.001419` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001119` (raises CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `0.001046` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001037` (raises CT win probability)
- `lag_11__CT_place_ADMIN`: coefficient `0.000987` (raises CT win probability)
- `lag_09__CT_place_VENTS`: coefficient `0.000950` (raises CT win probability)
- `lag_07__CT_place_HEAVEN`: coefficient `-0.000914` (lowers CT win probability)
- `lag_01__CT_place_CONTROL`: coefficient `0.000912` (raises CT win probability)
- `lag_00__T_place_SILO`: coefficient `-0.000904` (lowers CT win probability)
- `lag_01__CT_place_HUT`: coefficient `0.000900` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `70982`, seconds `12.00`, LSTM delta `+0.2009`

Top all feature movements:
- `lag_07__CT_place_RAFTERS`: contribution `+0.015169`
- `lag_07__CT_place_HEAVEN`: contribution `+0.009871`
- `lag_01__CT_place_CONTROL`: contribution `+0.009468`
- `lag_01__CT_place_HUT`: contribution `+0.008781`
- `lag_14__CT_place_HELL`: contribution `+0.007312`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.005654`

### tick `72486`, seconds `35.50`, LSTM delta `+0.1097`

Top all feature movements:
- `lag_09__CT_place_VENTS`: contribution `+0.007968`
- `lag_07__CT_place_ADMIN`: contribution `+0.005734`
- `lag_09__T_place_SILO`: contribution `+0.005136`
- `lag_12__CT_place_ADMIN`: contribution `+0.004490`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003887`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `+0.002516`

### tick `72198`, seconds `31.00`, LSTM delta `+0.0859`

Top all feature movements:
- `lag_00__CT_place_VENTS`: contribution `+0.008773`
- `lag_00__T_place_SILO`: contribution `+0.006139`
- `lag_03__T_place_SECRET`: contribution `+0.003643`
- `lag_00__CT_kills_last_3s`: contribution `+0.002994`
- `lag_07__T1__is_scoped`: contribution `+0.002697`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `70950`, seconds `11.50`, LSTM delta `-0.0632`

Top all feature movements:
- `lag_06__CT_place_HEAVEN`: contribution `-0.008167`
- `lag_00__CT_place_HUT`: contribution `-0.006929`
- `lag_13__CT_place_HELL`: contribution `+0.006317`
- `lag_07__CT_place_ADMIN`: contribution `+0.005734`
- `lag_00__CT_place_CONTROL`: contribution `-0.004975`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.002613`

### tick `70918`, seconds `11.00`, LSTM delta `+0.0395`

Top all feature movements:
- `lag_09__CT_place_HELL`: contribution `-0.004481`
- `lag_05__CT_place_HEAVEN`: contribution `+0.003224`
- `lag_02__CT1__flash_duration`: contribution `+0.002752`
- `lag_00__T_place_TROPHY`: contribution `-0.002572`
- `lag_09__CT_place_ADMIN`: contribution `+0.002337`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.002752`
