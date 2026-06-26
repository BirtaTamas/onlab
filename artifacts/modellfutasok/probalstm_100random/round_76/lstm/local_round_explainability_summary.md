# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `19`

## Largest probability jumps

- tick `150659`, seconds `81.00`, LSTM `0.1060`, delta `-0.2129`
- tick `150115`, seconds `72.50`, LSTM `0.3376`, delta `-0.0751`
- tick `150915`, seconds `85.00`, LSTM `0.0205`, delta `-0.0721`
- tick `150883`, seconds `84.50`, LSTM `0.0926`, delta `+0.0661`
- tick `150467`, seconds `78.00`, LSTM `0.2918`, delta `-0.0624`
- tick `150691`, seconds `81.50`, LSTM `0.0599`, delta `-0.0461`
- tick `150083`, seconds `72.00`, LSTM `0.4127`, delta `-0.0440`
- tick `150563`, seconds `79.50`, LSTM `0.3316`, delta `+0.0395`
- tick `150147`, seconds `73.00`, LSTM `0.3662`, delta `+0.0285`
- tick `150371`, seconds `76.50`, LSTM `0.3538`, delta `+0.0280`

## Top 15 local ridge features

- `lag_09__CT_place_PALACEALLEY`: coefficient `0.002262`, |coef| `0.002262`
- `lag_06__CT_place_TRAMP`: coefficient `0.001916`, |coef| `0.001916`
- `lag_15__CT_place_SHOP`: coefficient `-0.001432`, |coef| `0.001432`
- `lag_00__T_kills_last_3s`: coefficient `-0.001429`, |coef| `0.001429`
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `0.001232`, |coef| `0.001232`
- `lag_00__T_damage_last_5s`: coefficient `-0.001200`, |coef| `0.001200`
- `lag_12__T1__is_scoped`: coefficient `-0.001174`, |coef| `0.001174`
- `lag_05__CT_place_SHOP`: coefficient `-0.001146`, |coef| `0.001146`
- `lag_06__CT_place_SHOP`: coefficient `-0.001123`, |coef| `0.001123`
- `lag_09__CT_place_TRAMP`: coefficient `-0.001110`, |coef| `0.001110`
- `lag_00__kill_diff_last_3s`: coefficient `0.001080`, |coef| `0.001080`
- `lag_09__CT3__is_walking`: coefficient `0.001061`, |coef| `0.001061`
- `lag_12__CT1__is_scoped`: coefficient `-0.001044`, |coef| `0.001044`
- `lag_14__CT4__is_walking`: coefficient `0.001026`, |coef| `0.001026`
- `lag_10__T_B_site_active_infernos`: coefficient `-0.000963`, |coef| `0.000963`

## Top 10 utility ridge features

- `lag_10__T_B_site_active_infernos`: coefficient `-0.000963` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000869` (raises CT win probability)
- `lag_03__CT3__smoke`: coefficient `0.000770` (raises CT win probability)
- `lag_13__T5__molly`: coefficient `0.000741` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.000652` (raises CT win probability)
- `lag_10__T_active_infernos`: coefficient `-0.000645` (lowers CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000538` (raises CT win probability)
- `lag_03__CT3__utility_total`: coefficient `0.000487` (raises CT win probability)
- `lag_02__T_B_site_active_smokes`: coefficient `0.000436` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000430` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_PALACEALLEY`: coefficient `0.002262` (raises CT win probability)
- `lag_06__CT_place_TRAMP`: coefficient `0.001916` (raises CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `-0.001432` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001429` (lowers CT win probability)
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `0.001232` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001200` (lowers CT win probability)
- `lag_12__T1__is_scoped`: coefficient `-0.001174` (lowers CT win probability)
- `lag_05__CT_place_SHOP`: coefficient `-0.001146` (lowers CT win probability)
- `lag_06__CT_place_SHOP`: coefficient `-0.001123` (lowers CT win probability)
- `lag_09__CT_place_TRAMP`: coefficient `-0.001110` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `150659`, seconds `81.00`, LSTM delta `-0.2129`

Top all feature movements:
- `lag_09__CT_place_PALACEALLEY`: contribution `-0.034535`
- `lag_06__CT_place_TRAMP`: contribution `-0.025809`
- `lag_09__CT_place_TRAMP`: contribution `-0.014956`
- `lag_15__CT_place_SHOP`: contribution `-0.007184`
- `lag_12__T1__is_scoped`: contribution `-0.006709`

Top utility-only movements:
- `lag_10__T_B_site_active_infernos`: contribution `-0.002723`

### tick `150115`, seconds `72.50`, LSTM delta `-0.0751`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.004527`
- `lag_09__T1__is_scoped`: contribution `-0.003249`
- `lag_10__T_place_CONNECTOR`: contribution `-0.002908`
- `lag_00__T1__is_scoped`: contribution `-0.002855`
- `lag_04__CT5__duck_amount`: contribution `-0.002305`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `150915`, seconds `85.00`, LSTM delta `-0.0721`

Top all feature movements:
- `lag_14__CT_place_TRAMP`: contribution `-0.009095`
- `lag_06__CT_place_SHOP`: contribution `+0.005631`
- `lag_00__T_kills_last_3s`: contribution `-0.004527`
- `lag_02__T2__duck_amount`: contribution `-0.003643`
- `lag_09__T1__is_scoped`: contribution `-0.003249`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `150883`, seconds `84.50`, LSTM delta `+0.0661`

Top all feature movements:
- `lag_13__CT_place_TRAMP`: contribution `+0.005807`
- `lag_05__CT_place_SHOP`: contribution `+0.005746`
- `lag_07__CT_place_SHOP`: contribution `+0.004531`
- `lag_08__T2__duck_amount`: contribution `+0.003516`
- `lag_00__T_place_CONNECTOR`: contribution `+0.003059`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `+0.001167`

### tick `150467`, seconds `78.00`, LSTM delta `-0.0624`

Top all feature movements:
- `lag_00__CT_place_TRAMP`: contribution `-0.012865`
- `lag_03__CT_place_PALACEALLEY`: contribution `-0.012689`
- `lag_03__CT_place_TRAMP`: contribution `-0.007201`
- `lag_12__T1__is_scoped`: contribution `-0.006709`
- `lag_00__CT_place_SHOP`: contribution `+0.004640`

Top utility-only movements:
- No utility movement among the top local contributors.
