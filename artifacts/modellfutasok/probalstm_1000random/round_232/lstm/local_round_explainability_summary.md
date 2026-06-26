# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `26`

## Largest probability jumps

- tick `229386`, seconds `28.50`, LSTM `0.5386`, delta `-0.1377`
- tick `231242`, seconds `57.50`, LSTM `0.9260`, delta `+0.1242`
- tick `229194`, seconds `25.50`, LSTM `0.5717`, delta `+0.1080`
- tick `229098`, seconds `24.00`, LSTM `0.4851`, delta `-0.1070`
- tick `229322`, seconds `27.50`, LSTM `0.6585`, delta `+0.0957`
- tick `230666`, seconds `48.50`, LSTM `0.5821`, delta `+0.0939`
- tick `230698`, seconds `49.00`, LSTM `0.6451`, delta `+0.0630`
- tick `229514`, seconds `30.50`, LSTM `0.4685`, delta `-0.0497`
- tick `232426`, seconds `76.00`, LSTM `0.9678`, delta `+0.0474`
- tick `228138`, seconds `9.00`, LSTM `0.6356`, delta `+0.0443`

## Top 15 local ridge features

- `lag_08__CT_place_STORAGEROOM`: coefficient `0.001815`, |coef| `0.001815`
- `lag_00__kill_diff_last_3s`: coefficient `0.001732`, |coef| `0.001732`
- `lag_00__CT_kills_last_3s`: coefficient `0.001468`, |coef| `0.001468`
- `lag_01__T_place_PIPE`: coefficient `-0.001150`, |coef| `0.001150`
- `lag_00__damage_diff_last_5s`: coefficient `0.001058`, |coef| `0.001058`
- `lag_00__T_place_PIPE`: coefficient `-0.001007`, |coef| `0.001007`
- `lag_11__T_flashes_last_5s`: coefficient `0.000987`, |coef| `0.000987`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.000986`, |coef| `0.000986`
- `lag_00__T_place_CONSTRUCTION`: coefficient `0.000935`, |coef| `0.000935`
- `lag_00__T3__duck_amount`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_11__CT_place_CONSTRUCTION`: coefficient `-0.000862`, |coef| `0.000862`
- `lag_11__T_place_WATER`: coefficient `0.000860`, |coef| `0.000860`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.000835`, |coef| `0.000835`
- `lag_00__CT_place_UPPERPARK`: coefficient `-0.000817`, |coef| `0.000817`
- `lag_03__T_place_CONNECTOR`: coefficient `-0.000813`, |coef| `0.000813`

## Top 10 utility ridge features

- `lag_11__T_flashes_last_5s`: coefficient `0.000987` (raises CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `-0.000779` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000658` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.000521` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.000511` (lowers CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.000497` (raises CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.000464` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.000446` (lowers CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.000443` (raises CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.000435` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_STORAGEROOM`: coefficient `0.001815` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001732` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001468` (raises CT win probability)
- `lag_01__T_place_PIPE`: coefficient `-0.001150` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001058` (raises CT win probability)
- `lag_00__T_place_PIPE`: coefficient `-0.001007` (lowers CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.000986` (lowers CT win probability)
- `lag_00__T_place_CONSTRUCTION`: coefficient `0.000935` (raises CT win probability)
- `lag_00__T3__duck_amount`: coefficient `-0.000874` (lowers CT win probability)
- `lag_11__CT_place_CONSTRUCTION`: coefficient `-0.000862` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `229386`, seconds `28.50`, LSTM delta `-0.1377`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.008338`
- `lag_03__T_place_PIPE`: contribution `-0.007101`
- `lag_00__CT_kills_last_3s`: contribution `-0.004238`
- `lag_04__T_place_FOUNTAIN`: contribution `-0.003298`
- `lag_00__T3__duck_amount`: contribution `-0.003295`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `231242`, seconds `57.50`, LSTM delta `+0.1242`

Top all feature movements:
- `lag_08__CT_place_STORAGEROOM`: contribution `+0.038827`
- `lag_00__CT_place_FOUNTAIN`: contribution `+0.007589`
- `lag_00__CT_place_UPPERPARK`: contribution `+0.005814`
- `lag_00__CT_kills_last_3s`: contribution `+0.004238`
- `lag_00__kill_diff_last_3s`: contribution `+0.004169`

Top utility-only movements:
- `lag_06__CT1__flash_duration`: contribution `+0.002471`
- `lag_00__T5__utility_total`: contribution `+0.001636`
- `lag_00__T5__flash`: contribution `+0.001267`

### tick `229194`, seconds `25.50`, LSTM delta `+0.1080`

Top all feature movements:
- `lag_14__T_flashes_last_5s`: contribution `+0.007060`
- `lag_14__CT_place_CONSTRUCTION`: contribution `+0.006172`
- `lag_00__T_place_CONNECTOR`: contribution `+0.004775`
- `lag_00__CT_kills_last_3s`: contribution `+0.004238`
- `lag_00__kill_diff_last_3s`: contribution `+0.004169`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `+0.007060`
- `lag_15__T_utility_damage_last_5s`: contribution `+0.002082`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.001611`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.001520`

### tick `229098`, seconds `24.00`, LSTM delta `-0.1070`

Top all feature movements:
- `lag_11__CT_place_CONSTRUCTION`: contribution `-0.010841`
- `lag_11__T_flashes_last_5s`: contribution `-0.008945`
- `lag_00__kill_diff_last_3s`: contribution `-0.004169`
- `lag_00__T3__duck_amount`: contribution `-0.003046`
- `lag_06__T_place_FOUNTAIN`: contribution `-0.002608`

Top utility-only movements:
- `lag_11__T_flashes_last_5s`: contribution `-0.008945`
- `lag_14__T5__flash_duration`: contribution `-0.002541`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.001993`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.001025`

### tick `229322`, seconds `27.50`, LSTM delta `+0.0957`

Top all feature movements:
- `lag_01__T_place_PIPE`: contribution `+0.014685`
- `lag_00__CT_kills_last_3s`: contribution `+0.004238`
- `lag_00__kill_diff_last_3s`: contribution `+0.004169`
- `lag_00__damage_diff_last_5s`: contribution `+0.002673`
- `lag_02__T_place_LOWERPARK`: contribution `+0.002469`

Top utility-only movements:
- No utility movement among the top local contributors.
