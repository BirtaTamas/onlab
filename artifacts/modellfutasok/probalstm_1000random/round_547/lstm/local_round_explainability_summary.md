# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `17`

## Largest probability jumps

- tick `134210`, seconds `73.00`, LSTM `0.1621`, delta `-0.2435`
- tick `134306`, seconds `74.50`, LSTM `0.0801`, delta `-0.1920`
- tick `130434`, seconds `14.00`, LSTM `0.2471`, delta `-0.1356`
- tick `134274`, seconds `74.00`, LSTM `0.2721`, delta `+0.1001`
- tick `130530`, seconds `15.50`, LSTM `0.3549`, delta `+0.0873`
- tick `132226`, seconds `42.00`, LSTM `0.3692`, delta `+0.0559`
- tick `134338`, seconds `75.00`, LSTM `0.0258`, delta `-0.0543`
- tick `134082`, seconds `71.00`, LSTM `0.3527`, delta `-0.0506`
- tick `133922`, seconds `68.50`, LSTM `0.3464`, delta `-0.0483`
- tick `134114`, seconds `71.50`, LSTM `0.3958`, delta `+0.0431`

## Top 15 local ridge features

- `lag_01__T_shots_fired_sum`: coefficient `-0.001943`, |coef| `0.001943`
- `lag_02__CT_place_UNDERPASS`: coefficient `-0.001786`, |coef| `0.001786`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001761`, |coef| `0.001761`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001656`, |coef| `0.001656`
- `lag_00__CT1__is_walking`: coefficient `0.001604`, |coef| `0.001604`
- `lag_00__T_kills_last_3s`: coefficient `-0.001561`, |coef| `0.001561`
- `lag_04__CT_place_STAIRS`: coefficient `0.001511`, |coef| `0.001511`
- `lag_00__CT_place_UNDERPASS`: coefficient `0.001483`, |coef| `0.001483`
- `lag_11__T_place_CATWALK`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_08__T5__is_walking`: coefficient `-0.001230`, |coef| `0.001230`
- `lag_00__kill_diff_last_3s`: coefficient `0.001170`, |coef| `0.001170`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001107`, |coef| `0.001107`
- `lag_07__CT_place_STAIRS`: coefficient `0.001097`, |coef| `0.001097`
- `lag_08__CT_place_CATWALK`: coefficient `-0.001062`, |coef| `0.001062`
- `lag_14__T_he_last_5s`: coefficient `0.001003`, |coef| `0.001003`

## Top 10 utility ridge features

- `lag_14__T_he_last_5s`: coefficient `0.001003` (raises CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000979` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000746` (lowers CT win probability)
- `lag_12__T2__molly`: coefficient `0.000676` (raises CT win probability)
- `lag_09__T_active_infernos`: coefficient `-0.000653` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000628` (lowers CT win probability)
- `lag_08__CT1__smoke`: coefficient `0.000605` (raises CT win probability)
- `lag_04__T5__molly`: coefficient `0.000586` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000586` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000529` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_shots_fired_sum`: coefficient `-0.001943` (lowers CT win probability)
- `lag_02__CT_place_UNDERPASS`: coefficient `-0.001786` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.001761` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001656` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.001604` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001561` (lowers CT win probability)
- `lag_04__CT_place_STAIRS`: coefficient `0.001511` (raises CT win probability)
- `lag_00__CT_place_UNDERPASS`: coefficient `0.001483` (raises CT win probability)
- `lag_11__T_place_CATWALK`: coefficient `-0.001272` (lowers CT win probability)
- `lag_08__T5__is_walking`: coefficient `-0.001230` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `134210`, seconds `73.00`, LSTM delta `-0.2435`

Top all feature movements:
- `lag_04__CT_place_STAIRS`: contribution `-0.011759`
- `lag_00__CT_place_JUNGLE`: contribution `-0.011300`
- `lag_02__CT_place_UNDERPASS`: contribution `-0.010356`
- `lag_00__CT_place_UNDERPASS`: contribution `-0.008597`
- `lag_01__T_shots_fired_sum`: contribution `-0.007283`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `-0.002768`

### tick `134306`, seconds `74.50`, LSTM delta `-0.1920`

Top all feature movements:
- `lag_07__CT_place_STAIRS`: contribution `-0.008541`
- `lag_03__CT_place_JUNGLE`: contribution `-0.006150`
- `lag_01__CT_shots_fired_sum`: contribution `-0.006049`
- `lag_00__T_kills_last_3s`: contribution `-0.004944`
- `lag_01__T_shots_fired_sum`: contribution `-0.004370`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `130434`, seconds `14.00`, LSTM delta `-0.1356`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `-0.034563`
- `lag_14__T_he_last_5s`: contribution `-0.013089`
- `lag_02__CT_place_SCAFFOLDING`: contribution `-0.012450`
- `lag_01__CT_place_TRUCK`: contribution `-0.006143`
- `lag_09__T_flashes_last_5s`: contribution `-0.004610`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `-0.013089`
- `lag_09__T_flashes_last_5s`: contribution `-0.004610`
- `lag_08__CT4__flash_duration`: contribution `-0.001761`

### tick `134274`, seconds `74.00`, LSTM delta `+0.1001`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.011652`
- `lag_02__CT_place_UNDERPASS`: contribution `+0.010356`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004385`
- `lag_01__CT_shots_fired_sum`: contribution `-0.003361`
- `lag_06__CT_place_STAIRS`: contribution `+0.003240`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `130530`, seconds `15.50`, LSTM delta `+0.0873`

Top all feature movements:
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.034563`
- `lag_03__CT_place_SCAFFOLDING`: contribution `+0.004243`
- `lag_09__CT_place_TRUCK`: contribution `+0.003987`
- `lag_02__T_flashes_last_5s`: contribution `+0.003864`
- `lag_00__CT1__is_walking`: contribution `+0.003745`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.003864`
- `lag_12__T_flashes_last_5s`: contribution `+0.003618`
- `lag_01__CT4__flash_duration`: contribution `+0.001463`
- `lag_11__CT4__flash_duration`: contribution `+0.001421`
