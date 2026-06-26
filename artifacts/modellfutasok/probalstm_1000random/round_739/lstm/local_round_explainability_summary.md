# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `10`

## Largest probability jumps

- tick `81593`, seconds `64.00`, LSTM `0.4911`, delta `+0.2857`
- tick `81273`, seconds `59.00`, LSTM `0.4711`, delta `-0.1808`
- tick `82105`, seconds `72.00`, LSTM `0.7778`, delta `+0.1639`
- tick `81305`, seconds `59.50`, LSTM `0.3718`, delta `-0.0993`
- tick `81337`, seconds `60.00`, LSTM `0.2915`, delta `-0.0803`
- tick `82041`, seconds `71.00`, LSTM `0.5622`, delta `+0.0799`
- tick `82937`, seconds `85.00`, LSTM `0.8866`, delta `+0.0798`
- tick `82073`, seconds `71.50`, LSTM `0.6139`, delta `+0.0517`
- tick `81465`, seconds `62.00`, LSTM `0.2063`, delta `+0.0510`
- tick `82137`, seconds `72.50`, LSTM `0.8282`, delta `+0.0505`

## Top 15 local ridge features

- `lag_15__CT_place_RESTROOM`: coefficient `-0.002492`, |coef| `0.002492`
- `lag_00__damage_diff_last_5s`: coefficient `0.002190`, |coef| `0.002190`
- `lag_13__CT_place_RESTROOM`: coefficient `-0.002000`, |coef| `0.002000`
- `lag_12__CT_place_RESTROOM`: coefficient `-0.001924`, |coef| `0.001924`
- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.001875`, |coef| `0.001875`
- `lag_14__CT_place_RESTROOM`: coefficient `-0.001630`, |coef| `0.001630`
- `lag_11__CT_place_RESTROOM`: coefficient `-0.001608`, |coef| `0.001608`
- `lag_00__CT_kills_last_3s`: coefficient `0.001543`, |coef| `0.001543`
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.001538`, |coef| `0.001538`
- `lag_00__kill_diff_last_3s`: coefficient `0.001520`, |coef| `0.001520`
- `lag_04__CT_place_UPPERPARK`: coefficient `0.001465`, |coef| `0.001465`
- `lag_02__CT_place_UPPERPARK`: coefficient `0.001366`, |coef| `0.001366`
- `lag_03__CT_flashed_players`: coefficient `0.001362`, |coef| `0.001362`
- `lag_08__CT_shots_fired_sum`: coefficient `-0.001354`, |coef| `0.001354`
- `lag_00__CT_damage_last_5s`: coefficient `0.001269`, |coef| `0.001269`

## Top 10 utility ridge features

- `lag_08__CT_utility_damage_last_5s`: coefficient `-0.001875` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `-0.001538` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.001205` (raises CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.001150` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `0.001058` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.001004` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `0.000958` (raises CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `-0.000903` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000889` (raises CT win probability)
- `lag_10__CT1__flash`: coefficient `-0.000848` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_RESTROOM`: coefficient `-0.002492` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002190` (raises CT win probability)
- `lag_13__CT_place_RESTROOM`: coefficient `-0.002000` (lowers CT win probability)
- `lag_12__CT_place_RESTROOM`: coefficient `-0.001924` (lowers CT win probability)
- `lag_14__CT_place_RESTROOM`: coefficient `-0.001630` (lowers CT win probability)
- `lag_11__CT_place_RESTROOM`: coefficient `-0.001608` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001543` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001520` (raises CT win probability)
- `lag_04__CT_place_UPPERPARK`: coefficient `0.001465` (raises CT win probability)
- `lag_02__CT_place_UPPERPARK`: coefficient `0.001366` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `81593`, seconds `64.00`, LSTM delta `+0.2857`

Top all feature movements:
- `lag_14__CT_place_BACKOFA`: contribution `+0.012232`
- `lag_00__damage_diff_last_5s`: contribution `+0.011561`
- `lag_00__T_flashes_last_5s`: contribution `+0.009590`
- `lag_03__CT_flashed_players`: contribution `+0.008950`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.008872`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `+0.009590`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.008872`
- `lag_03__CT_flash_duration_sum`: contribution `+0.006855`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.005971`
- `lag_03__CT4__flash_duration`: contribution `+0.005803`

### tick `81273`, seconds `59.00`, LSTM delta `-0.1808`

Top all feature movements:
- `lag_11__CT_place_RESTROOM`: contribution `-0.022922`
- `lag_06__CT_place_RESTROOM`: contribution `-0.016261`
- `lag_11__CT_place_UPPERPARK`: contribution `-0.009032`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.008872`
- `lag_12__T2__is_scoped`: contribution `-0.006033`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.008872`
- `lag_08__utility_damage_diff_last_5s`: contribution `-0.005971`
- `lag_00__CT1__flash`: contribution `-0.002554`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002055`

### tick `82105`, seconds `72.00`, LSTM delta `+0.1639`

Top all feature movements:
- `lag_15__CT_place_RESTROOM`: contribution `+0.035536`
- `lag_06__T_flashes_last_5s`: contribution `+0.010421`
- `lag_04__T_place_WATER`: contribution `+0.007005`
- `lag_02__CT5__flash_duration`: contribution `+0.006915`
- `lag_02__T4__flash_duration`: contribution `+0.005875`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.010421`
- `lag_02__CT5__flash_duration`: contribution `+0.006915`
- `lag_02__T4__flash_duration`: contribution `+0.005875`
- `lag_08__CT4__flash_duration`: contribution `+0.004717`
- `lag_00__T4__flash_duration`: contribution `+0.003507`

### tick `81305`, seconds `59.50`, LSTM delta `-0.0993`

Top all feature movements:
- `lag_12__CT_place_RESTROOM`: contribution `-0.027427`
- `lag_07__CT_place_RESTROOM`: contribution `-0.013441`
- `lag_12__CT_place_UPPERPARK`: contribution `-0.007414`
- `lag_11__T2__is_scoped`: contribution `+0.005540`
- `lag_12__T_place_ALLEY`: contribution `-0.004592`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `-0.004272`
- `lag_09__utility_damage_diff_last_5s`: contribution `-0.002861`
- `lag_01__CT1__flash`: contribution `-0.002189`

### tick `81337`, seconds `60.00`, LSTM delta `-0.0803`

Top all feature movements:
- `lag_13__CT_place_RESTROOM`: contribution `-0.028518`
- `lag_13__CT_place_UPPERPARK`: contribution `-0.006667`
- `lag_12__T2__is_scoped`: contribution `+0.006033`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004787`
- `lag_04__CT_place_STAIRS`: contribution `+0.004175`

Top utility-only movements:
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.002758`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.001836`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.001820`
