# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-spirit-vs-virtuspro-bo3-NVE3FTuEWJ64hP6AT-Vo9S/spirit-vs-virtus-pro-m2-overpass.csv`
- round_num: `11`

## Largest probability jumps

- tick `91045`, seconds `99.00`, LSTM `0.1942`, delta `-0.3306`
- tick `90469`, seconds `90.00`, LSTM `0.7832`, delta `+0.2364`
- tick `91621`, seconds `108.00`, LSTM `0.2505`, delta `+0.1893`
- tick `90245`, seconds `86.50`, LSTM `0.6257`, delta `+0.1764`
- tick `90757`, seconds `94.50`, LSTM `0.6562`, delta `-0.1440`
- tick `87653`, seconds `46.00`, LSTM `0.4666`, delta `-0.1240`
- tick `90149`, seconds `85.00`, LSTM `0.3761`, delta `+0.0789`
- tick `90853`, seconds `96.00`, LSTM `0.5562`, delta `-0.0623`
- tick `87429`, seconds `42.50`, LSTM `0.5735`, delta `-0.0619`
- tick `91525`, seconds `106.50`, LSTM `0.0223`, delta `-0.0614`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003487`, |coef| `0.003487`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003016`, |coef| `0.003016`
- `lag_00__CT_place_LOWERPARK`: coefficient `0.002860`, |coef| `0.002860`
- `lag_02__T_place_UPPERPARK`: coefficient `-0.002535`, |coef| `0.002535`
- `lag_06__CT5__flash_duration`: coefficient `0.002508`, |coef| `0.002508`
- `lag_03__CT1__flash_duration`: coefficient `0.002508`, |coef| `0.002508`
- `lag_09__CT_place_LOWERPARK`: coefficient `0.002507`, |coef| `0.002507`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002459`, |coef| `0.002459`
- `lag_03__T_place_UPPERPARK`: coefficient `-0.002369`, |coef| `0.002369`
- `lag_00__T_kills_last_3s`: coefficient `-0.002321`, |coef| `0.002321`
- `lag_15__CT1__flash_duration`: coefficient `-0.002226`, |coef| `0.002226`
- `lag_15__CT4__flash_duration`: coefficient `-0.002186`, |coef| `0.002186`
- `lag_00__CT_kills_last_3s`: coefficient `0.002067`, |coef| `0.002067`
- `lag_00__damage_diff_last_5s`: coefficient `0.002041`, |coef| `0.002041`
- `lag_08__CT_place_WATER`: coefficient `0.001982`, |coef| `0.001982`

## Top 10 utility ridge features

- `lag_06__CT5__flash_duration`: coefficient `0.002508` (raises CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `0.002508` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `-0.002226` (lowers CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.002186` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001689` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `-0.001515` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `0.001425` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.001415` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `0.001347` (raises CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.001255` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003487` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003016` (raises CT win probability)
- `lag_00__CT_place_LOWERPARK`: coefficient `0.002860` (raises CT win probability)
- `lag_02__T_place_UPPERPARK`: coefficient `-0.002535` (lowers CT win probability)
- `lag_09__CT_place_LOWERPARK`: coefficient `0.002507` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.002459` (raises CT win probability)
- `lag_03__T_place_UPPERPARK`: coefficient `-0.002369` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002321` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002067` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002041` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `91045`, seconds `99.00`, LSTM delta `-0.3306`

Top all feature movements:
- `lag_03__CT1__flash_duration`: contribution `-0.016218`
- `lag_06__CT5__flash_duration`: contribution `-0.015906`
- `lag_15__CT1__flash_duration`: contribution `-0.014396`
- `lag_00__CT_place_LOWERPARK`: contribution `-0.012776`
- `lag_08__CT_place_WATER`: contribution `-0.012043`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.016218`
- `lag_06__CT5__flash_duration`: contribution `-0.015906`
- `lag_15__CT1__flash_duration`: contribution `-0.014396`
- `lag_15__CT4__flash_duration`: contribution `-0.011154`
- `lag_15__CT_flash_duration_sum`: contribution `-0.008889`

### tick `90469`, seconds `90.00`, LSTM delta `+0.2364`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `+0.020180`
- `lag_05__CT5__shots_fired`: contribution `+0.012887`
- `lag_03__T_place_UPPERPARK`: contribution `+0.012490`
- `lag_00__CT5__flash_duration`: contribution `+0.010712`
- `lag_00__kill_diff_last_3s`: contribution `+0.008394`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `+0.010712`
- `lag_00__T5__flash_duration`: contribution `+0.007097`
- `lag_04__T_A_site_active_infernos`: contribution `+0.003228`
- `lag_00__CT_flash_duration_sum`: contribution `+0.003216`

### tick `91621`, seconds `108.00`, LSTM delta `+0.1893`

Top all feature movements:
- `lag_14__CT_place_BACKOFA`: contribution `+0.015932`
- `lag_14__CT_place_STAIRS`: contribution `+0.011476`
- `lag_10__CT_place_BACKOFA`: contribution `+0.011349`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010476`
- `lag_03__T2__is_scoped`: contribution `+0.008465`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `90245`, seconds `86.50`, LSTM delta `+0.1764`

Top all feature movements:
- `lag_10__T2__is_scoped`: contribution `+0.017385`
- `lag_02__T_place_UPPERPARK`: contribution `+0.013366`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010476`
- `lag_06__T_place_CONNECTOR`: contribution `+0.009141`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008543`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.003222`

### tick `90757`, seconds `94.50`, LSTM delta `-0.1440`

Top all feature movements:
- `lag_00__CT_place_LOWERPARK`: contribution `-0.012776`
- `lag_14__CT_shots_fired_sum`: contribution `-0.009198`
- `lag_00__kill_diff_last_3s`: contribution `-0.008394`
- `lag_09__CT_flashed_players`: contribution `-0.007898`
- `lag_00__T_kills_last_3s`: contribution `-0.007355`

Top utility-only movements:
- `lag_06__CT_flash_duration_sum`: contribution `+0.007037`
- `lag_06__CT4__flash_duration`: contribution `+0.006873`
- `lag_09__T5__flash_duration`: contribution `-0.005225`
- `lag_06__CT1__flash_duration`: contribution `-0.003717`
- `lag_09__CT5__flash_duration`: contribution `-0.003631`
