# Local Round Explainability

- csv_path: `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv`
- round_num: `8`

## Largest probability jumps

- tick `54316`, seconds `96.00`, LSTM `0.8849`, delta `+0.3810`
- tick `51020`, seconds `44.50`, LSTM `0.3491`, delta `-0.2055`
- tick `52332`, seconds `65.00`, LSTM `0.4253`, delta `+0.2055`
- tick `52364`, seconds `65.50`, LSTM `0.6129`, delta `+0.1876`
- tick `51052`, seconds `45.00`, LSTM `0.2563`, delta `-0.0927`
- tick `53772`, seconds `87.50`, LSTM `0.5382`, delta `-0.0884`
- tick `53708`, seconds `86.50`, LSTM `0.6566`, delta `-0.0738`
- tick `49420`, seconds `19.50`, LSTM `0.5744`, delta `+0.0728`
- tick `49324`, seconds `18.00`, LSTM `0.5195`, delta `-0.0709`
- tick `51596`, seconds `53.50`, LSTM `0.2437`, delta `+0.0678`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005277`, |coef| `0.005277`
- `lag_00__CT_kills_last_3s`: coefficient `0.004124`, |coef| `0.004124`
- `lag_04__T_bomb_zone_count`: coefficient `-0.003542`, |coef| `0.003542`
- `lag_15__T_bomb_zone_count`: coefficient `0.003294`, |coef| `0.003294`
- `lag_00__T4__flash_duration`: coefficient `0.003175`, |coef| `0.003175`
- `lag_08__T3__flash_duration`: coefficient `0.003141`, |coef| `0.003141`
- `lag_00__T1__flash_duration`: coefficient `0.002760`, |coef| `0.002760`
- `lag_12__CT_place_MAINHALL`: coefficient `-0.002454`, |coef| `0.002454`
- `lag_00__T_kills_last_3s`: coefficient `-0.002420`, |coef| `0.002420`
- `lag_09__CT5__flash_duration`: coefficient `-0.002396`, |coef| `0.002396`
- `lag_02__T5__is_scoped`: coefficient `0.002348`, |coef| `0.002348`
- `lag_01__T_place_ALLEY`: coefficient `0.002334`, |coef| `0.002334`
- `lag_01__T5__is_scoped`: coefficient `0.002261`, |coef| `0.002261`
- `lag_09__T3__flash_duration`: coefficient `0.002255`, |coef| `0.002255`
- `lag_01__kill_diff_last_3s`: coefficient `0.002232`, |coef| `0.002232`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `0.003175` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.003141` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.002760` (raises CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `-0.002396` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `0.002255` (raises CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `0.001771` (raises CT win probability)
- `lag_10__CT2__molly`: coefficient `-0.001724` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `-0.001605` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001552` (raises CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001422` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005277` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004124` (raises CT win probability)
- `lag_04__T_bomb_zone_count`: coefficient `-0.003542` (lowers CT win probability)
- `lag_15__T_bomb_zone_count`: coefficient `0.003294` (raises CT win probability)
- `lag_12__CT_place_MAINHALL`: coefficient `-0.002454` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002420` (lowers CT win probability)
- `lag_02__T5__is_scoped`: coefficient `0.002348` (raises CT win probability)
- `lag_01__T_place_ALLEY`: coefficient `0.002334` (raises CT win probability)
- `lag_01__T5__is_scoped`: coefficient `0.002261` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.002232` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `54316`, seconds `96.00`, LSTM delta `+0.3810`

Top all feature movements:
- `lag_04__T_bomb_zone_count`: contribution `+0.020622`
- `lag_15__T_bomb_zone_count`: contribution `+0.019175`
- `lag_00__T1__flash_duration`: contribution `+0.017598`
- `lag_00__T4__flash_duration`: contribution `+0.017501`
- `lag_05__CT_shots_fired_sum`: contribution `+0.013922`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.017598`
- `lag_00__T4__flash_duration`: contribution `+0.017501`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.006085`

### tick `51020`, seconds `44.50`, LSTM delta `-0.2055`

Top all feature movements:
- `lag_08__T3__flash_duration`: contribution `-0.024744`
- `lag_00__kill_diff_last_3s`: contribution `-0.012701`
- `lag_11__T1__shots_fired`: contribution `-0.008716`
- `lag_00__T_kills_last_3s`: contribution `-0.007667`
- `lag_08__T_place_TSIDELOWER`: contribution `-0.005834`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.024744`
- `lag_08__T_flash_duration_sum`: contribution `-0.004035`

### tick `52332`, seconds `65.00`, LSTM delta `+0.2055`

Top all feature movements:
- `lag_09__CT5__flash_duration`: contribution `+0.012816`
- `lag_00__kill_diff_last_3s`: contribution `+0.012701`
- `lag_00__CT_kills_last_3s`: contribution `+0.011907`
- `lag_02__T5__is_scoped`: contribution `+0.011199`
- `lag_06__CT2__duck_amount`: contribution `+0.006723`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `+0.012816`
- `lag_00__T5__utility_total`: contribution `+0.004552`
- `lag_00__T5__flash`: contribution `+0.003578`
- `lag_02__T_B_site_active_infernos`: contribution `+0.003191`

### tick `52364`, seconds `65.50`, LSTM delta `+0.1876`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012701`
- `lag_00__CT_kills_last_3s`: contribution `+0.011907`
- `lag_10__CT5__flash_duration`: contribution `+0.008587`
- `lag_00__T_duck_amount_mean`: contribution `+0.006960`
- `lag_03__T4__duck_amount`: contribution `+0.006478`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `+0.008587`

### tick `51052`, seconds `45.00`, LSTM delta `-0.0927`

Top all feature movements:
- `lag_09__T3__flash_duration`: contribution `-0.017761`
- `lag_13__T_shots_fired_sum`: contribution `-0.005732`
- `lag_01__kill_diff_last_3s`: contribution `-0.005373`
- `lag_03__T3__duck_amount`: contribution `-0.005033`
- `lag_14__T_shots_fired_sum`: contribution `-0.004801`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `-0.017761`
- `lag_09__T_flash_duration_sum`: contribution `-0.003170`
