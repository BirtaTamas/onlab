# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `97398`, seconds `38.50`, LSTM `0.3946`, delta `-0.1698`
- tick `100246`, seconds `83.00`, LSTM `0.6449`, delta `-0.1602`
- tick `99478`, seconds `71.00`, LSTM `0.5067`, delta `-0.1581`
- tick `100054`, seconds `80.00`, LSTM `0.7314`, delta `+0.1462`
- tick `99446`, seconds `70.50`, LSTM `0.6648`, delta `+0.1212`
- tick `96502`, seconds `24.50`, LSTM `0.5960`, delta `-0.1159`
- tick `99766`, seconds `75.50`, LSTM `0.6534`, delta `+0.1159`
- tick `96630`, seconds `26.50`, LSTM `0.6792`, delta `+0.0979`
- tick `97430`, seconds `39.00`, LSTM `0.3075`, delta `-0.0870`
- tick `97462`, seconds `39.50`, LSTM `0.2241`, delta `-0.0835`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002241`, |coef| `0.002241`
- `lag_00__kill_diff_last_3s`: coefficient `0.002061`, |coef| `0.002061`
- `lag_13__T5__shots_fired`: coefficient `0.001991`, |coef| `0.001991`
- `lag_10__T5__flash_duration`: coefficient `0.001769`, |coef| `0.001769`
- `lag_00__T_kills_last_3s`: coefficient `-0.001577`, |coef| `0.001577`
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.001497`, |coef| `0.001497`
- `lag_13__T_shots_fired_sum`: coefficient `0.001497`, |coef| `0.001497`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001483`, |coef| `0.001483`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001418`, |coef| `0.001418`
- `lag_07__T5__shots_fired`: coefficient `-0.001417`, |coef| `0.001417`
- `lag_02__CT5__flash_duration`: coefficient `-0.001368`, |coef| `0.001368`
- `lag_13__CT_shots_fired_sum`: coefficient `0.001363`, |coef| `0.001363`
- `lag_08__CT_he_last_5s`: coefficient `-0.001350`, |coef| `0.001350`
- `lag_15__CT5__shots_fired`: coefficient `0.001296`, |coef| `0.001296`
- `lag_08__CT5__flash_duration`: coefficient `-0.001259`, |coef| `0.001259`

## Top 10 utility ridge features

- `lag_10__T5__flash_duration`: coefficient `0.001769` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `-0.001497` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001368` (lowers CT win probability)
- `lag_08__CT_he_last_5s`: coefficient `-0.001350` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `-0.001259` (lowers CT win probability)
- `lag_06__T_utility_damage_last_5s`: coefficient `-0.001160` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.001148` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.001147` (lowers CT win probability)
- `lag_02__CT_he_last_5s`: coefficient `-0.001143` (lowers CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `-0.001045` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002241` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002061` (raises CT win probability)
- `lag_13__T5__shots_fired`: coefficient `0.001991` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001577` (lowers CT win probability)
- `lag_13__T_shots_fired_sum`: coefficient `0.001497` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001483` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001418` (lowers CT win probability)
- `lag_07__T5__shots_fired`: coefficient `-0.001417` (lowers CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `0.001363` (raises CT win probability)
- `lag_15__CT5__shots_fired`: coefficient `0.001296` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `97398`, seconds `38.50`, LSTM delta `-0.1698`

Top all feature movements:
- `lag_05__T_utility_damage_last_5s`: contribution `-0.010476`
- `lag_15__CT5__shots_fired`: contribution `-0.010282`
- `lag_15__CT_shots_fired_sum`: contribution `-0.010236`
- `lag_06__CT5__flash_duration`: contribution `-0.008793`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007784`

Top utility-only movements:
- `lag_05__T_utility_damage_last_5s`: contribution `-0.010476`
- `lag_06__CT5__flash_duration`: contribution `-0.008793`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.003500`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.002423`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002361`

### tick `100246`, seconds `83.00`, LSTM delta `-0.1602`

Top all feature movements:
- `lag_13__T5__shots_fired`: contribution `-0.033049`
- `lag_13__T_shots_fired_sum`: contribution `-0.030310`
- `lag_13__CT_shots_fired_sum`: contribution `-0.013255`
- `lag_06__CT_defusing_count`: contribution `-0.010468`
- `lag_02__CT4__flash_duration`: contribution `-0.009055`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.009055`
- `lag_02__T5__flash_duration`: contribution `-0.005100`
- `lag_02__CT2__flash_duration`: contribution `-0.004965`

### tick `99478`, seconds `71.00`, LSTM delta `-0.1581`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.009341`
- `lag_00__T_kills_last_3s`: contribution `-0.004997`
- `lag_00__kill_diff_last_3s`: contribution `-0.004962`
- `lag_11__T5__flash_duration`: contribution `-0.004311`
- `lag_11__T_flashed_players`: contribution `-0.003711`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `-0.004311`
- `lag_01__T1__flash_duration`: contribution `-0.003588`
- `lag_05__T5__flash_duration`: contribution `-0.002788`
- `lag_11__T_flash_duration_sum`: contribution `-0.002554`
- `lag_11__T1__flash_duration`: contribution `-0.002192`

### tick `100054`, seconds `80.00`, LSTM delta `+0.1462`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.024396`
- `lag_07__T5__shots_fired`: contribution `+0.023514`
- `lag_10__T5__flash_duration`: contribution `+0.012955`
- `lag_07__CT_shots_fired_sum`: contribution `+0.008811`
- `lag_00__CT_defusing_count`: contribution `+0.008065`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.012955`
- `lag_10__CT4__flash_duration`: contribution `+0.005376`
- `lag_10__CT2__flash_duration`: contribution `+0.004106`
- `lag_10__CT_flash_duration_sum`: contribution `+0.004052`
- `lag_10__T_flash_duration_sum`: contribution `+0.002409`

### tick `99446`, seconds `70.50`, LSTM delta `+0.1212`

Top all feature movements:
- `lag_10__T5__flash_duration`: contribution `+0.011487`
- `lag_02__CT5__flash_duration`: contribution `+0.006542`
- `lag_00__kill_diff_last_3s`: contribution `+0.004962`
- `lag_10__T_flash_duration_sum`: contribution `+0.004068`
- `lag_02__CT_shots_fired_sum`: contribution `+0.003863`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `+0.011487`
- `lag_02__CT5__flash_duration`: contribution `+0.006542`
- `lag_10__T_flash_duration_sum`: contribution `+0.004068`
