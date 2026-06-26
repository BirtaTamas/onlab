# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `17`

## Largest probability jumps

- tick `141638`, seconds `69.50`, LSTM `0.6535`, delta `+0.4094`
- tick `141574`, seconds `68.50`, LSTM `0.2001`, delta `-0.2933`
- tick `140550`, seconds `52.50`, LSTM `0.8370`, delta `+0.2229`
- tick `142278`, seconds `79.50`, LSTM `0.8968`, delta `+0.1945`
- tick `140582`, seconds `53.00`, LSTM `0.6599`, delta `-0.1771`
- tick `139846`, seconds `41.50`, LSTM `0.5067`, delta `-0.1519`
- tick `139590`, seconds `37.50`, LSTM `0.6572`, delta `+0.1305`
- tick `140230`, seconds `47.50`, LSTM `0.5746`, delta `+0.0925`
- tick `141350`, seconds `65.00`, LSTM `0.5259`, delta `+0.0869`
- tick `141670`, seconds `70.00`, LSTM `0.5782`, delta `-0.0754`

## Top 15 local ridge features

- `lag_00__T2__is_scoped`: coefficient `-0.004940`, |coef| `0.004940`
- `lag_09__CT_place_OUTSIDETUNNEL`: coefficient `-0.004738`, |coef| `0.004738`
- `lag_00__kill_diff_last_3s`: coefficient `0.004102`, |coef| `0.004102`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003186`, |coef| `0.003186`
- `lag_00__CT_kills_last_3s`: coefficient `0.002918`, |coef| `0.002918`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002765`, |coef| `0.002765`
- `lag_07__CT_place_OUTSIDETUNNEL`: coefficient `0.002681`, |coef| `0.002681`
- `lag_05__T2__is_scoped`: coefficient `-0.002487`, |coef| `0.002487`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002468`, |coef| `0.002468`
- `lag_00__damage_diff_last_5s`: coefficient `0.002466`, |coef| `0.002466`
- `lag_06__T2__is_scoped`: coefficient `0.002317`, |coef| `0.002317`
- `lag_00__T_kills_last_3s`: coefficient `-0.002198`, |coef| `0.002198`
- `lag_04__T2__is_scoped`: coefficient `-0.002148`, |coef| `0.002148`
- `lag_10__T4__duck_amount`: coefficient `0.002125`, |coef| `0.002125`
- `lag_12__CT3__duck_amount`: coefficient `-0.002067`, |coef| `0.002067`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002765` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.001331` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.001179` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.001095` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001019` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `-0.000971` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000969` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `0.000966` (raises CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000935` (raises CT win probability)
- `lag_05__T2__molly`: coefficient `-0.000874` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T2__is_scoped`: coefficient `-0.004940` (lowers CT win probability)
- `lag_09__CT_place_OUTSIDETUNNEL`: coefficient `-0.004738` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004102` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003186` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002918` (raises CT win probability)
- `lag_07__CT_place_OUTSIDETUNNEL`: coefficient `0.002681` (raises CT win probability)
- `lag_05__T2__is_scoped`: coefficient `-0.002487` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002468` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002466` (raises CT win probability)
- `lag_06__T2__is_scoped`: coefficient `0.002317` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `141638`, seconds `69.50`, LSTM delta `+0.4094`

Top all feature movements:
- `lag_09__CT_place_OUTSIDETUNNEL`: contribution `+0.101929`
- `lag_00__T2__is_scoped`: contribution `+0.043541`
- `lag_05__T2__is_scoped`: contribution `+0.021919`
- `lag_06__T2__is_scoped`: contribution `+0.020419`
- `lag_09__CT_place_UPPERTUNNEL`: contribution `+0.014218`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141574`, seconds `68.50`, LSTM delta `-0.2933`

Top all feature movements:
- `lag_07__CT_place_OUTSIDETUNNEL`: contribution `-0.057687`
- `lag_00__T2__is_scoped`: contribution `-0.043541`
- `lag_04__T2__is_scoped`: contribution `-0.018931`
- `lag_03__T2__is_scoped`: contribution `-0.016303`
- `lag_07__CT_place_UPPERTUNNEL`: contribution `-0.014776`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140550`, seconds `52.50`, LSTM delta `+0.2229`

Top all feature movements:
- `lag_13__CT_place_OUTSIDELONG`: contribution `+0.018191`
- `lag_04__CT_place_OUTSIDELONG`: contribution `+0.014435`
- `lag_04__CT_place_TSPAWN`: contribution `+0.010521`
- `lag_00__kill_diff_last_3s`: contribution `+0.009873`
- `lag_11__CT_place_EXTENDEDA`: contribution `+0.008564`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.007876`
- `lag_02__T4__flash_duration`: contribution `+0.006059`
- `lag_03__T_B_site_active_infernos`: contribution `+0.003762`

### tick `142278`, seconds `79.50`, LSTM delta `+0.1945`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016776`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015495`
- `lag_13__T_duck_amount_mean`: contribution `+0.010900`
- `lag_14__T_duck_amount_mean`: contribution `+0.010416`
- `lag_00__kill_diff_last_3s`: contribution `+0.009873`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.016776`
- `lag_03__T_B_site_active_infernos`: contribution `+0.003762`
- `lag_00__T_utility_damage_last_5s`: contribution `+0.002482`

### tick `140582`, seconds `53.00`, LSTM delta `-0.1771`

Top all feature movements:
- `lag_05__CT_place_OUTSIDELONG`: contribution `-0.014582`
- `lag_00__kill_diff_last_3s`: contribution `-0.009873`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.008564`
- `lag_03__T4__flash_duration`: contribution `-0.007403`
- `lag_00__T_kills_last_3s`: contribution `-0.006962`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.007403`
- `lag_05__CT5__flash_duration`: contribution `-0.005968`
