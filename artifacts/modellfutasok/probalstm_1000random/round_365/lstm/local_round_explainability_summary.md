# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `138017`, seconds `44.50`, LSTM `0.4856`, delta `+0.3669`
- tick `137889`, seconds `42.50`, LSTM `0.1559`, delta `-0.2492`
- tick `140993`, seconds `91.00`, LSTM `0.8134`, delta `+0.2064`
- tick `137217`, seconds `32.00`, LSTM `0.4041`, delta `-0.1691`
- tick `137537`, seconds `37.00`, LSTM `0.4306`, delta `+0.1265`
- tick `141089`, seconds `92.50`, LSTM `0.9584`, delta `+0.0853`
- tick `140801`, seconds `88.00`, LSTM `0.5230`, delta `+0.0758`
- tick `138337`, seconds `49.50`, LSTM `0.3548`, delta `-0.0634`
- tick `137313`, seconds `33.50`, LSTM `0.3258`, delta `-0.0558`
- tick `137857`, seconds `42.00`, LSTM `0.4051`, delta `-0.0533`

## Top 15 local ridge features

- `lag_13__CT_place_TRAMP`: coefficient `0.003935`, |coef| `0.003935`
- `lag_07__CT_place_TRAMP`: coefficient `-0.003654`, |coef| `0.003654`
- `lag_03__CT_place_TRAMP`: coefficient `0.002598`, |coef| `0.002598`
- `lag_00__kill_diff_last_3s`: coefficient `0.002495`, |coef| `0.002495`
- `lag_09__CT_place_TRAMP`: coefficient `-0.002421`, |coef| `0.002421`
- `lag_13__T_bomb_zone_count`: coefficient `0.002400`, |coef| `0.002400`
- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.002376`, |coef| `0.002376`
- `lag_02__CT_place_TRUCK`: coefficient `0.002342`, |coef| `0.002342`
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.002014`, |coef| `0.002014`
- `lag_00__CT_kills_last_3s`: coefficient `0.001895`, |coef| `0.001895`
- `lag_00__damage_diff_last_5s`: coefficient `0.001861`, |coef| `0.001861`
- `lag_14__CT_place_SHOP`: coefficient `0.001813`, |coef| `0.001813`
- `lag_01__T4__duck_amount`: coefficient `-0.001726`, |coef| `0.001726`
- `lag_06__T_bomb_zone_count`: coefficient `-0.001713`, |coef| `0.001713`
- `lag_12__T1__duck_amount`: coefficient `0.001637`, |coef| `0.001637`

## Top 10 utility ridge features

- `lag_11__CT_A_site_active_infernos`: coefficient `-0.001303` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.001233` (lowers CT win probability)
- `lag_11__CT_active_infernos`: coefficient `-0.001072` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001050` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.001008` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.000978` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000968` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000952` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000948` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `-0.000926` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_TRAMP`: coefficient `0.003935` (raises CT win probability)
- `lag_07__CT_place_TRAMP`: coefficient `-0.003654` (lowers CT win probability)
- `lag_03__CT_place_TRAMP`: coefficient `0.002598` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002495` (raises CT win probability)
- `lag_09__CT_place_TRAMP`: coefficient `-0.002421` (lowers CT win probability)
- `lag_13__T_bomb_zone_count`: coefficient `0.002400` (raises CT win probability)
- `lag_15__CT_place_SNIPERSNEST`: coefficient `0.002376` (raises CT win probability)
- `lag_02__CT_place_TRUCK`: coefficient `0.002342` (raises CT win probability)
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.002014` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001895` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `138017`, seconds `44.50`, LSTM delta `+0.3669`

Top all feature movements:
- `lag_13__CT_place_TRAMP`: contribution `+0.053014`
- `lag_07__CT_place_TRAMP`: contribution `+0.049227`
- `lag_14__CT_place_SHOP`: contribution `+0.009093`
- `lag_03__CT_place_PALACEINTERIOR`: contribution `+0.008207`
- `lag_15__T_place_CONNECTOR`: contribution `+0.007645`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `+0.004599`

### tick `137889`, seconds `42.50`, LSTM delta `-0.2492`

Top all feature movements:
- `lag_03__CT_place_TRAMP`: contribution `-0.035004`
- `lag_09__CT_place_TRAMP`: contribution `-0.032623`
- `lag_15__CT_place_SNIPERSNEST`: contribution `-0.012727`
- `lag_03__CT_place_PALACEINTERIOR`: contribution `-0.008207`
- `lag_00__kill_diff_last_3s`: contribution `-0.006006`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `-0.003360`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.002735`
- `lag_08__T_A_site_active_infernos`: contribution `-0.002211`

### tick `140993`, seconds `91.00`, LSTM delta `+0.2064`

Top all feature movements:
- `lag_02__CT_place_TRUCK`: contribution `+0.015105`
- `lag_13__T_bomb_zone_count`: contribution `+0.013972`
- `lag_06__T_bomb_zone_count`: contribution `+0.009970`
- `lag_12__T1__duck_amount`: contribution `+0.006411`
- `lag_02__CT_place_APARTMENTS`: contribution `+0.005965`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `+0.003368`
- `lag_06__utility_damage_diff_last_5s`: contribution `+0.002280`

### tick `137217`, seconds `32.00`, LSTM delta `-0.1691`

Top all feature movements:
- `lag_15__CT_place_SNIPERSNEST`: contribution `-0.012727`
- `lag_02__CT4__flash_duration`: contribution `-0.008540`
- `lag_04__CT_place_SNIPERSNEST`: contribution `-0.007629`
- `lag_08__CT_place_JUNGLE`: contribution `-0.006951`
- `lag_00__kill_diff_last_3s`: contribution `-0.006006`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.008540`
- `lag_00__CT4__flash_duration`: contribution `-0.002803`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.002157`

### tick `137537`, seconds `37.00`, LSTM delta `+0.1265`

Top all feature movements:
- `lag_04__CT_place_SNIPERSNEST`: contribution `+0.007629`
- `lag_07__CT_shots_fired_sum`: contribution `+0.007053`
- `lag_00__kill_diff_last_3s`: contribution `+0.006006`
- `lag_14__CT_place_SNIPERSNEST`: contribution `+0.005860`
- `lag_00__CT_kills_last_3s`: contribution `+0.005470`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.003360`
- `lag_12__CT4__flash_duration`: contribution `+0.002892`
- `lag_10__CT4__flash_duration`: contribution `+0.002872`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.002735`
- `lag_09__CT4__flash_duration`: contribution `+0.002719`
