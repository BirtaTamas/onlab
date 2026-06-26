# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `44263`, seconds `101.00`, LSTM `0.2215`, delta `-0.4287`
- tick `45735`, seconds `124.00`, LSTM `0.2526`, delta `-0.3622`
- tick `44391`, seconds `103.00`, LSTM `0.5998`, delta `+0.2370`
- tick `41703`, seconds `61.00`, LSTM `0.7260`, delta `+0.1692`
- tick `39175`, seconds `21.50`, LSTM `0.5402`, delta `-0.1572`
- tick `45927`, seconds `127.00`, LSTM `0.0627`, delta `-0.1439`
- tick `38951`, seconds `18.00`, LSTM `0.6639`, delta `+0.1376`
- tick `44359`, seconds `102.50`, LSTM `0.3628`, delta `+0.1327`
- tick `45415`, seconds `119.00`, LSTM `0.6842`, delta `+0.0786`
- tick `41671`, seconds `60.50`, LSTM `0.5568`, delta `+0.0680`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.008089`, |coef| `0.008089`
- `lag_00__T_kills_last_3s`: coefficient `-0.007525`, |coef| `0.007525`
- `lag_00__damage_diff_last_5s`: coefficient `0.005343`, |coef| `0.005343`
- `lag_12__CT_place_CATWALK`: coefficient `0.004646`, |coef| `0.004646`
- `lag_00__CT_place_TRAMP`: coefficient `0.004632`, |coef| `0.004632`
- `lag_00__T_damage_last_5s`: coefficient `-0.004505`, |coef| `0.004505`
- `lag_02__T_shots_fired_sum`: coefficient `-0.004279`, |coef| `0.004279`
- `lag_01__T_shots_fired_sum`: coefficient `-0.004226`, |coef| `0.004226`
- `lag_11__T_bomb_zone_count`: coefficient `0.004107`, |coef| `0.004107`
- `lag_03__CT_place_UNDERPASS`: coefficient `0.003833`, |coef| `0.003833`
- `lag_00__CT_flash_alpha_mean`: coefficient `0.003739`, |coef| `0.003739`
- `lag_00__CT5__alive`: coefficient `0.003370`, |coef| `0.003370`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003303`, |coef| `0.003303`
- `lag_08__CT_place_PALACEINTERIOR`: coefficient `0.003140`, |coef| `0.003140`
- `lag_10__CT4__is_walking`: coefficient `0.003012`, |coef| `0.003012`

## Top 10 utility ridge features

- `lag_00__CT_flash_alpha_mean`: coefficient `0.003739` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.002485` (raises CT win probability)
- `lag_03__CT5__smoke`: coefficient `0.002336` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.002102` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001894` (lowers CT win probability)
- `lag_05__CT1__flash`: coefficient `0.001569` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.001522` (lowers CT win probability)
- `lag_06__CT4__flash`: coefficient `0.001504` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.001403` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.001395` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.008089` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.007525` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005343` (raises CT win probability)
- `lag_12__CT_place_CATWALK`: coefficient `0.004646` (raises CT win probability)
- `lag_00__CT_place_TRAMP`: coefficient `0.004632` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.004505` (lowers CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.004279` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.004226` (lowers CT win probability)
- `lag_11__T_bomb_zone_count`: coefficient `0.004107` (raises CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `0.003833` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `44263`, seconds `101.00`, LSTM delta `-0.4287`

Top all feature movements:
- `lag_00__CT_place_TRAMP`: contribution `-0.062400`
- `lag_11__T_bomb_zone_count`: contribution `-0.023909`
- `lag_00__T_kills_last_3s`: contribution `-0.023841`
- `lag_00__kill_diff_last_3s`: contribution `-0.019470`
- `lag_12__CT_place_CATWALK`: contribution `-0.018507`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.007397`

### tick `45735`, seconds `124.00`, LSTM delta `-0.3622`

Top all feature movements:
- `lag_12__CT_place_TRAMP`: contribution `-0.026224`
- `lag_00__T_kills_last_3s`: contribution `-0.023841`
- `lag_03__CT_place_UNDERPASS`: contribution `-0.022229`
- `lag_00__kill_diff_last_3s`: contribution `-0.019470`
- `lag_01__T_shots_fired_sum`: contribution `-0.015843`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44391`, seconds `103.00`, LSTM delta `+0.2370`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.019470`
- `lag_12__CT_place_CATWALK`: contribution `+0.018507`
- `lag_15__T_bomb_zone_count`: contribution `+0.013179`
- `lag_03__T_shots_fired_sum`: contribution `+0.012602`
- `lag_04__CT_place_TRAMP`: contribution `-0.012007`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41703`, seconds `61.00`, LSTM delta `+0.1692`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.019470`
- `lag_02__T_place_UNDERPASS`: contribution `+0.010692`
- `lag_10__CT_place_UNDERPASS`: contribution `+0.010087`
- `lag_00__CT_kills_last_3s`: contribution `+0.008214`
- `lag_09__CT3__duck_amount`: contribution `+0.007481`

Top utility-only movements:
- `lag_00__T2__flash`: contribution `+0.005577`
- `lag_06__T_B_site_active_infernos`: contribution `+0.003822`

### tick `39175`, seconds `21.50`, LSTM delta `-0.1572`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.023841`
- `lag_00__kill_diff_last_3s`: contribution `-0.019470`
- `lag_12__CT_place_CATWALK`: contribution `-0.018507`
- `lag_00__damage_diff_last_5s`: contribution `-0.008679`
- `lag_00__T_damage_last_5s`: contribution `-0.007777`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `-0.005931`
- `lag_00__CT3__smoke`: contribution `-0.004650`
- `lag_00__T_A_site_active_infernos`: contribution `+0.003879`
