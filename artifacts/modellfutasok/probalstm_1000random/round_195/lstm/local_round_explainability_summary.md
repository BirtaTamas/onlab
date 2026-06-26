# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-vitality-bo5-g3-5jFl1QSVPqll-eeCKIE/mouz-vs-vitality-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `39735`, seconds `84.00`, LSTM `0.2556`, delta `-0.2192`
- tick `35799`, seconds `22.50`, LSTM `0.5701`, delta `-0.1857`
- tick `35735`, seconds `21.50`, LSTM `0.7806`, delta `+0.1466`
- tick `39799`, seconds `85.00`, LSTM `0.1071`, delta `-0.0968`
- tick `35639`, seconds `20.00`, LSTM `0.5982`, delta `+0.0758`
- tick `39703`, seconds `83.50`, LSTM `0.4748`, delta `-0.0664`
- tick `39767`, seconds `84.50`, LSTM `0.2039`, delta `-0.0517`
- tick `36503`, seconds `33.50`, LSTM `0.6092`, delta `+0.0464`
- tick `35959`, seconds `25.00`, LSTM `0.5459`, delta `-0.0457`
- tick `35831`, seconds `23.00`, LSTM `0.6021`, delta `+0.0320`

## Top 15 local ridge features

- `lag_08__T_place_PIT`: coefficient `-0.002475`, |coef| `0.002475`
- `lag_02__CT_place_LIBRARY`: coefficient `0.001659`, |coef| `0.001659`
- `lag_07__T_place_PIT`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_06__T_place_PIT`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_12__CT_place_LIBRARY`: coefficient `-0.001275`, |coef| `0.001275`
- `lag_00__CT5__shots_fired`: coefficient `0.001271`, |coef| `0.001271`
- `lag_00__T_kills_last_3s`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_15__CT2__duck_amount`: coefficient `-0.001241`, |coef| `0.001241`
- `lag_00__kill_diff_last_3s`: coefficient `0.001232`, |coef| `0.001232`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001226`, |coef| `0.001226`
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.001212`, |coef| `0.001212`
- `lag_00__CT_place_ARCH`: coefficient `0.001157`, |coef| `0.001157`
- `lag_00__CT5__flash_duration`: coefficient `0.001154`, |coef| `0.001154`
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_00__CT1__alive`: coefficient `0.001094`, |coef| `0.001094`

## Top 10 utility ridge features

- `lag_03__CT_utility_damage_last_5s`: coefficient `0.001212` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001154` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001099` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.001074` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.001048` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001043` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.000980` (raises CT win probability)
- `lag_09__T5__molly`: coefficient `0.000915` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `-0.000890` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000833` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_08__T_place_PIT`: coefficient `-0.002475` (lowers CT win probability)
- `lag_02__CT_place_LIBRARY`: coefficient `0.001659` (raises CT win probability)
- `lag_07__T_place_PIT`: coefficient `-0.001465` (lowers CT win probability)
- `lag_06__T_place_PIT`: coefficient `-0.001434` (lowers CT win probability)
- `lag_12__CT_place_LIBRARY`: coefficient `-0.001275` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.001271` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001259` (lowers CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `-0.001241` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001232` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001226` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `39735`, seconds `84.00`, LSTM delta `-0.2192`

Top all feature movements:
- `lag_08__T_place_PIT`: contribution `-0.015615`
- `lag_11__T_place_BALCONY`: contribution `-0.014123`
- `lag_09__T_place_BALCONY`: contribution `-0.011552`
- `lag_02__CT_place_LIBRARY`: contribution `-0.010637`
- `lag_06__T_place_PIT`: contribution `-0.009050`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.003272`

### tick `35799`, seconds `22.50`, LSTM delta `-0.1857`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.017040`
- `lag_00__CT5__shots_fired`: contribution `-0.013442`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.012915`
- `lag_10__T_place_BALCONY`: contribution `-0.011136`
- `lag_00__CT5__flash_duration`: contribution `-0.009174`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.012915`
- `lag_00__CT5__flash_duration`: contribution `-0.009174`
- `lag_05__utility_damage_diff_last_5s`: contribution `-0.009002`
- `lag_05__CT5__flash_duration`: contribution `-0.003861`
- `lag_00__T1__flash_duration`: contribution `-0.002122`

### tick `35735`, seconds `21.50`, LSTM delta `+0.1466`

Top all feature movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.014938`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.009911`
- `lag_07__T4__flash_duration`: contribution `+0.008054`
- `lag_08__T_place_BALCONY`: contribution `+0.007551`
- `lag_03__CT5__flash_duration`: contribution `+0.006618`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.014938`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.009911`
- `lag_07__T4__flash_duration`: contribution `+0.008054`
- `lag_03__CT5__flash_duration`: contribution `+0.006618`
- `lag_10__T_A_site_active_infernos`: contribution `+0.001888`

### tick `39799`, seconds `85.00`, LSTM delta `-0.0968`

Top all feature movements:
- `lag_08__T_place_PIT`: contribution `-0.015615`
- `lag_11__T_place_BALCONY`: contribution `-0.014123`
- `lag_10__T_place_BALCONY`: contribution `+0.011136`
- `lag_13__T_place_BALCONY`: contribution `-0.010207`
- `lag_08__T_place_BALCONY`: contribution `-0.007551`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.001482`

### tick `35639`, seconds `20.00`, LSTM delta `+0.0758`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.012856`
- `lag_00__CT5__flash_duration`: contribution `+0.009174`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.008338`
- `lag_04__T4__flash_duration`: contribution `+0.003923`
- `lag_07__T_A_site_active_infernos`: contribution `+0.003272`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.012856`
- `lag_00__CT5__flash_duration`: contribution `+0.009174`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.008338`
- `lag_04__T4__flash_duration`: contribution `+0.003923`
- `lag_07__T_A_site_active_infernos`: contribution `+0.003272`
