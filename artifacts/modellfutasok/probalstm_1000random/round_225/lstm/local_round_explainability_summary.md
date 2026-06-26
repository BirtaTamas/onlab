# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `120592`, seconds `40.50`, LSTM `0.1029`, delta `-0.1341`
- tick `120560`, seconds `40.00`, LSTM `0.2369`, delta `-0.0800`
- tick `118064`, seconds `1.00`, LSTM `0.2448`, delta `+0.0666`
- tick `120240`, seconds `35.00`, LSTM `0.3189`, delta `+0.0658`
- tick `120656`, seconds `41.50`, LSTM `0.0336`, delta `-0.0656`
- tick `118096`, seconds `1.50`, LSTM `0.3102`, delta `+0.0654`
- tick `118448`, seconds `7.00`, LSTM `0.3621`, delta `-0.0436`
- tick `118704`, seconds `11.00`, LSTM `0.2807`, delta `-0.0385`
- tick `120528`, seconds `39.50`, LSTM `0.3169`, delta `-0.0379`
- tick `120496`, seconds `39.00`, LSTM `0.3548`, delta `+0.0371`

## Top 15 local ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.001675`, |coef| `0.001675`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001107`, |coef| `0.001107`
- `lag_00__CT_place_BALCONY`: coefficient `-0.001056`, |coef| `0.001056`
- `lag_10__CT_smokes_last_5s`: coefficient `0.001011`, |coef| `0.001011`
- `lag_03__T_shots_fired_sum`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_01__CT_flashes_last_5s`: coefficient `0.000888`, |coef| `0.000888`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000866`, |coef| `0.000866`
- `lag_00__T4__shots_fired`: coefficient `0.000783`, |coef| `0.000783`
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000777`, |coef| `0.000777`
- `lag_01__CT_smokes_last_5s`: coefficient `0.000764`, |coef| `0.000764`
- `lag_03__T4__shots_fired`: coefficient `-0.000749`, |coef| `0.000749`
- `lag_02__CT_smokes_last_5s`: coefficient `0.000743`, |coef| `0.000743`
- `lag_02__T_B_site_active_infernos`: coefficient `0.000740`, |coef| `0.000740`
- `lag_00__CT5__is_scoped`: coefficient `0.000730`, |coef| `0.000730`
- `lag_01__T4__shots_fired`: coefficient `-0.000691`, |coef| `0.000691`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.001675` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001107` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `0.001011` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.000888` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000866` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000777` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000764` (raises CT win probability)
- `lag_02__CT_smokes_last_5s`: coefficient `0.000743` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.000740` (raises CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.000662` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_BALCONY`: coefficient `-0.001056` (lowers CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `-0.000974` (lowers CT win probability)
- `lag_00__T4__shots_fired`: coefficient `0.000783` (raises CT win probability)
- `lag_03__T4__shots_fired`: coefficient `-0.000749` (lowers CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `0.000730` (raises CT win probability)
- `lag_01__T4__shots_fired`: coefficient `-0.000691` (lowers CT win probability)
- `lag_02__CT_place_BALCONY`: coefficient `0.000669` (raises CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.000663` (lowers CT win probability)
- `lag_02__CT_place_ARCH`: coefficient `0.000662` (raises CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `0.000655` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `120592`, seconds `40.50`, LSTM delta `-0.1341`

Top all feature movements:
- `lag_00__T4__shots_fired`: contribution `-0.010637`
- `lag_03__T_shots_fired_sum`: contribution `-0.004380`
- `lag_03__CT_flashed_players`: contribution `-0.003941`
- `lag_02__T_flashed_players`: contribution `-0.003836`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.003718`

Top utility-only movements:
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.003718`
- `lag_13__T_B_site_active_infernos`: contribution `-0.003417`
- `lag_00__T3__flash_duration`: contribution `-0.003225`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.002696`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.002399`

### tick `120560`, seconds `40.00`, LSTM delta `-0.0800`

Top all feature movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.005298`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.003526`
- `lag_01__T4__shots_fired`: contribution `-0.002987`
- `lag_02__T_shots_fired_sum`: contribution `-0.002938`
- `lag_03__T_shots_fired_sum`: contribution `-0.002920`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.005298`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.003526`
- `lag_12__T_B_site_active_infernos`: contribution `-0.002793`
- `lag_02__T4__flash_duration`: contribution `-0.001788`

### tick `118064`, seconds `1.00`, LSTM delta `+0.0666`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.028946`
- `lag_01__CT_flashes_last_5s`: contribution `+0.009766`
- `lag_02__CT_place_CTSPAWN`: contribution `+0.001393`
- `lag_01__CT4__flash`: contribution `+0.001380`
- `lag_02__CT_closest_enemy_dist`: contribution `+0.001220`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.028946`
- `lag_01__CT_flashes_last_5s`: contribution `+0.009766`
- `lag_01__CT4__flash`: contribution `+0.001380`
- `lag_01__CT4__utility_total`: contribution `+0.000622`
- `lag_02__molly_inv_diff`: contribution `+0.000357`

### tick `120240`, seconds `35.00`, LSTM delta `+0.0658`

Top all feature movements:
- `lag_00__CT_place_BALCONY`: contribution `+0.006780`
- `lag_03__T_shots_fired_sum`: contribution `+0.006570`
- `lag_02__CT_place_BALCONY`: contribution `+0.004295`
- `lag_02__T_B_site_active_infernos`: contribution `+0.004184`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003698`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `+0.004184`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.003698`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.003526`
- `lag_02__T_active_infernos`: contribution `+0.002324`

### tick `120656`, seconds `41.50`, LSTM delta `-0.0656`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.008325`
- `lag_02__T4__shots_fired`: contribution `+0.006139`
- `lag_03__T_shots_fired_sum`: contribution `-0.005110`
- `lag_02__T3__flash_duration`: contribution `-0.002678`
- `lag_15__CT_place_BALCONY`: contribution `-0.002557`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `-0.002678`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.001745`
- `lag_15__T_B_site_active_infernos`: contribution `-0.001732`
- `lag_01__T_B_site_active_infernos`: contribution `-0.001672`
- `lag_02__T_flash_duration_sum`: contribution `-0.001634`
