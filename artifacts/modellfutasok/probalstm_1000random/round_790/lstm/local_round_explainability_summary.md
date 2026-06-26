# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `16`

## Largest probability jumps

- tick `132191`, seconds `35.00`, LSTM `0.7638`, delta `+0.2212`
- tick `131583`, seconds `25.50`, LSTM `0.5566`, delta `-0.1733`
- tick `133503`, seconds `55.50`, LSTM `0.8953`, delta `+0.1345`
- tick `132415`, seconds `38.50`, LSTM `0.8299`, delta `+0.1191`
- tick `131519`, seconds `24.50`, LSTM `0.7290`, delta `-0.0871`
- tick `132447`, seconds `39.00`, LSTM `0.7481`, delta `-0.0818`
- tick `130655`, seconds `11.00`, LSTM `0.7515`, delta `+0.0643`
- tick `132511`, seconds `40.00`, LSTM `0.8007`, delta `+0.0620`
- tick `130719`, seconds `12.00`, LSTM `0.7890`, delta `+0.0608`
- tick `130943`, seconds `15.50`, LSTM `0.8583`, delta `+0.0517`

## Top 15 local ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003360`, |coef| `0.003360`
- `lag_00__kill_diff_last_3s`: coefficient `0.003025`, |coef| `0.003025`
- `lag_11__T_duck_amount_mean`: coefficient `0.002893`, |coef| `0.002893`
- `lag_07__T_shots_fired_sum`: coefficient `-0.002721`, |coef| `0.002721`
- `lag_00__CT_kills_last_3s`: coefficient `0.002402`, |coef| `0.002402`
- `lag_00__CT_defusing_count`: coefficient `0.002317`, |coef| `0.002317`
- `lag_07__T1__shots_fired`: coefficient `-0.001867`, |coef| `0.001867`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001865`, |coef| `0.001865`
- `lag_01__T_duck_amount_mean`: coefficient `-0.001829`, |coef| `0.001829`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001616`, |coef| `0.001616`
- `lag_01__T5__flash_duration`: coefficient `0.001568`, |coef| `0.001568`
- `lag_15__CT1__is_walking`: coefficient `0.001552`, |coef| `0.001552`
- `lag_00__CT_closest_enemy_dist`: coefficient `-0.001539`, |coef| `0.001539`
- `lag_12__T3__is_walking`: coefficient `-0.001520`, |coef| `0.001520`
- `lag_00__damage_diff_last_5s`: coefficient `0.001512`, |coef| `0.001512`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.003360` (lowers CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001568` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001432` (lowers CT win probability)
- `lag_04__T_B_site_active_smokes`: coefficient `-0.001358` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001269` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.001231` (raises CT win probability)
- `lag_15__T3__smoke`: coefficient `-0.001045` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `0.001011` (raises CT win probability)
- `lag_13__T_B_site_active_smokes`: coefficient `0.000965` (raises CT win probability)
- `lag_04__active_infernos_total`: coefficient `0.000932` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003025` (raises CT win probability)
- `lag_11__T_duck_amount_mean`: coefficient `0.002893` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `-0.002721` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002402` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002317` (raises CT win probability)
- `lag_07__T1__shots_fired`: coefficient `-0.001867` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001865` (lowers CT win probability)
- `lag_01__T_duck_amount_mean`: coefficient `-0.001829` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001616` (raises CT win probability)
- `lag_15__CT1__is_walking`: coefficient `0.001552` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `132191`, seconds `35.00`, LSTM delta `+0.2212`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.040799`
- `lag_07__T1__shots_fired`: contribution `+0.022315`
- `lag_11__T_duck_amount_mean`: contribution `+0.011218`
- `lag_05__CT_place_LIBRARY`: contribution `+0.008475`
- `lag_00__kill_diff_last_3s`: contribution `+0.007281`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131583`, seconds `25.50`, LSTM delta `-0.1733`

Top all feature movements:
- `lag_01__T5__flash_duration`: contribution `-0.011826`
- `lag_01__CT_place_BALCONY`: contribution `-0.008658`
- `lag_11__CT5__flash_duration`: contribution `-0.007879`
- `lag_08__CT_place_BALCONY`: contribution `-0.007305`
- `lag_00__kill_diff_last_3s`: contribution `-0.007281`

Top utility-only movements:
- `lag_01__T5__flash_duration`: contribution `-0.011826`
- `lag_11__CT5__flash_duration`: contribution `-0.007879`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.002626`

### tick `133503`, seconds `55.50`, LSTM delta `+0.1345`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.020384`
- `lag_11__T_duck_amount_mean`: contribution `+0.014941`
- `lag_01__T_duck_amount_mean`: contribution `+0.010636`
- `lag_00__kill_diff_last_3s`: contribution `+0.007281`
- `lag_00__CT_kills_last_3s`: contribution `+0.006934`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.020384`

### tick `132415`, seconds `38.50`, LSTM delta `+0.1191`

Top all feature movements:
- `lag_14__T_shots_fired_sum`: contribution `+0.015284`
- `lag_14__T1__shots_fired`: contribution `+0.009322`
- `lag_00__kill_diff_last_3s`: contribution `+0.007281`
- `lag_00__CT_kills_last_3s`: contribution `+0.006934`
- `lag_15__T_shots_fired_sum`: contribution `+0.005590`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.001791`

### tick `131519`, seconds `24.50`, LSTM delta `-0.0871`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.007281`
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.006025`
- `lag_09__CT5__flash_duration`: contribution `-0.005541`
- `lag_06__CT_place_BALCONY`: contribution `-0.004488`
- `lag_00__T_kills_last_3s`: contribution `-0.004266`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.006025`
- `lag_09__CT5__flash_duration`: contribution `-0.005541`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.004113`
- `lag_04__CT_B_site_active_infernos`: contribution `-0.002699`
- `lag_04__CT_active_infernos`: contribution `-0.002330`
