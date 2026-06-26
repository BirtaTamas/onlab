# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `17`

## Largest probability jumps

- tick `154099`, seconds `82.50`, LSTM `0.8138`, delta `+0.5095`
- tick `154131`, seconds `83.00`, LSTM `0.4943`, delta `-0.3194`
- tick `151699`, seconds `45.00`, LSTM `0.7050`, delta `+0.2814`
- tick `154803`, seconds `93.50`, LSTM `0.9039`, delta `+0.1758`
- tick `151635`, seconds `44.00`, LSTM `0.3394`, delta `+0.1727`
- tick `154611`, seconds `90.50`, LSTM `0.7480`, delta `+0.1593`
- tick `150771`, seconds `30.50`, LSTM `0.5061`, delta `-0.1355`
- tick `152115`, seconds `51.50`, LSTM `0.9124`, delta `+0.1277`
- tick `153043`, seconds `66.00`, LSTM `0.3767`, delta `-0.1252`
- tick `152499`, seconds `57.50`, LSTM `0.5736`, delta `-0.1236`

## Top 15 local ridge features

- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.003685`, |coef| `0.003685`
- `lag_00__kill_diff_last_3s`: coefficient `0.003339`, |coef| `0.003339`
- `lag_00__CT_place_STORAGEROOM`: coefficient `-0.003293`, |coef| `0.003293`
- `lag_00__CT_kills_last_3s`: coefficient `0.002990`, |coef| `0.002990`
- `lag_00__damage_diff_last_5s`: coefficient `0.002954`, |coef| `0.002954`
- `lag_00__T_place_LOWERPARK`: coefficient `-0.002873`, |coef| `0.002873`
- `lag_08__CT_place_STORAGEROOM`: coefficient `0.002830`, |coef| `0.002830`
- `lag_02__T_place_RESTROOM`: coefficient `0.002716`, |coef| `0.002716`
- `lag_05__T_place_RESTROOM`: coefficient `0.002558`, |coef| `0.002558`
- `lag_06__CT1__duck_amount`: coefficient `0.002477`, |coef| `0.002477`
- `lag_02__T_place_UPPERPARK`: coefficient `-0.002435`, |coef| `0.002435`
- `lag_04__CT_place_WATER`: coefficient `0.002379`, |coef| `0.002379`
- `lag_09__T_place_FOUNTAIN`: coefficient `-0.002286`, |coef| `0.002286`
- `lag_00__CT_damage_last_5s`: coefficient `0.002185`, |coef| `0.002185`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002159`, |coef| `0.002159`

## Top 10 utility ridge features

- `lag_01__T_utility_damage_last_5s`: coefficient `0.001605` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001162` (lowers CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `-0.001121` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `-0.001096` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001035` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.001019` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000998` (lowers CT win probability)
- `lag_13__CT_smokes_last_5s`: coefficient `-0.000975` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.000918` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.000811` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.003685` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003339` (raises CT win probability)
- `lag_00__CT_place_STORAGEROOM`: coefficient `-0.003293` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002990` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002954` (raises CT win probability)
- `lag_00__T_place_LOWERPARK`: coefficient `-0.002873` (lowers CT win probability)
- `lag_08__CT_place_STORAGEROOM`: coefficient `0.002830` (raises CT win probability)
- `lag_02__T_place_RESTROOM`: coefficient `0.002716` (raises CT win probability)
- `lag_05__T_place_RESTROOM`: coefficient `0.002558` (raises CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `0.002477` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `154099`, seconds `82.50`, LSTM delta `+0.5095`

Top all feature movements:
- `lag_05__CT_place_STORAGEROOM`: contribution `+0.078834`
- `lag_08__CT_place_STORAGEROOM`: contribution `+0.060544`
- `lag_02__T_place_RESTROOM`: contribution `+0.052394`
- `lag_08__CT_place_BACKOFA`: contribution `+0.014658`
- `lag_05__CT_place_LOBBY`: contribution `+0.014298`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154131`, seconds `83.00`, LSTM delta `-0.3194`

Top all feature movements:
- `lag_06__CT_place_STORAGEROOM`: contribution `-0.045289`
- `lag_09__CT_place_STORAGEROOM`: contribution `-0.042262`
- `lag_06__CT_place_LOBBY`: contribution `-0.013602`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013500`
- `lag_00__T_shots_fired_sum`: contribution `-0.011021`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `151699`, seconds `45.00`, LSTM delta `+0.2814`

Top all feature movements:
- `lag_05__T_place_RESTROOM`: contribution `+0.049342`
- `lag_00__T_place_RESTROOM`: contribution `+0.026124`
- `lag_09__T_place_FOUNTAIN`: contribution `+0.010805`
- `lag_06__CT_place_WATER`: contribution `+0.010744`
- `lag_06__CT_place_WALKWAY`: contribution `+0.010319`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154803`, seconds `93.50`, LSTM delta `+0.1758`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `+0.018029`
- `lag_01__T_utility_damage_last_5s`: contribution `+0.013516`
- `lag_00__T_place_LOWERPARK`: contribution `+0.011582`
- `lag_06__CT1__duck_amount`: contribution `+0.009449`
- `lag_00__CT_kills_last_3s`: contribution `+0.008634`

Top utility-only movements:
- `lag_01__T_utility_damage_last_5s`: contribution `+0.013516`
- `lag_00__T_flash_alpha_mean`: contribution `+0.007049`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.005841`

### tick `151635`, seconds `44.00`, LSTM delta `+0.1727`

Top all feature movements:
- `lag_04__CT_place_WATER`: contribution `+0.014458`
- `lag_09__T_place_FOUNTAIN`: contribution `+0.010805`
- `lag_00__CT_kills_last_3s`: contribution `+0.008634`
- `lag_03__T_place_RESTROOM`: contribution `-0.008550`
- `lag_00__kill_diff_last_3s`: contribution `+0.008036`

Top utility-only movements:
- No utility movement among the top local contributors.
