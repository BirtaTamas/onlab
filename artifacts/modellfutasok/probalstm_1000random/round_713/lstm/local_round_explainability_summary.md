# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `74356`, seconds `65.00`, LSTM `0.8533`, delta `+0.1735`
- tick `74580`, seconds `68.50`, LSTM `0.9077`, delta `+0.1199`
- tick `71892`, seconds `26.50`, LSTM `0.5865`, delta `+0.0899`
- tick `74324`, seconds `64.50`, LSTM `0.6797`, delta `+0.0678`
- tick `72212`, seconds `31.50`, LSTM `0.5791`, delta `-0.0522`
- tick `75988`, seconds `90.50`, LSTM `0.9551`, delta `+0.0473`
- tick `76276`, seconds `95.00`, LSTM `0.9585`, delta `+0.0434`
- tick `74420`, seconds `66.00`, LSTM `0.8128`, delta `-0.0369`
- tick `70708`, seconds `8.00`, LSTM `0.5323`, delta `+0.0356`
- tick `70964`, seconds `12.00`, LSTM `0.5454`, delta `+0.0280`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001512`, |coef| `0.001512`
- `lag_03__T_flashed_players`: coefficient `0.001308`, |coef| `0.001308`
- `lag_00__kill_diff_last_3s`: coefficient `0.001297`, |coef| `0.001297`
- `lag_02__CT_place_TOPOFMID`: coefficient `0.001267`, |coef| `0.001267`
- `lag_01__T2__shots_fired`: coefficient `0.001124`, |coef| `0.001124`
- `lag_02__T2__shots_fired`: coefficient `0.001111`, |coef| `0.001111`
- `lag_03__CT_flashed_players`: coefficient `0.001102`, |coef| `0.001102`
- `lag_00__T2__shots_fired`: coefficient `0.001056`, |coef| `0.001056`
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001048`, |coef| `0.001048`
- `lag_05__CT_place_QUAD`: coefficient `0.001027`, |coef| `0.001027`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001022`, |coef| `0.001022`
- `lag_00__T3__alive`: coefficient `-0.000987`, |coef| `0.000987`
- `lag_08__T5__duck_amount`: coefficient `-0.000954`, |coef| `0.000954`
- `lag_03__CT2__flash_duration`: coefficient `0.000954`, |coef| `0.000954`
- `lag_00__T_flashed_players`: coefficient `-0.000947`, |coef| `0.000947`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001048` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.000954` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000860` (raises CT win probability)
- `lag_10__CT_B_site_active_infernos`: coefficient `0.000852` (raises CT win probability)
- `lag_14__CT1__molly`: coefficient `-0.000840` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000806` (raises CT win probability)
- `lag_02__CT2__flash_duration`: coefficient `0.000688` (raises CT win probability)
- `lag_14__CT5__smoke`: coefficient `-0.000648` (lowers CT win probability)
- `lag_10__CT_active_infernos`: coefficient `0.000638` (raises CT win probability)
- `lag_13__CT1__molly`: coefficient `-0.000631` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001512` (raises CT win probability)
- `lag_03__T_flashed_players`: coefficient `0.001308` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001297` (raises CT win probability)
- `lag_02__CT_place_TOPOFMID`: coefficient `0.001267` (raises CT win probability)
- `lag_01__T2__shots_fired`: coefficient `0.001124` (raises CT win probability)
- `lag_02__T2__shots_fired`: coefficient `0.001111` (raises CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.001102` (raises CT win probability)
- `lag_00__T2__shots_fired`: coefficient `0.001056` (raises CT win probability)
- `lag_05__CT_place_QUAD`: coefficient `0.001027` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001022` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `74356`, seconds `65.00`, LSTM delta `+0.1735`

Top all feature movements:
- `lag_03__T_flashed_players`: contribution `+0.005050`
- `lag_03__CT_flashed_players`: contribution `+0.004827`
- `lag_02__CT_place_TOPOFMID`: contribution `+0.004597`
- `lag_00__CT_kills_last_3s`: contribution `+0.004366`
- `lag_08__T5__is_scoped`: contribution `+0.003677`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `+0.003105`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.002929`
- `lag_14__CT1__molly`: contribution `+0.002092`

### tick `74580`, seconds `68.50`, LSTM delta `+0.1199`

Top all feature movements:
- `lag_05__CT_place_QUAD`: contribution `+0.008092`
- `lag_00__CT_kills_last_3s`: contribution `+0.004366`
- `lag_01__T2__shots_fired`: contribution `+0.003306`
- `lag_02__T2__shots_fired`: contribution `+0.003268`
- `lag_00__kill_diff_last_3s`: contribution `+0.003121`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71892`, seconds `26.50`, LSTM delta `+0.0899`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.013155`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.008851`
- `lag_03__T_flashed_players`: contribution `+0.005050`
- `lag_03__T2__flash_duration`: contribution `+0.004585`
- `lag_03__CT_place_BALCONY`: contribution `+0.004033`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.013155`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.008851`
- `lag_03__T2__flash_duration`: contribution `+0.004585`
- `lag_03__T_flash_duration_sum`: contribution `+0.003016`
- `lag_06__CT3__flash_duration`: contribution `+0.002848`

### tick `74324`, seconds `64.50`, LSTM delta `+0.0678`

Top all feature movements:
- `lag_07__T5__duck_amount`: contribution `+0.003353`
- `lag_02__T_flashed_players`: contribution `+0.003312`
- `lag_01__CT_place_TOPOFMID`: contribution `+0.002879`
- `lag_00__CT5__duck_amount`: contribution `+0.002738`
- `lag_02__CT_flashed_players`: contribution `+0.002641`

Top utility-only movements:
- `lag_02__CT2__flash_duration`: contribution `+0.002239`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.001728`
- `lag_13__CT1__molly`: contribution `+0.001571`

### tick `72212`, seconds `31.50`, LSTM delta `-0.0522`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.013155`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.008851`
- `lag_03__T_place_BALCONY`: contribution `-0.004624`
- `lag_03__T2__flash_duration`: contribution `-0.004585`
- `lag_09__T1__duck_amount`: contribution `-0.003145`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.013155`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.008851`
- `lag_03__T2__flash_duration`: contribution `-0.004585`
- `lag_03__T_flash_duration_sum`: contribution `-0.001441`
- `lag_02__T4__flash_duration`: contribution `-0.001369`
