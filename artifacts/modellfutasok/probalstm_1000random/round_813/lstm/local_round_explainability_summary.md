# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `15641`, seconds `48.50`, LSTM `0.2797`, delta `-0.2872`
- tick `16505`, seconds `62.00`, LSTM `0.0504`, delta `-0.1378`
- tick `13561`, seconds `16.00`, LSTM `0.3575`, delta `-0.1038`
- tick `14937`, seconds `37.50`, LSTM `0.4243`, delta `-0.0984`
- tick `15673`, seconds `49.00`, LSTM `0.1877`, delta `-0.0920`
- tick `15193`, seconds `41.50`, LSTM `0.6255`, delta `+0.0692`
- tick `13657`, seconds `17.50`, LSTM `0.4342`, delta `+0.0689`
- tick `15097`, seconds `40.00`, LSTM `0.5307`, delta `+0.0641`
- tick `15225`, seconds `42.00`, LSTM `0.5684`, delta `-0.0570`
- tick `13209`, seconds `10.50`, LSTM `0.4423`, delta `-0.0466`

## Top 15 local ridge features

- `lag_14__T_shots_fired_sum`: coefficient `0.003418`, |coef| `0.003418`
- `lag_14__T5__shots_fired`: coefficient `0.002866`, |coef| `0.002866`
- `lag_00__T_kills_last_3s`: coefficient `-0.002609`, |coef| `0.002609`
- `lag_06__T_place_LONGDOG`: coefficient `0.002412`, |coef| `0.002412`
- `lag_00__kill_diff_last_3s`: coefficient `0.002315`, |coef| `0.002315`
- `lag_13__CT_shots_fired_sum`: coefficient `0.002300`, |coef| `0.002300`
- `lag_00__T_damage_last_5s`: coefficient `-0.002041`, |coef| `0.002041`
- `lag_10__CT_place_ENTRANCE`: coefficient `0.001821`, |coef| `0.001821`
- `lag_13__CT3__shots_fired`: coefficient `0.001799`, |coef| `0.001799`
- `lag_04__T2__has_bomb`: coefficient `-0.001647`, |coef| `0.001647`
- `lag_07__T_place_LONGDOG`: coefficient `0.001609`, |coef| `0.001609`
- `lag_07__T_A_site_active_infernos`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_02__T_place_IVY`: coefficient `-0.001573`, |coef| `0.001573`
- `lag_06__T_place_DUMPSTER`: coefficient `0.001545`, |coef| `0.001545`
- `lag_00__CT2__alive`: coefficient `0.001464`, |coef| `0.001464`

## Top 10 utility ridge features

- `lag_07__T_A_site_active_infernos`: coefficient `-0.001594` (lowers CT win probability)
- `lag_13__T2__molly`: coefficient `0.001187` (raises CT win probability)
- `lag_09__T5__molly`: coefficient `0.001173` (raises CT win probability)
- `lag_05__T2__molly`: coefficient `0.001169` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001149` (lowers CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.001139` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001094` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `0.001041` (raises CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `0.001012` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `-0.000948` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_shots_fired_sum`: coefficient `0.003418` (raises CT win probability)
- `lag_14__T5__shots_fired`: coefficient `0.002866` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002609` (lowers CT win probability)
- `lag_06__T_place_LONGDOG`: coefficient `0.002412` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002315` (raises CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `0.002300` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002041` (lowers CT win probability)
- `lag_10__CT_place_ENTRANCE`: coefficient `0.001821` (raises CT win probability)
- `lag_13__CT3__shots_fired`: coefficient `0.001799` (raises CT win probability)
- `lag_04__T2__has_bomb`: coefficient `-0.001647` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15641`, seconds `48.50`, LSTM delta `-0.2872`

Top all feature movements:
- `lag_14__T_shots_fired_sum`: contribution `-0.028189`
- `lag_14__T5__shots_fired`: contribution `-0.019381`
- `lag_10__CT_place_ENTRANCE`: contribution `-0.016161`
- `lag_13__CT_shots_fired_sum`: contribution `-0.012783`
- `lag_06__T_place_LONGDOG`: contribution `-0.011222`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.004744`

### tick `16505`, seconds `62.00`, LSTM delta `-0.1378`

Top all feature movements:
- `lag_06__T_place_DUMPSTER`: contribution `-0.014044`
- `lag_11__T_place_DUMPSTER`: contribution `-0.008342`
- `lag_00__T_kills_last_3s`: contribution `-0.008265`
- `lag_08__T_place_DUMPSTER`: contribution `-0.006504`
- `lag_00__kill_diff_last_3s`: contribution `-0.005571`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.003091`
- `lag_05__T4__flash_duration`: contribution `-0.002174`
- `lag_14__CT_flash_duration_sum`: contribution `-0.002010`
- `lag_02__CT_B_site_active_smokes`: contribution `-0.001680`

### tick `13561`, seconds `16.00`, LSTM delta `-0.1038`

Top all feature movements:
- `lag_11__CT_flashed_players`: contribution `-0.009872`
- `lag_00__T_place_ALLEY`: contribution `-0.009589`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `-0.008186`
- `lag_15__T_place_DUMPSTER`: contribution `-0.006640`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.005749`

Top utility-only movements:
- `lag_11__CT4__flash_duration`: contribution `-0.005523`
- `lag_11__CT_flash_duration_sum`: contribution `-0.002985`
- `lag_04__T4__flash_duration`: contribution `-0.002934`
- `lag_11__T4__flash_duration`: contribution `-0.002457`

### tick `14937`, seconds `37.50`, LSTM delta `-0.0984`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.008265`
- `lag_00__kill_diff_last_3s`: contribution `-0.005571`
- `lag_00__T_shots_fired_sum`: contribution `-0.004086`
- `lag_04__T_place_IVY`: contribution `-0.004054`
- `lag_10__T_place_TMAIN`: contribution `-0.003262`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.002787`
- `lag_13__T1__flash_duration`: contribution `-0.002198`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.001837`

### tick `15673`, seconds `49.00`, LSTM delta `-0.0920`

Top all feature movements:
- `lag_07__T_place_LONGDOG`: contribution `-0.007486`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.005789`
- `lag_01__T4__duck_amount`: contribution `-0.002883`
- `lag_05__T2__has_bomb`: contribution `-0.002874`
- `lag_01__T_kills_last_3s`: contribution `-0.002840`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `-0.002513`
- `lag_14__T2__molly`: contribution `-0.001899`
