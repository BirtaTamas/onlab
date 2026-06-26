# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `17`

## Largest probability jumps

- tick `140839`, seconds `87.50`, LSTM `0.7395`, delta `+0.2618`
- tick `139207`, seconds `62.00`, LSTM `0.7778`, delta `+0.2578`
- tick `140743`, seconds `86.00`, LSTM `0.4938`, delta `-0.2112`
- tick `141159`, seconds `92.50`, LSTM `0.8292`, delta `+0.1587`
- tick `137223`, seconds `31.00`, LSTM `0.4904`, delta `-0.1021`
- tick `138599`, seconds `52.50`, LSTM `0.4292`, delta `-0.0895`
- tick `136935`, seconds `26.50`, LSTM `0.5144`, delta `-0.0797`
- tick `137447`, seconds `34.50`, LSTM `0.3371`, delta `-0.0787`
- tick `141351`, seconds `95.50`, LSTM `0.9503`, delta `+0.0630`
- tick `138055`, seconds `44.00`, LSTM `0.3686`, delta `-0.0612`

## Top 15 local ridge features

- `lag_13__CT_place_TOPOFMID`: coefficient `-0.003194`, |coef| `0.003194`
- `lag_00__T_place_SHOP`: coefficient `-0.003176`, |coef| `0.003176`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002968`, |coef| `0.002968`
- `lag_00__kill_diff_last_3s`: coefficient `0.002808`, |coef| `0.002808`
- `lag_10__CT5__duck_amount`: coefficient `-0.002797`, |coef| `0.002797`
- `lag_03__CT5__duck_amount`: coefficient `-0.002616`, |coef| `0.002616`
- `lag_15__T1__duck_amount`: coefficient `-0.002500`, |coef| `0.002500`
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.002322`, |coef| `0.002322`
- `lag_13__CT_place_MIDDLE`: coefficient `0.002317`, |coef| `0.002317`
- `lag_04__T2__duck_amount`: coefficient `-0.002252`, |coef| `0.002252`
- `lag_00__CT_kills_last_3s`: coefficient `0.002200`, |coef| `0.002200`
- `lag_10__T1__is_walking`: coefficient `-0.002190`, |coef| `0.002190`
- `lag_00__CT_place_UNDERPASS`: coefficient `0.002099`, |coef| `0.002099`
- `lag_12__CT_place_SHOP`: coefficient `-0.002003`, |coef| `0.002003`
- `lag_05__T_place_SCAFFOLDING`: coefficient `-0.001949`, |coef| `0.001949`

## Top 10 utility ridge features

- `lag_15__T_smokes_last_5s`: coefficient `-0.001297` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001234` (lowers CT win probability)
- `lag_13__T_smokes_last_5s`: coefficient `0.000933` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000903` (lowers CT win probability)
- `lag_03__CT5__flash`: coefficient `-0.000858` (lowers CT win probability)
- `lag_10__T_smokes_last_5s`: coefficient `-0.000815` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000803` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000800` (lowers CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.000778` (lowers CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `0.000703` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_TOPOFMID`: coefficient `-0.003194` (lowers CT win probability)
- `lag_00__T_place_SHOP`: coefficient `-0.003176` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002968` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002808` (raises CT win probability)
- `lag_10__CT5__duck_amount`: coefficient `-0.002797` (lowers CT win probability)
- `lag_03__CT5__duck_amount`: coefficient `-0.002616` (lowers CT win probability)
- `lag_15__T1__duck_amount`: coefficient `-0.002500` (lowers CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `-0.002322` (lowers CT win probability)
- `lag_13__CT_place_MIDDLE`: coefficient `0.002317` (raises CT win probability)
- `lag_04__T2__duck_amount`: coefficient `-0.002252` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `140839`, seconds `87.50`, LSTM delta `+0.2618`

Top all feature movements:
- `lag_03__CT_place_UNDERPASS`: contribution `+0.013463`
- `lag_13__CT_place_TOPOFMID`: contribution `+0.011592`
- `lag_10__CT5__duck_amount`: contribution `+0.010560`
- `lag_12__CT_place_SHOP`: contribution `+0.010044`
- `lag_03__CT5__duck_amount`: contribution `+0.009876`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `139207`, seconds `62.00`, LSTM delta `+0.2578`

Top all feature movements:
- `lag_00__T_place_SHOP`: contribution `+0.054828`
- `lag_00__T_shots_fired_sum`: contribution `+0.022255`
- `lag_11__CT_place_STAIRS`: contribution `+0.007633`
- `lag_00__kill_diff_last_3s`: contribution `+0.006758`
- `lag_08__CT_place_STAIRS`: contribution `+0.006499`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140743`, seconds `86.00`, LSTM delta `-0.2112`

Top all feature movements:
- `lag_00__CT_place_UNDERPASS`: contribution `-0.012170`
- `lag_13__CT_place_TOPOFMID`: contribution `-0.011592`
- `lag_10__CT5__duck_amount`: contribution `-0.010560`
- `lag_03__CT5__duck_amount`: contribution `-0.009876`
- `lag_09__CT_place_SHOP`: contribution `-0.008033`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141159`, seconds `92.50`, LSTM delta `+0.1587`

Top all feature movements:
- `lag_05__T_place_SCAFFOLDING`: contribution `+0.066383`
- `lag_07__T_place_SCAFFOLDING`: contribution `+0.036496`
- `lag_00__kill_diff_last_3s`: contribution `+0.006758`
- `lag_00__CT_kills_last_3s`: contribution `+0.006351`
- `lag_15__T4__duck_amount`: contribution `+0.005830`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137223`, seconds `31.00`, LSTM delta `-0.1021`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.022342`
- `lag_05__CT_place_STAIRS`: contribution `-0.007434`
- `lag_03__CT_place_LADDER`: contribution `-0.006829`
- `lag_08__T_place_PALACEALLEY`: contribution `-0.004620`
- `lag_13__CT_place_JUNGLE`: contribution `-0.004497`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.003835`
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.003462`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.002511`
- `lag_09__CT3__flash_duration`: contribution `-0.002458`
- `lag_00__T_A_site_active_infernos`: contribution `+0.002390`
