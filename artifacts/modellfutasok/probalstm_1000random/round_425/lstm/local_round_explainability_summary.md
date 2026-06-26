# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `31`

## Largest probability jumps

- tick `269215`, seconds `65.00`, LSTM `0.9456`, delta `+0.1617`
- tick `267871`, seconds `44.00`, LSTM `0.8469`, delta `+0.1507`
- tick `265599`, seconds `8.50`, LSTM `0.5933`, delta `+0.0998`
- tick `265727`, seconds `10.50`, LSTM `0.6930`, delta `+0.0450`
- tick `267423`, seconds `37.00`, LSTM `0.6544`, delta `-0.0448`
- tick `265695`, seconds `10.00`, LSTM `0.6481`, delta `+0.0346`
- tick `266367`, seconds `20.50`, LSTM `0.5991`, delta `-0.0321`
- tick `267295`, seconds `35.00`, LSTM `0.6907`, delta `+0.0319`
- tick `268319`, seconds `51.00`, LSTM `0.7950`, delta `-0.0249`
- tick `266911`, seconds `29.00`, LSTM `0.6418`, delta `+0.0249`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001625`, |coef| `0.001625`
- `lag_00__kill_diff_last_3s`: coefficient `0.001306`, |coef| `0.001306`
- `lag_00__CT_damage_last_5s`: coefficient `0.001254`, |coef| `0.001254`
- `lag_00__damage_diff_last_5s`: coefficient `0.001203`, |coef| `0.001203`
- `lag_15__CT5__flash_duration`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_15__CT_place_EXTENDEDA`: coefficient `0.001166`, |coef| `0.001166`
- `lag_00__T1__flash_duration`: coefficient `-0.001054`, |coef| `0.001054`
- `lag_11__CT_place_EXTENDEDA`: coefficient `0.000998`, |coef| `0.000998`
- `lag_15__T_he_last_5s`: coefficient `0.000971`, |coef| `0.000971`
- `lag_11__T_place_LONGA`: coefficient `0.000965`, |coef| `0.000965`
- `lag_03__T1__flash_duration`: coefficient `0.000960`, |coef| `0.000960`
- `lag_08__CT_place_LONGA`: coefficient `0.000937`, |coef| `0.000937`
- `lag_10__CT1__flash_duration`: coefficient `0.000924`, |coef| `0.000924`
- `lag_01__CT1__flash_duration`: coefficient `-0.000877`, |coef| `0.000877`
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000869`, |coef| `0.000869`

## Top 10 utility ridge features

- `lag_15__CT5__flash_duration`: coefficient `-0.001166` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.001054` (lowers CT win probability)
- `lag_15__T_he_last_5s`: coefficient `0.000971` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000960` (raises CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `0.000924` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `-0.000877` (lowers CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000869` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000759` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000644` (lowers CT win probability)
- `lag_05__T_he_last_5s`: coefficient `-0.000637` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001625` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001306` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001254` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001203` (raises CT win probability)
- `lag_15__CT_place_EXTENDEDA`: coefficient `0.001166` (raises CT win probability)
- `lag_11__CT_place_EXTENDEDA`: coefficient `0.000998` (raises CT win probability)
- `lag_11__T_place_LONGA`: coefficient `0.000965` (raises CT win probability)
- `lag_08__CT_place_LONGA`: coefficient `0.000937` (raises CT win probability)
- `lag_05__T1__is_scoped`: coefficient `0.000866` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000850` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `269215`, seconds `65.00`, LSTM delta `+0.1617`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.009382`
- `lag_00__T1__flash_duration`: contribution `+0.007091`
- `lag_15__CT_place_EXTENDEDA`: contribution `+0.006548`
- `lag_03__T1__flash_duration`: contribution `+0.006459`
- `lag_00__kill_diff_last_3s`: contribution `+0.006285`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.007091`
- `lag_03__T1__flash_duration`: contribution `+0.006459`

### tick `267871`, seconds `44.00`, LSTM delta `+0.1507`

Top all feature movements:
- `lag_15__CT5__flash_duration`: contribution `+0.006966`
- `lag_15__CT_place_EXTENDEDA`: contribution `+0.006548`
- `lag_05__T1__is_scoped`: contribution `+0.004948`
- `lag_00__CT_kills_last_3s`: contribution `+0.004691`
- `lag_10__CT1__flash_duration`: contribution `+0.004540`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `+0.006966`
- `lag_10__CT1__flash_duration`: contribution `+0.004540`
- `lag_01__CT1__flash_duration`: contribution `+0.004313`
- `lag_04__T_A_site_active_infernos`: contribution `+0.002586`

### tick `265599`, seconds `8.50`, LSTM delta `+0.0998`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `+0.012674`
- `lag_05__T_he_last_5s`: contribution `+0.008313`
- `lag_00__CT_kills_last_3s`: contribution `+0.004691`
- `lag_00__kill_diff_last_3s`: contribution `+0.003142`
- `lag_00__CT_damage_last_5s`: contribution `+0.002733`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `+0.012674`
- `lag_05__T_he_last_5s`: contribution `+0.008313`
- `lag_02__CT1__flash_duration`: contribution `+0.002330`
- `lag_00__T4__utility_total`: contribution `+0.001654`

### tick `265727`, seconds `10.50`, LSTM delta `+0.0450`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.004130`
- `lag_00__T_place_TUNNELSTAIRS`: contribution `+0.003479`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002952`
- `lag_09__T_he_last_5s`: contribution `+0.002705`
- `lag_06__CT1__flash_duration`: contribution `+0.001918`

Top utility-only movements:
- `lag_09__T_he_last_5s`: contribution `+0.002705`
- `lag_06__CT1__flash_duration`: contribution `+0.001918`
- `lag_03__T_flash_duration_sum`: contribution `+0.000798`
- `lag_00__CT_active_infernos`: contribution `+0.000796`

### tick `267423`, seconds `37.00`, LSTM delta `-0.0448`

Top all feature movements:
- `lag_12__CT5__flash_duration`: contribution `-0.004536`
- `lag_03__CT_place_EXTENDEDA`: contribution `-0.003230`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002586`
- `lag_01__CT5__flash_duration`: contribution `-0.002327`
- `lag_00__CT2__duck_amount`: contribution `-0.002291`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `-0.004536`
- `lag_04__T_A_site_active_infernos`: contribution `-0.002586`
- `lag_01__CT5__flash_duration`: contribution `-0.002327`
- `lag_04__T_active_infernos`: contribution `-0.001262`
