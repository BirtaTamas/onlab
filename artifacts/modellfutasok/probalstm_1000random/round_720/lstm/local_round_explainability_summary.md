# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `22`

## Largest probability jumps

- tick `187451`, seconds `67.00`, LSTM `0.8685`, delta `+0.1856`
- tick `186171`, seconds `47.00`, LSTM `0.7131`, delta `+0.1820`
- tick `185691`, seconds `39.50`, LSTM `0.3974`, delta `-0.1535`
- tick `185179`, seconds `31.50`, LSTM `0.5918`, delta `+0.0811`
- tick `185883`, seconds `42.50`, LSTM `0.4458`, delta `+0.0678`
- tick `187195`, seconds `63.00`, LSTM `0.6854`, delta `-0.0555`
- tick `185211`, seconds `32.00`, LSTM `0.6434`, delta `+0.0516`
- tick `186363`, seconds `50.00`, LSTM `0.7010`, delta `-0.0500`
- tick `183931`, seconds `12.00`, LSTM `0.4248`, delta `+0.0450`
- tick `188667`, seconds `86.00`, LSTM `0.9742`, delta `+0.0430`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003069`, |coef| `0.003069`
- `lag_00__CT_kills_last_3s`: coefficient `0.002882`, |coef| `0.002882`
- `lag_00__damage_diff_last_5s`: coefficient `0.002252`, |coef| `0.002252`
- `lag_00__CT_place_CATWALK`: coefficient `0.002001`, |coef| `0.002001`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001912`, |coef| `0.001912`
- `lag_00__T1__shots_fired`: coefficient `0.001904`, |coef| `0.001904`
- `lag_15__CT_place_CATWALK`: coefficient `-0.001847`, |coef| `0.001847`
- `lag_00__CT_damage_last_5s`: coefficient `0.001824`, |coef| `0.001824`
- `lag_08__CT_place_JUNGLE`: coefficient `-0.001810`, |coef| `0.001810`
- `lag_08__T_place_SIDEALLEY`: coefficient `-0.001771`, |coef| `0.001771`
- `lag_15__CT_place_UNDERPASS`: coefficient `0.001729`, |coef| `0.001729`
- `lag_00__CT3__is_walking`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_01__T1__shots_fired`: coefficient `0.001619`, |coef| `0.001619`
- `lag_12__CT5__duck_amount`: coefficient `-0.001443`, |coef| `0.001443`

## Top 10 utility ridge features

- `lag_03__T_B_site_active_infernos`: coefficient `-0.001407` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.001256` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.001052` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.001015` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.001012` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000985` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000873` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000815` (raises CT win probability)
- `lag_06__CT_B_site_active_smokes`: coefficient `-0.000814` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000810` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003069` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002882` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002252` (raises CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `0.002001` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001976` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001912` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `0.001904` (raises CT win probability)
- `lag_15__CT_place_CATWALK`: coefficient `-0.001847` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001824` (raises CT win probability)
- `lag_08__CT_place_JUNGLE`: coefficient `-0.001810` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `187451`, seconds `67.00`, LSTM delta `+0.1856`

Top all feature movements:
- `lag_08__CT_place_JUNGLE`: contribution `+0.011609`
- `lag_15__CT_place_UNDERPASS`: contribution `+0.010029`
- `lag_00__CT_kills_last_3s`: contribution `+0.008322`
- `lag_00__kill_diff_last_3s`: contribution `+0.007387`
- `lag_15__CT_place_CATWALK`: contribution `+0.007357`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `186171`, seconds `47.00`, LSTM delta `+0.1820`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008322`
- `lag_00__T_shots_fired_sum`: contribution `+0.007407`
- `lag_00__kill_diff_last_3s`: contribution `+0.007387`
- `lag_15__CT_place_CATWALK`: contribution `+0.007357`
- `lag_15__T_shots_fired_sum`: contribution `+0.006879`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `+0.002862`
- `lag_00__T2__flash`: contribution `+0.002569`

### tick `185691`, seconds `39.50`, LSTM delta `-0.1535`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.013333`
- `lag_05__CT_place_TRUCK`: contribution `-0.008628`
- `lag_00__CT_place_CATWALK`: contribution `-0.007971`
- `lag_03__T_B_site_active_infernos`: contribution `-0.007956`
- `lag_00__kill_diff_last_3s`: contribution `-0.007387`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.007956`
- `lag_07__T2__flash_duration`: contribution `-0.005205`
- `lag_03__T_active_infernos`: contribution `-0.004384`

### tick `185179`, seconds `31.50`, LSTM delta `+0.0811`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008322`
- `lag_00__kill_diff_last_3s`: contribution `+0.007387`
- `lag_03__T2__flash_duration`: contribution `+0.005424`
- `lag_00__damage_diff_last_5s`: contribution `+0.005080`
- `lag_00__CT3__is_walking`: contribution `-0.004088`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.005424`
- `lag_00__T4__utility_total`: contribution `+0.001316`

### tick `185883`, seconds `42.50`, LSTM delta `+0.0678`

Top all feature movements:
- `lag_00__CT_place_CATWALK`: contribution `+0.007971`
- `lag_00__kill_diff_last_3s`: contribution `+0.007387`
- `lag_13__T2__flash_duration`: contribution `+0.004550`
- `lag_06__T_shots_fired_sum`: contribution `+0.004473`
- `lag_05__T1__duck_amount`: contribution `-0.004131`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.004550`
