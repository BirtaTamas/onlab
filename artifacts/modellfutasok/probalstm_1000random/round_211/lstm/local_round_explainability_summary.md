# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `17`

## Largest probability jumps

- tick `151145`, seconds `86.50`, LSTM `0.6717`, delta `+0.1505`
- tick `151241`, seconds `88.00`, LSTM `0.8751`, delta `+0.1072`
- tick `151209`, seconds `87.50`, LSTM `0.7679`, delta `+0.0701`
- tick `145897`, seconds `4.50`, LSTM `0.5517`, delta `+0.0543`
- tick `151657`, seconds `94.50`, LSTM `0.9546`, delta `+0.0450`
- tick `151305`, seconds `89.00`, LSTM `0.9324`, delta `+0.0448`
- tick `145929`, seconds `5.00`, LSTM `0.5071`, delta `-0.0446`
- tick `149961`, seconds `68.00`, LSTM `0.4819`, delta `+0.0336`
- tick `151177`, seconds `87.00`, LSTM `0.6978`, delta `+0.0261`
- tick `150281`, seconds `73.00`, LSTM `0.4687`, delta `-0.0253`

## Top 15 local ridge features

- `lag_09__CT_place_BRICKS`: coefficient `0.002229`, |coef| `0.002229`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001524`, |coef| `0.001524`
- `lag_12__CT_place_BRICKS`: coefficient `0.001272`, |coef| `0.001272`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_11__CT_place_BRICKS`: coefficient `0.001213`, |coef| `0.001213`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001039`, |coef| `0.001039`
- `lag_00__CT_place_BRICKS`: coefficient `-0.001020`, |coef| `0.001020`
- `lag_10__CT_place_BRIDGE`: coefficient `-0.000890`, |coef| `0.000890`
- `lag_04__CT5__flash_duration`: coefficient `0.000837`, |coef| `0.000837`
- `lag_09__CT_place_CTSIDEUPPER`: coefficient `0.000801`, |coef| `0.000801`
- `lag_00__CT_kills_last_3s`: coefficient `0.000792`, |coef| `0.000792`
- `lag_10__CT_place_BRICKS`: coefficient `0.000755`, |coef| `0.000755`
- `lag_08__T5__is_walking`: coefficient `-0.000753`, |coef| `0.000753`
- `lag_12__T_place_CONNECTOR`: coefficient `0.000719`, |coef| `0.000719`
- `lag_06__T5__flash_duration`: coefficient `-0.000718`, |coef| `0.000718`

## Top 10 utility ridge features

- `lag_04__CT5__flash_duration`: coefficient `0.000837` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.000718` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `0.000582` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000570` (raises CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.000557` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `-0.000553` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000472` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000469` (lowers CT win probability)
- `lag_05__T_he_last_5s`: coefficient `0.000463` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.000443` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_BRICKS`: coefficient `0.002229` (raises CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001524` (lowers CT win probability)
- `lag_12__CT_place_BRICKS`: coefficient `0.001272` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001240` (lowers CT win probability)
- `lag_11__CT_place_BRICKS`: coefficient `0.001213` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001039` (raises CT win probability)
- `lag_00__CT_place_BRICKS`: coefficient `-0.001020` (lowers CT win probability)
- `lag_10__CT_place_BRIDGE`: coefficient `-0.000890` (lowers CT win probability)
- `lag_09__CT_place_CTSIDEUPPER`: coefficient `0.000801` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000792` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `151145`, seconds `86.50`, LSTM delta `+0.1505`

Top all feature movements:
- `lag_09__CT_place_BRICKS`: contribution `+0.042799`
- `lag_00__CT_place_BRICKS`: contribution `+0.019588`
- `lag_00__T_place_CONNECTOR`: contribution `+0.006006`
- `lag_04__CT5__flash_duration`: contribution `+0.004691`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003608`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.004691`

### tick `151241`, seconds `88.00`, LSTM delta `+0.1072`

Top all feature movements:
- `lag_12__CT_place_BRICKS`: contribution `+0.024431`
- `lag_03__CT_place_BRICKS`: contribution `+0.013304`
- `lag_00__CT_place_CTSIDEUPPER`: contribution `+0.007872`
- `lag_00__T_place_CONNECTOR`: contribution `+0.006006`
- `lag_04__T_flashed_players`: contribution `+0.004062`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.003193`
- `lag_11__T_B_site_active_infernos`: contribution `+0.001016`

### tick `151209`, seconds `87.50`, LSTM delta `+0.0701`

Top all feature movements:
- `lag_11__CT_place_BRICKS`: contribution `+0.023285`
- `lag_02__CT_place_BRICKS`: contribution `+0.013301`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004329`
- `lag_06__CT5__flash_duration`: contribution `+0.001848`
- `lag_04__T_flashed_players`: contribution `-0.001354`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.001848`
- `lag_10__T_B_site_active_infernos`: contribution `+0.000927`
- `lag_10__CT_B_site_active_infernos`: contribution `+0.000718`

### tick `145897`, seconds `4.50`, LSTM delta `+0.0543`

Top all feature movements:
- `lag_09__CT_place_CTSIDEUPPER`: contribution `+0.020633`
- `lag_05__T_he_last_5s`: contribution `+0.006043`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.004036`
- `lag_05__CT_place_LOWERTUNNEL`: contribution `+0.002980`
- `lag_01__CT_place_PALACEINTERIOR`: contribution `+0.002493`

Top utility-only movements:
- `lag_05__T_he_last_5s`: contribution `+0.006043`
- `lag_09__CT4__molly`: contribution `-0.000493`
- `lag_09__T3__flash`: contribution `+0.000464`
- `lag_00__T2__smoke`: contribution `+0.000447`

### tick `151657`, seconds `94.50`, LSTM delta `+0.0450`

Top all feature movements:
- `lag_01__CT_place_TUNNEL`: contribution `+0.007278`
- `lag_03__CT_place_TUNNELSTAIRS`: contribution `+0.005787`
- `lag_01__CT_place_TUNNELSTAIRS`: contribution `+0.003972`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003608`
- `lag_00__CT_kills_last_3s`: contribution `+0.002286`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.001965`
- `lag_10__T_B_site_active_infernos`: contribution `-0.000927`
