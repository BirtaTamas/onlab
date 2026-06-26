# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-liquid-bo3-5QhocJqNMgLFMdnITD1OfU/mouz-vs-liquid-m3-mirage.csv`
- round_num: `32`

## Largest probability jumps

- tick `253165`, seconds `53.00`, LSTM `0.0420`, delta `-0.0972`
- tick `250669`, seconds `14.00`, LSTM `0.4422`, delta `-0.0764`
- tick `252045`, seconds `35.50`, LSTM `0.0759`, delta `-0.0682`
- tick `251853`, seconds `32.50`, LSTM `0.2062`, delta `-0.0671`
- tick `250733`, seconds `15.00`, LSTM `0.3208`, delta `-0.0657`
- tick `250701`, seconds `14.50`, LSTM `0.3865`, delta `-0.0558`
- tick `251725`, seconds `30.50`, LSTM `0.3144`, delta `+0.0552`
- tick `251021`, seconds `19.50`, LSTM `0.1291`, delta `-0.0547`
- tick `250829`, seconds `16.50`, LSTM `0.2708`, delta `-0.0540`
- tick `250989`, seconds `19.00`, LSTM `0.1838`, delta `-0.0507`

## Top 15 local ridge features

- `lag_13__CT_place_SHOP`: coefficient `0.001086`, |coef| `0.001086`
- `lag_13__CT_place_SIDEALLEY`: coefficient `0.001070`, |coef| `0.001070`
- `lag_14__CT_place_SHOP`: coefficient `0.001048`, |coef| `0.001048`
- `lag_09__T_place_JUNGLE`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_11__T_place_JUNGLE`: coefficient `-0.000916`, |coef| `0.000916`
- `lag_13__CT_place_UNDERPASS`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_12__CT_place_SHOP`: coefficient `0.000889`, |coef| `0.000889`
- `lag_05__T_place_JUNGLE`: coefficient `-0.000849`, |coef| `0.000849`
- `lag_12__T_place_JUNGLE`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_08__T_place_JUNGLE`: coefficient `-0.000848`, |coef| `0.000848`
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.000839`, |coef| `0.000839`
- `lag_02__T_place_JUNGLE`: coefficient `-0.000839`, |coef| `0.000839`
- `lag_06__CT_place_PALACEALLEY`: coefficient `-0.000827`, |coef| `0.000827`
- `lag_10__T_place_JUNGLE`: coefficient `-0.000800`, |coef| `0.000800`
- `lag_13__T5__is_scoped`: coefficient `-0.000779`, |coef| `0.000779`

## Top 10 utility ridge features

- `lag_05__CT3__flash_duration`: coefficient `-0.000462` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000456` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000447` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.000432` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000427` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000411` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.000409` (raises CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.000407` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.000397` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000396` (raises CT win probability)

## Top 10 flash ridge features

- `lag_09__T_flashed_players`: coefficient `-0.000576` (lowers CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.000462` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000456` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.000432` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000427` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.000411` (lowers CT win probability)
- `lag_01__CT3__flash_duration`: coefficient `0.000407` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000396` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.000396` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `0.000378` (raises CT win probability)

## Top 10 smoke ridge features

- `lag_00__CT3__smoke`: coefficient `0.000328` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `-0.000287` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000279` (raises CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `-0.000257` (lowers CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `-0.000250` (lowers CT win probability)
- `lag_10__CT_A_site_active_smokes`: coefficient `-0.000237` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000236` (raises CT win probability)
- `lag_00__T_smoke_inv`: coefficient `0.000236` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.000236` (raises CT win probability)
- `lag_14__T_active_smokes`: coefficient `-0.000234` (lowers CT win probability)

## Top 10 inferno/molotov ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.000706` (raises CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `0.000531` (raises CT win probability)
- `lag_06__CT_shots_fired_sum`: coefficient `0.000519` (raises CT win probability)
- `lag_06__CT3__shots_fired`: coefficient `0.000480` (raises CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.000454` (lowers CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000447` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.000421` (lowers CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `-0.000415` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.000409` (raises CT win probability)
- `lag_00__CT_A_site_active_infernos`: coefficient `0.000397` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_SHOP`: coefficient `0.001086` (raises CT win probability)
- `lag_13__CT_place_SIDEALLEY`: coefficient `0.001070` (raises CT win probability)
- `lag_14__CT_place_SHOP`: coefficient `0.001048` (raises CT win probability)
- `lag_09__T_place_JUNGLE`: coefficient `-0.000979` (lowers CT win probability)
- `lag_11__T_place_JUNGLE`: coefficient `-0.000916` (lowers CT win probability)
- `lag_13__CT_place_UNDERPASS`: coefficient `-0.000897` (lowers CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.000889` (raises CT win probability)
- `lag_05__T_place_JUNGLE`: coefficient `-0.000849` (lowers CT win probability)
- `lag_12__T_place_JUNGLE`: coefficient `-0.000848` (lowers CT win probability)
- `lag_08__T_place_JUNGLE`: coefficient `-0.000848` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `253165`, seconds `53.00`, LSTM delta `-0.0972`

Top all feature movements:
- `lag_13__CT_place_SIDEALLEY`: contribution `-0.019521`
- `lag_13__CT_place_TSPAWN`: contribution `-0.005712`
- `lag_09__CT_place_JUNGLE`: contribution `-0.003159`
- `lag_11__CT4__is_scoped`: contribution `-0.002260`
- `lag_05__T5__is_scoped`: contribution `-0.002141`

Top utility-only movements:
- `lag_12__T4__flash_duration`: contribution `-0.002030`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.001227`

Top flash movements:
- `lag_12__T4__flash_duration`: contribution `-0.002030`

Top smoke movements:
- No smoke movement among the top local contributors.

Top inferno/molotov movements:
- `lag_06__CT_shots_fired_sum`: contribution `-0.001805`
- `lag_06__CT3__shots_fired`: contribution `-0.001728`
- `lag_07__CT_shots_fired_sum`: contribution `-0.001260`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.001227`
- `lag_07__CT3__shots_fired`: contribution `-0.000955`

### tick `250669`, seconds `14.00`, LSTM delta `-0.0764`

Top all feature movements:
- `lag_13__CT_place_SHOP`: contribution `-0.005447`
- `lag_13__CT_place_UNDERPASS`: contribution `-0.005201`
- `lag_12__CT_place_SHOP`: contribution `-0.004458`
- `lag_13__T5__is_scoped`: contribution `-0.003716`
- `lag_00__CT_place_STAIRS`: contribution `-0.002996`

Top utility-only movements:
- `lag_04__CT_A_site_active_infernos`: contribution `-0.001085`

Top flash movements:
- No flash movement among the top local contributors.

Top smoke movements:
- No smoke movement among the top local contributors.

Top inferno/molotov movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.002818`
- `lag_06__CT_shots_fired_sum`: contribution `+0.001805`
- `lag_04__CT3__shots_fired`: contribution `-0.001270`
- `lag_06__CT3__shots_fired`: contribution `+0.001234`
- `lag_04__CT_A_site_active_infernos`: contribution `-0.001085`

### tick `252045`, seconds `35.50`, LSTM delta `-0.0682`

Top all feature movements:
- `lag_02__T_place_JUNGLE`: contribution `-0.010864`
- `lag_06__CT_shots_fired_sum`: contribution `-0.007218`
- `lag_06__CT3__shots_fired`: contribution `-0.004937`
- `lag_02__CT_place_LADDER`: contribution `-0.004510`
- `lag_15__T5__is_scoped`: contribution `-0.002492`

Top utility-only movements:
- `lag_01__CT3__flash_duration`: contribution `-0.002218`
- `lag_11__CT3__flash_duration`: contribution `-0.001378`

Top flash movements:
- `lag_01__CT3__flash_duration`: contribution `-0.002218`
- `lag_11__CT3__flash_duration`: contribution `-0.001378`

Top smoke movements:
- No smoke movement among the top local contributors.

Top inferno/molotov movements:
- `lag_06__CT_shots_fired_sum`: contribution `-0.007218`
- `lag_06__CT3__shots_fired`: contribution `-0.004937`
- `lag_07__CT_shots_fired_sum`: contribution `-0.001575`
- `lag_08__CT_shots_fired_sum`: contribution `-0.001441`
- `lag_10__CT_shots_fired_sum`: contribution `-0.001063`

### tick `251853`, seconds `32.50`, LSTM delta `-0.0671`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.009814`
- `lag_00__CT3__shots_fired`: contribution `-0.005461`
- `lag_14__CT_place_SHOP`: contribution `-0.005255`
- `lag_09__T_flashed_players`: contribution `-0.003334`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.002949`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `-0.002518`
- `lag_15__T2__flash_duration`: contribution `-0.002210`
- `lag_14__CT3__flash_duration`: contribution `-0.002161`
- `lag_01__T5__flash_duration`: contribution `-0.001700`
- `lag_09__T5__flash_duration`: contribution `-0.001073`

Top flash movements:
- `lag_09__T_flashed_players`: contribution `-0.003334`
- `lag_05__CT3__flash_duration`: contribution `-0.002518`
- `lag_15__T2__flash_duration`: contribution `-0.002210`
- `lag_14__CT3__flash_duration`: contribution `-0.002161`
- `lag_01__T5__flash_duration`: contribution `-0.001700`

Top smoke movements:
- No smoke movement among the top local contributors.

Top inferno/molotov movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.009814`
- `lag_00__CT3__shots_fired`: contribution `-0.005461`
- `lag_04__CT_shots_fired_sum`: contribution `+0.001281`
- `lag_03__CT_shots_fired_sum`: contribution `-0.001201`

### tick `250733`, seconds `15.00`, LSTM delta `-0.0657`

Top all feature movements:
- `lag_14__CT_place_SHOP`: contribution `-0.005255`
- `lag_15__CT_place_UNDERPASS`: contribution `-0.004352`
- `lag_06__CT_shots_fired_sum`: contribution `-0.003970`
- `lag_01__CT_place_UNDERPASS`: contribution `-0.003054`
- `lag_06__CT3__shots_fired`: contribution `-0.002715`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.001329`
- `lag_00__T3__flash_duration`: contribution `-0.001306`

Top flash movements:
- `lag_00__T3__flash_duration`: contribution `-0.001306`

Top smoke movements:
- No smoke movement among the top local contributors.

Top inferno/molotov movements:
- `lag_06__CT_shots_fired_sum`: contribution `-0.003970`
- `lag_06__CT3__shots_fired`: contribution `-0.002715`
- `lag_07__CT_shots_fired_sum`: contribution `-0.001575`
- `lag_08__CT_shots_fired_sum`: contribution `-0.001441`
- `lag_07__T_A_site_active_infernos`: contribution `-0.001329`
