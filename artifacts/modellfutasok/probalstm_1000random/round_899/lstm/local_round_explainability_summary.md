# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `34`

## Largest probability jumps

- tick `284071`, seconds `31.50`, LSTM `0.2579`, delta `-0.2847`
- tick `283975`, seconds `30.00`, LSTM `0.5362`, delta `-0.1066`
- tick `284103`, seconds `32.00`, LSTM `0.1702`, delta `-0.0877`
- tick `283943`, seconds `29.50`, LSTM `0.6429`, delta `+0.0718`
- tick `284263`, seconds `34.50`, LSTM `0.0265`, delta `-0.0542`
- tick `284135`, seconds `32.50`, LSTM `0.1234`, delta `-0.0468`
- tick `284199`, seconds `33.50`, LSTM `0.1048`, delta `-0.0281`
- tick `282087`, seconds `0.50`, LSTM `0.5267`, delta `+0.0275`
- tick `282887`, seconds `13.00`, LSTM `0.5242`, delta `+0.0252`
- tick `282823`, seconds `12.00`, LSTM `0.4988`, delta `-0.0242`

## Top 15 local ridge features

- `lag_10__T_place_SCAFFOLDING`: coefficient `0.001617`, |coef| `0.001617`
- `lag_04__CT_place_LADDER`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001312`, |coef| `0.001312`
- `lag_12__CT_place_SHOP`: coefficient `0.000970`, |coef| `0.000970`
- `lag_08__T_place_SCAFFOLDING`: coefficient `0.000925`, |coef| `0.000925`
- `lag_13__T_flashes_last_5s`: coefficient `-0.000820`, |coef| `0.000820`
- `lag_00__CT2__shots_fired`: coefficient `0.000819`, |coef| `0.000819`
- `lag_04__T2__flash_duration`: coefficient `0.000788`, |coef| `0.000788`
- `lag_05__CT_place_LADDER`: coefficient `-0.000781`, |coef| `0.000781`
- `lag_04__CT5__shots_fired`: coefficient `-0.000773`, |coef| `0.000773`
- `lag_03__CT_flashed_players`: coefficient `0.000748`, |coef| `0.000748`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000730`, |coef| `0.000730`
- `lag_12__T_place_SCAFFOLDING`: coefficient `-0.000717`, |coef| `0.000717`
- `lag_10__CT_place_LADDER`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.000710`, |coef| `0.000710`

## Top 10 utility ridge features

- `lag_13__T_flashes_last_5s`: coefficient `-0.000820` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.000788` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000676` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `-0.000650` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000649` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000647` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000634` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `-0.000631` (lowers CT win probability)
- `lag_00__CT2__molly`: coefficient `0.000594` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000582` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__T_place_SCAFFOLDING`: coefficient `0.001617` (raises CT win probability)
- `lag_04__CT_place_LADDER`: coefficient `-0.001460` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001312` (raises CT win probability)
- `lag_12__CT_place_SHOP`: coefficient `0.000970` (raises CT win probability)
- `lag_08__T_place_SCAFFOLDING`: coefficient `0.000925` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.000819` (raises CT win probability)
- `lag_05__CT_place_LADDER`: coefficient `-0.000781` (lowers CT win probability)
- `lag_04__CT5__shots_fired`: coefficient `-0.000773` (lowers CT win probability)
- `lag_03__CT_flashed_players`: coefficient `0.000748` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000730` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `284071`, seconds `31.50`, LSTM delta `-0.2847`

Top all feature movements:
- `lag_10__T_place_SCAFFOLDING`: contribution `-0.055075`
- `lag_12__T_place_SCAFFOLDING`: contribution `-0.024402`
- `lag_04__CT_place_LADDER`: contribution `-0.015181`
- `lag_00__CT_shots_fired_sum`: contribution `-0.010939`
- `lag_04__CT_shots_fired_sum`: contribution `-0.004931`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `-0.004521`
- `lag_06__T_flash_duration_sum`: contribution `-0.003483`
- `lag_06__T2__flash_duration`: contribution `-0.002585`
- `lag_06__T4__flash_duration`: contribution `-0.002494`

### tick `283975`, seconds `30.00`, LSTM delta `-0.1066`

Top all feature movements:
- `lag_07__T_place_SCAFFOLDING`: contribution `-0.019621`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008204`
- `lag_01__CT_place_LADDER`: contribution `-0.006422`
- `lag_09__T_place_SCAFFOLDING`: contribution `+0.003941`
- `lag_13__CT3__flash_duration`: contribution `-0.002507`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.002507`
- `lag_03__CT5__flash_duration`: contribution `-0.002049`
- `lag_13__CT_flash_duration_sum`: contribution `-0.001762`

### tick `284103`, seconds `32.00`, LSTM delta `-0.0877`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `-0.014112`
- `lag_05__CT_place_LADDER`: contribution `-0.008124`
- `lag_00__T_shots_fired_sum`: contribution `-0.005470`
- `lag_04__CT_shots_fired_sum`: contribution `+0.004438`
- `lag_11__T_place_SCAFFOLDING`: contribution `-0.004434`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.004063`
- `lag_07__T_flash_duration_sum`: contribution `-0.001319`
- `lag_07__T4__flash_duration`: contribution `-0.001262`
- `lag_00__CT_flash_duration_sum`: contribution `-0.001235`

### tick `283943`, seconds `29.50`, LSTM delta `+0.0718`

Top all feature movements:
- `lag_08__T_place_SCAFFOLDING`: contribution `+0.031491`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009116`
- `lag_12__CT_place_SHOP`: contribution `+0.004868`
- `lag_06__T_place_SCAFFOLDING`: contribution `+0.003039`
- `lag_08__T_place_PALACEINTERIOR`: contribution `+0.002268`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.001605`
- `lag_12__CT5__flash_duration`: contribution `+0.001188`
- `lag_13__CT5__flash_duration`: contribution `+0.001068`

### tick `284263`, seconds `34.50`, LSTM delta `-0.0542`

Top all feature movements:
- `lag_10__CT_place_LADDER`: contribution `-0.007418`
- `lag_02__T_flashes_last_5s`: contribution `-0.003245`
- `lag_00__CT_place_JUNGLE`: contribution `-0.003197`
- `lag_06__CT_shots_fired_sum`: contribution `+0.002630`
- `lag_05__T_shots_fired_sum`: contribution `-0.002416`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `-0.003245`
- `lag_05__CT3__flash_duration`: contribution `-0.001742`
- `lag_10__T2__flash_duration`: contribution `+0.001539`
- `lag_12__T2__flash_duration`: contribution `-0.001313`
- `lag_12__T_flash_duration_sum`: contribution `-0.001259`
