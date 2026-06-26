# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `8`

## Largest probability jumps

- tick `66538`, seconds `22.50`, LSTM `0.8882`, delta `+0.1323`
- tick `65898`, seconds `12.50`, LSTM `0.7422`, delta `+0.1288`
- tick `67658`, seconds `40.00`, LSTM `0.9397`, delta `+0.0438`
- tick `65930`, seconds `13.00`, LSTM `0.7078`, delta `-0.0344`
- tick `66026`, seconds `14.50`, LSTM `0.7455`, delta `+0.0310`
- tick `65994`, seconds `14.00`, LSTM `0.7144`, delta `+0.0275`
- tick `65386`, seconds `4.50`, LSTM `0.5731`, delta `-0.0252`
- tick `66122`, seconds `16.00`, LSTM `0.7689`, delta `+0.0238`
- tick `69130`, seconds `63.00`, LSTM `0.9727`, delta `+0.0225`
- tick `66090`, seconds `15.50`, LSTM `0.7451`, delta `-0.0216`

## Top 15 local ridge features

- `lag_12__T_he_last_5s`: coefficient `-0.000965`, |coef| `0.000965`
- `lag_01__CT_place_JUNGLE`: coefficient `0.000939`, |coef| `0.000939`
- `lag_13__CT_place_TRUCK`: coefficient `0.000812`, |coef| `0.000812`
- `lag_00__CT_kills_last_3s`: coefficient `0.000781`, |coef| `0.000781`
- `lag_05__CT_place_UNDERPASS`: coefficient `0.000773`, |coef| `0.000773`
- `lag_12__T4__flash_duration`: coefficient `0.000703`, |coef| `0.000703`
- `lag_09__CT3__flash_duration`: coefficient `-0.000681`, |coef| `0.000681`
- `lag_09__T3__flash_duration`: coefficient `-0.000668`, |coef| `0.000668`
- `lag_00__kill_diff_last_3s`: coefficient `0.000651`, |coef| `0.000651`
- `lag_09__T5__flash_duration`: coefficient `-0.000620`, |coef| `0.000620`
- `lag_07__CT1__is_scoped`: coefficient `0.000592`, |coef| `0.000592`
- `lag_00__CT4__is_scoped`: coefficient `0.000591`, |coef| `0.000591`
- `lag_06__T_he_last_5s`: coefficient `-0.000585`, |coef| `0.000585`
- `lag_04__T4__flash_duration`: coefficient `0.000584`, |coef| `0.000584`
- `lag_12__CT2__shots_fired`: coefficient `-0.000543`, |coef| `0.000543`

## Top 10 utility ridge features

- `lag_12__T_he_last_5s`: coefficient `-0.000965` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.000703` (raises CT win probability)
- `lag_09__CT3__flash_duration`: coefficient `-0.000681` (lowers CT win probability)
- `lag_09__T3__flash_duration`: coefficient `-0.000668` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `-0.000620` (lowers CT win probability)
- `lag_06__T_he_last_5s`: coefficient `-0.000585` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.000584` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.000534` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000513` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.000506` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_JUNGLE`: coefficient `0.000939` (raises CT win probability)
- `lag_13__CT_place_TRUCK`: coefficient `0.000812` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000781` (raises CT win probability)
- `lag_05__CT_place_UNDERPASS`: coefficient `0.000773` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000651` (raises CT win probability)
- `lag_07__CT1__is_scoped`: coefficient `0.000592` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `0.000591` (raises CT win probability)
- `lag_12__CT2__shots_fired`: coefficient `-0.000543` (lowers CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `0.000541` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `0.000540` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `66538`, seconds `22.50`, LSTM delta `+0.1323`

Top all feature movements:
- `lag_01__CT_place_JUNGLE`: contribution `+0.006021`
- `lag_13__CT_place_TRUCK`: contribution `+0.005235`
- `lag_05__CT_place_UNDERPASS`: contribution `+0.004483`
- `lag_12__CT_shots_fired_sum`: contribution `+0.003829`
- `lag_09__T3__flash_duration`: contribution `+0.003615`

Top utility-only movements:
- `lag_09__T3__flash_duration`: contribution `+0.003615`
- `lag_09__CT3__flash_duration`: contribution `+0.003505`
- `lag_09__T5__flash_duration`: contribution `+0.003039`
- `lag_15__CT4__flash_duration`: contribution `+0.002330`
- `lag_09__T_flash_duration_sum`: contribution `+0.002179`

### tick `65898`, seconds `12.50`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_12__T_he_last_5s`: contribution `+0.012597`
- `lag_15__CT_place_SHOP`: contribution `+0.005426`
- `lag_10__CT5__flash_duration`: contribution `+0.003353`
- `lag_12__T4__flash_duration`: contribution `+0.003346`
- `lag_10__CT_place_SNIPERSNEST`: contribution `+0.002542`

Top utility-only movements:
- `lag_12__T_he_last_5s`: contribution `+0.012597`
- `lag_10__CT5__flash_duration`: contribution `+0.003353`
- `lag_12__T4__flash_duration`: contribution `+0.003346`
- `lag_03__T3__flash_duration`: contribution `+0.002442`
- `lag_03__CT4__flash_duration`: contribution `+0.001617`

### tick `67658`, seconds `40.00`, LSTM delta `+0.0438`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.003180`
- `lag_00__CT_kills_last_3s`: contribution `+0.002255`
- `lag_05__T1__flash_duration`: contribution `+0.001995`
- `lag_00__T1__flash_duration`: contribution `+0.001812`
- `lag_00__kill_diff_last_3s`: contribution `+0.001567`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.001995`
- `lag_00__T1__flash_duration`: contribution `+0.001812`

### tick `65930`, seconds `13.00`, LSTM delta `-0.0344`

Top all feature movements:
- `lag_13__T_he_last_5s`: contribution `-0.003809`
- `lag_04__T4__flash_duration`: contribution `-0.002779`
- `lag_00__CT_shots_fired_sum`: contribution `-0.001715`
- `lag_04__T5__duck_amount`: contribution `-0.001483`
- `lag_11__CT_place_SHOP`: contribution `+0.001408`

Top utility-only movements:
- `lag_13__T_he_last_5s`: contribution `-0.003809`
- `lag_04__T4__flash_duration`: contribution `-0.002779`
- `lag_13__T4__flash_duration`: contribution `+0.001237`
- `lag_00__T4__flash_duration`: contribution `+0.001075`
- `lag_02__CT_active_infernos`: contribution `-0.000908`

### tick `66026`, seconds `14.50`, LSTM delta `+0.0310`

Top all feature movements:
- `lag_15__CT_place_SHOP`: contribution `-0.002713`
- `lag_00__T2__duck_amount`: contribution `-0.002066`
- `lag_09__CT3__flash_duration`: contribution `+0.001770`
- `lag_07__CT_place_TRUCK`: contribution `+0.001612`
- `lag_01__T4__flash_duration`: contribution `+0.001593`

Top utility-only movements:
- `lag_09__CT3__flash_duration`: contribution `+0.001770`
- `lag_01__T4__flash_duration`: contribution `+0.001593`
- `lag_14__CT5__flash_duration`: contribution `+0.001325`
- `lag_02__CT5__flash_duration`: contribution `+0.001237`
- `lag_07__T4__flash_duration`: contribution `-0.001205`
