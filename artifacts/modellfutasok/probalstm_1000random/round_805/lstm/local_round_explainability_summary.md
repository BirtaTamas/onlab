# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-saw-vs-big-bo3-Eh5yMCium2D2NNwnLk7jHb/saw-vs-big-m1-ancient.csv`
- round_num: `2`

## Largest probability jumps

- tick `28659`, seconds `54.50`, LSTM `0.1155`, delta `-0.1939`
- tick `28627`, seconds `54.00`, LSTM `0.3094`, delta `+0.1591`
- tick `28723`, seconds `55.50`, LSTM `0.0585`, delta `-0.0822`
- tick `26867`, seconds `26.50`, LSTM `0.3803`, delta `+0.0733`
- tick `28435`, seconds `51.00`, LSTM `0.2009`, delta `-0.0604`
- tick `25235`, seconds `1.00`, LSTM `0.2436`, delta `-0.0580`
- tick `28115`, seconds `46.00`, LSTM `0.3193`, delta `-0.0575`
- tick `25843`, seconds `10.50`, LSTM `0.2887`, delta `+0.0550`
- tick `25203`, seconds `0.50`, LSTM `0.3016`, delta `-0.0392`
- tick `28211`, seconds `47.50`, LSTM `0.3411`, delta `+0.0381`

## Top 15 local ridge features

- `lag_04__T_flashed_players`: coefficient `0.001546`, |coef| `0.001546`
- `lag_02__CT_place_HOUSE`: coefficient `-0.001288`, |coef| `0.001288`
- `lag_01__T1__flash_duration`: coefficient `0.001263`, |coef| `0.001263`
- `lag_13__T4__duck_amount`: coefficient `0.001254`, |coef| `0.001254`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001230`, |coef| `0.001230`
- `lag_03__CT1__is_walking`: coefficient `0.001163`, |coef| `0.001163`
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001142`, |coef| `0.001142`
- `lag_05__T_flashed_players`: coefficient `-0.001127`, |coef| `0.001127`
- `lag_00__CT2__is_walking`: coefficient `-0.001077`, |coef| `0.001077`
- `lag_10__CT3__is_walking`: coefficient `0.001074`, |coef| `0.001074`
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.001072`, |coef| `0.001072`
- `lag_01__CT1__is_walking`: coefficient `0.001059`, |coef| `0.001059`
- `lag_06__CT_place_HOUSE`: coefficient `-0.001035`, |coef| `0.001035`
- `lag_00__CT_place_TSIDELOWER`: coefficient `0.001033`, |coef| `0.001033`
- `lag_00__T_flashed_players`: coefficient `-0.001013`, |coef| `0.001013`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `0.001263` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001142` (lowers CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000928` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000862` (lowers CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000834` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.000804` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `-0.000772` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `0.000740` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000739` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.000716` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_flashed_players`: coefficient `0.001546` (raises CT win probability)
- `lag_02__CT_place_HOUSE`: coefficient `-0.001288` (lowers CT win probability)
- `lag_13__T4__duck_amount`: coefficient `0.001254` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001230` (raises CT win probability)
- `lag_03__CT1__is_walking`: coefficient `0.001163` (raises CT win probability)
- `lag_05__T_flashed_players`: coefficient `-0.001127` (lowers CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001077` (lowers CT win probability)
- `lag_10__CT3__is_walking`: coefficient `0.001074` (raises CT win probability)
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.001072` (lowers CT win probability)
- `lag_01__CT1__is_walking`: coefficient `0.001059` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `28659`, seconds `54.50`, LSTM delta `-0.1939`

Top all feature movements:
- `lag_05__T_flashed_players`: contribution `-0.008700`
- `lag_00__T_shots_fired_sum`: contribution `-0.006147`
- `lag_01__T1__flash_duration`: contribution `-0.005744`
- `lag_01__T_flashed_players`: contribution `-0.004787`
- `lag_13__T4__duck_amount`: contribution `-0.004638`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.005744`
- `lag_00__CT2__flash_duration`: contribution `-0.004088`
- `lag_07__T_A_site_active_infernos`: contribution `-0.002298`

### tick `28627`, seconds `54.00`, LSTM delta `+0.1591`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `+0.011931`
- `lag_00__T_flashed_players`: contribution `+0.005863`
- `lag_13__T4__duck_amount`: contribution `+0.004638`
- `lag_02__CT_place_HOUSE`: contribution `+0.004549`
- `lag_01__T1__flash_duration`: contribution `+0.003871`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `+0.003871`
- `lag_01__CT2__flash_duration`: contribution `+0.003544`
- `lag_00__T1__flash_duration`: contribution `+0.003257`
- `lag_05__T_A_site_active_infernos`: contribution `+0.002204`
- `lag_02__T_A_site_active_infernos`: contribution `+0.002103`

### tick `28723`, seconds `55.50`, LSTM delta `-0.0822`

Top all feature movements:
- `lag_03__T_flashed_players`: contribution `-0.005009`
- `lag_02__T_shots_fired_sum`: contribution `-0.004725`
- `lag_06__CT_place_HOUSE`: contribution `+0.003658`
- `lag_07__T_flashed_players`: contribution `-0.003330`
- `lag_00__T_shots_fired_sum`: contribution `-0.003074`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.002204`
- `lag_03__T1__flash_duration`: contribution `-0.001938`
- `lag_02__CT2__flash_duration`: contribution `+0.001835`
- `lag_09__T_A_site_active_infernos`: contribution `-0.001451`

### tick `26867`, seconds `26.50`, LSTM delta `+0.0733`

Top all feature movements:
- `lag_07__CT_place_SIDEHALL`: contribution `+0.004585`
- `lag_15__T5__duck_amount`: contribution `+0.002771`
- `lag_03__CT1__is_walking`: contribution `+0.002716`
- `lag_00__CT2__is_walking`: contribution `+0.002543`
- `lag_13__T4__is_walking`: contribution `+0.002285`

Top utility-only movements:
- `lag_00__CT_B_site_active_smokes`: contribution `+0.001385`
- `lag_04__CT1__smoke`: contribution `+0.001036`

### tick `28435`, seconds `51.00`, LSTM delta `-0.0604`

Top all feature movements:
- `lag_06__CT_place_HOUSE`: contribution `-0.003658`
- `lag_00__T_A_site_active_infernos`: contribution `-0.003400`
- `lag_03__CT_place_HOUSE`: contribution `-0.002992`
- `lag_00__CT2__is_walking`: contribution `+0.002543`
- `lag_06__CT_place_ALLEY`: contribution `-0.002308`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.003400`
- `lag_00__T_active_infernos`: contribution `-0.001795`
- `lag_10__T_A_site_active_infernos`: contribution `-0.001537`
- `lag_00__T_B_site_active_infernos`: contribution `-0.001417`
- `lag_05__CT_B_site_active_smokes`: contribution `-0.001189`
