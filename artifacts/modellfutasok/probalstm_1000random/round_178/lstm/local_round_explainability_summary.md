# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `66053`, seconds `92.50`, LSTM `0.2469`, delta `-0.2630`
- tick `64997`, seconds `76.00`, LSTM `0.7616`, delta `+0.2298`
- tick `64869`, seconds `74.00`, LSTM `0.5340`, delta `-0.1844`
- tick `64133`, seconds `62.50`, LSTM `0.6556`, delta `+0.1791`
- tick `65189`, seconds `79.00`, LSTM `0.7103`, delta `-0.1258`
- tick `66149`, seconds `94.00`, LSTM `0.0207`, delta `-0.0942`
- tick `66117`, seconds `93.50`, LSTM `0.1149`, delta `-0.0845`
- tick `64677`, seconds `71.00`, LSTM `0.7329`, delta `-0.0554`
- tick `65317`, seconds `81.00`, LSTM `0.6204`, delta `-0.0508`
- tick `65349`, seconds `81.50`, LSTM `0.5720`, delta `-0.0483`

## Top 15 local ridge features

- `lag_03__CT_place_UNDERPASS`: coefficient `0.004354`, |coef| `0.004354`
- `lag_00__kill_diff_last_3s`: coefficient `0.003388`, |coef| `0.003388`
- `lag_00__T_kills_last_3s`: coefficient `-0.003000`, |coef| `0.003000`
- `lag_03__CT_place_CATWALK`: coefficient `-0.002551`, |coef| `0.002551`
- `lag_00__damage_diff_last_5s`: coefficient `0.002418`, |coef| `0.002418`
- `lag_10__CT4__duck_amount`: coefficient `-0.002321`, |coef| `0.002321`
- `lag_02__CT_place_UNDERPASS`: coefficient `0.002303`, |coef| `0.002303`
- `lag_02__T_place_SCAFFOLDING`: coefficient `-0.002094`, |coef| `0.002094`
- `lag_00__T_place_SCAFFOLDING`: coefficient `-0.002093`, |coef| `0.002093`
- `lag_04__CT_place_UNDERPASS`: coefficient `0.001966`, |coef| `0.001966`
- `lag_15__T_place_PALACEINTERIOR`: coefficient `-0.001957`, |coef| `0.001957`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.001951`, |coef| `0.001951`
- `lag_00__CT5__alive`: coefficient `0.001931`, |coef| `0.001931`
- `lag_06__CT4__duck_amount`: coefficient `0.001875`, |coef| `0.001875`
- `lag_02__CT2__duck_amount`: coefficient `-0.001850`, |coef| `0.001850`

## Top 10 utility ridge features

- `lag_01__T1__molly`: coefficient `0.001695` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001693` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.001528` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.001230` (raises CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `0.001180` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.001151` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `0.001124` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.001120` (lowers CT win probability)
- `lag_08__T3__flash`: coefficient `0.001119` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `0.001056` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_UNDERPASS`: coefficient `0.004354` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003388` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003000` (lowers CT win probability)
- `lag_03__CT_place_CATWALK`: coefficient `-0.002551` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002418` (raises CT win probability)
- `lag_10__CT4__duck_amount`: coefficient `-0.002321` (lowers CT win probability)
- `lag_02__CT_place_UNDERPASS`: coefficient `0.002303` (raises CT win probability)
- `lag_02__T_place_SCAFFOLDING`: coefficient `-0.002094` (lowers CT win probability)
- `lag_00__T_place_SCAFFOLDING`: coefficient `-0.002093` (lowers CT win probability)
- `lag_04__CT_place_UNDERPASS`: coefficient `0.001966` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `66053`, seconds `92.50`, LSTM delta `-0.2630`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009505`
- `lag_10__CT4__duck_amount`: contribution `-0.008525`
- `lag_00__kill_diff_last_3s`: contribution `-0.008154`
- `lag_00__CT_place_CONNECTOR`: contribution `-0.006977`
- `lag_06__CT4__duck_amount`: contribution `-0.006888`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.005048`
- `lag_00__CT5__flash_duration`: contribution `-0.004558`
- `lag_01__T1__molly`: contribution `-0.003752`

### tick `64997`, seconds `76.00`, LSTM delta `+0.2298`

Top all feature movements:
- `lag_03__CT_place_UNDERPASS`: contribution `+0.025249`
- `lag_03__CT_place_CATWALK`: contribution `+0.010160`
- `lag_10__CT4__duck_amount`: contribution `+0.008525`
- `lag_00__kill_diff_last_3s`: contribution `+0.008154`
- `lag_10__T2__flash_duration`: contribution `+0.007146`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `+0.007146`
- `lag_09__T_A_site_active_infernos`: contribution `+0.003347`

### tick `64869`, seconds `74.00`, LSTM delta `-0.1844`

Top all feature movements:
- `lag_03__CT_place_UNDERPASS`: contribution `-0.025249`
- `lag_03__CT_place_CATWALK`: contribution `-0.010160`
- `lag_00__T_kills_last_3s`: contribution `-0.009505`
- `lag_00__kill_diff_last_3s`: contribution `-0.008154`
- `lag_06__T2__flash_duration`: contribution `-0.007847`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.007847`

### tick `64133`, seconds `62.50`, LSTM delta `+0.1791`

Top all feature movements:
- `lag_03__CT_place_UNDERPASS`: contribution `+0.025249`
- `lag_03__CT_place_CATWALK`: contribution `+0.010160`
- `lag_11__T5__flash_duration`: contribution `+0.008916`
- `lag_00__kill_diff_last_3s`: contribution `+0.008154`
- `lag_00__T5__flash_duration`: contribution `+0.007348`

Top utility-only movements:
- `lag_11__T5__flash_duration`: contribution `+0.008916`
- `lag_00__T5__flash_duration`: contribution `+0.007348`

### tick `65189`, seconds `79.00`, LSTM delta `-0.1258`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009505`
- `lag_00__kill_diff_last_3s`: contribution `-0.008154`
- `lag_13__CT_place_UNDERPASS`: contribution `-0.004935`
- `lag_14__CT_place_JUNGLE`: contribution `-0.003640`
- `lag_05__T1__duck_amount`: contribution `-0.003513`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.002337`
