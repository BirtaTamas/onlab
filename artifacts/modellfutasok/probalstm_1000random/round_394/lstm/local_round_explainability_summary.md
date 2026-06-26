# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `30239`, seconds `90.50`, LSTM `0.1344`, delta `-0.3559`
- tick `30495`, seconds `94.50`, LSTM `0.0164`, delta `-0.1534`
- tick `26047`, seconds `25.00`, LSTM `0.5037`, delta `-0.1234`
- tick `30399`, seconds `93.00`, LSTM `0.1919`, delta `+0.1197`
- tick `30271`, seconds `91.00`, LSTM `0.0817`, delta `-0.0528`
- tick `26079`, seconds `25.50`, LSTM `0.4522`, delta `-0.0515`
- tick `26239`, seconds `28.00`, LSTM `0.4425`, delta `+0.0423`
- tick `25215`, seconds `12.00`, LSTM `0.6886`, delta `+0.0416`
- tick `29983`, seconds `86.50`, LSTM `0.5130`, delta `-0.0387`
- tick `25727`, seconds `20.00`, LSTM `0.6495`, delta `-0.0362`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003064`, |coef| `0.003064`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002668`, |coef| `0.002668`
- `lag_04__T_place_PALACEINTERIOR`: coefficient `-0.002594`, |coef| `0.002594`
- `lag_00__T_place_PALACEINTERIOR`: coefficient `0.002518`, |coef| `0.002518`
- `lag_00__kill_diff_last_3s`: coefficient `0.002443`, |coef| `0.002443`
- `lag_00__T_macro_A`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002442`, |coef| `0.002442`
- `lag_00__T_damage_last_5s`: coefficient `-0.002422`, |coef| `0.002422`
- `lag_12__T_place_PALACEINTERIOR`: coefficient `-0.002389`, |coef| `0.002389`
- `lag_14__T_place_TRAMP`: coefficient `-0.002379`, |coef| `0.002379`
- `lag_00__CT4__alive`: coefficient `0.002370`, |coef| `0.002370`
- `lag_00__CT4__hp`: coefficient `0.002335`, |coef| `0.002335`
- `lag_00__damage_diff_last_5s`: coefficient `0.002236`, |coef| `0.002236`
- `lag_00__CT4__armor`: coefficient `0.002188`, |coef| `0.002188`
- `lag_04__T_place_TRAMP`: coefficient `0.002098`, |coef| `0.002098`

## Top 10 utility ridge features

- `lag_01__CT4__flash`: coefficient `0.001489` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001033` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000821` (raises CT win probability)
- `lag_01__CT4__utility_total`: coefficient `0.000812` (raises CT win probability)
- `lag_02__CT4__flash`: coefficient `0.000803` (raises CT win probability)
- `lag_05__T4__flash_duration`: coefficient `-0.000781` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.000752` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000690` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `-0.000658` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000646` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.003064` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002668` (raises CT win probability)
- `lag_04__T_place_PALACEINTERIOR`: coefficient `-0.002594` (lowers CT win probability)
- `lag_00__T_place_PALACEINTERIOR`: coefficient `0.002518` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002443` (raises CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002442` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002442` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002422` (lowers CT win probability)
- `lag_12__T_place_PALACEINTERIOR`: coefficient `-0.002389` (lowers CT win probability)
- `lag_14__T_place_TRAMP`: coefficient `-0.002379` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `30239`, seconds `90.50`, LSTM delta `-0.3559`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.017119`
- `lag_00__T_kills_last_3s`: contribution `-0.009706`
- `lag_03__CT_place_JUNGLE`: contribution `-0.009600`
- `lag_04__T_place_PALACEINTERIOR`: contribution `-0.008702`
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.008445`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `30495`, seconds `94.50`, LSTM delta `-0.1534`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009706`
- `lag_12__T_place_PALACEINTERIOR`: contribution `-0.008012`
- `lag_11__CT_place_JUNGLE`: contribution `-0.006168`
- `lag_00__kill_diff_last_3s`: contribution `-0.005881`
- `lag_00__T_damage_last_5s`: contribution `-0.005342`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `-0.004336`
- `lag_05__T_flash_duration_sum`: contribution `-0.004217`
- `lag_05__T5__flash_duration`: contribution `-0.002692`

### tick `26047`, seconds `25.00`, LSTM delta `-0.1234`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `-0.017119`
- `lag_00__T_kills_last_3s`: contribution `-0.009706`
- `lag_10__CT_place_STAIRS`: contribution `-0.006370`
- `lag_00__kill_diff_last_3s`: contribution `-0.005881`
- `lag_00__T_damage_last_5s`: contribution `-0.005806`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `-0.002992`
- `lag_01__CT_A_site_active_infernos`: contribution `-0.001698`

### tick `30399`, seconds `93.00`, LSTM delta `+0.1197`

Top all feature movements:
- `lag_04__T_place_PALACEINTERIOR`: contribution `+0.008702`
- `lag_14__T_place_TRAMP`: contribution `+0.006963`
- `lag_02__T_flashed_players`: contribution `+0.006107`
- `lag_00__kill_diff_last_3s`: contribution `+0.005881`
- `lag_02__T5__flash_duration`: contribution `+0.005429`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.005429`
- `lag_02__T_flash_duration_sum`: contribution `+0.004043`

### tick `30271`, seconds `91.00`, LSTM delta `-0.0528`

Top all feature movements:
- `lag_00__T_place_PALACEINTERIOR`: contribution `-0.008445`
- `lag_01__CT_place_JUNGLE`: contribution `+0.008091`
- `lag_15__T_place_TRAMP`: contribution `-0.005364`
- `lag_13__T_place_PALACEINTERIOR`: contribution `-0.004926`
- `lag_01__T5__duck_amount`: contribution `+0.004733`

Top utility-only movements:
- No utility movement among the top local contributors.
