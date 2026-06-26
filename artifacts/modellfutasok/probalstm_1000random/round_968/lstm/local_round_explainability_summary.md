# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `19`

## Largest probability jumps

- tick `171426`, seconds `27.00`, LSTM `0.1661`, delta `-0.3649`
- tick `176674`, seconds `109.00`, LSTM `0.0826`, delta `-0.2314`
- tick `170658`, seconds `15.00`, LSTM `0.3013`, delta `-0.1943`
- tick `171618`, seconds `30.00`, LSTM `0.2307`, delta `+0.1872`
- tick `174594`, seconds `76.50`, LSTM `0.2811`, delta `-0.1140`
- tick `171458`, seconds `27.50`, LSTM `0.0617`, delta `-0.1045`
- tick `170946`, seconds `19.50`, LSTM `0.4637`, delta `+0.1033`
- tick `170914`, seconds `19.00`, LSTM `0.3604`, delta `+0.0762`
- tick `171682`, seconds `31.00`, LSTM `0.3436`, delta `+0.0708`
- tick `170978`, seconds `20.00`, LSTM `0.5293`, delta `+0.0656`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005422`, |coef| `0.005422`
- `lag_00__kill_diff_last_3s`: coefficient `0.004576`, |coef| `0.004576`
- `lag_00__damage_diff_last_5s`: coefficient `0.004389`, |coef| `0.004389`
- `lag_00__T_damage_last_5s`: coefficient `-0.004175`, |coef| `0.004175`
- `lag_02__CT_flashed_players`: coefficient `-0.004150`, |coef| `0.004150`
- `lag_13__CT_place_BRICKS`: coefficient `0.003756`, |coef| `0.003756`
- `lag_05__CT5__is_walking`: coefficient `0.003130`, |coef| `0.003130`
- `lag_00__CT3__alive`: coefficient `0.003103`, |coef| `0.003103`
- `lag_00__CT3__hp`: coefficient `0.003060`, |coef| `0.003060`
- `lag_00__CT3__armor`: coefficient `0.002934`, |coef| `0.002934`
- `lag_00__CT3__has_defuser`: coefficient `0.002904`, |coef| `0.002904`
- `lag_00__CT_flashed_players`: coefficient `0.002757`, |coef| `0.002757`
- `lag_01__T_kills_last_3s`: coefficient `-0.002691`, |coef| `0.002691`
- `lag_13__T5__is_walking`: coefficient `0.002634`, |coef| `0.002634`
- `lag_00__CT3__has_helmet`: coefficient `0.002583`, |coef| `0.002583`

## Top 10 utility ridge features

- `lag_05__CT5__flash`: coefficient `0.001888` (raises CT win probability)
- `lag_07__T5__flash_duration`: coefficient `0.001765` (raises CT win probability)
- `lag_08__T1__flash_duration`: coefficient `0.001606` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001575` (lowers CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.001237` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.001175` (lowers CT win probability)
- `lag_14__active_infernos_total`: coefficient `0.001109` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001008` (raises CT win probability)
- `lag_05__CT5__utility_total`: coefficient `0.001003` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000931` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005422` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004576` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004389` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.004175` (lowers CT win probability)
- `lag_02__CT_flashed_players`: coefficient `-0.004150` (lowers CT win probability)
- `lag_13__CT_place_BRICKS`: coefficient `0.003756` (raises CT win probability)
- `lag_05__CT5__is_walking`: coefficient `0.003130` (raises CT win probability)
- `lag_00__CT3__alive`: coefficient `0.003103` (raises CT win probability)
- `lag_00__CT3__hp`: coefficient `0.003060` (raises CT win probability)
- `lag_00__CT3__armor`: coefficient `0.002934` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `171426`, seconds `27.00`, LSTM delta `-0.3649`

Top all feature movements:
- `lag_13__CT_place_BRICKS`: contribution `-0.072116`
- `lag_12__T_shots_fired_sum`: contribution `-0.020376`
- `lag_12__T1__shots_fired`: contribution `-0.017316`
- `lag_00__T_kills_last_3s`: contribution `-0.017179`
- `lag_07__T5__flash_duration`: contribution `-0.012465`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `-0.012465`
- `lag_08__T1__flash_duration`: contribution `-0.009750`
- `lag_14__T_active_infernos`: contribution `-0.005153`

### tick `176674`, seconds `109.00`, LSTM delta `-0.2314`

Top all feature movements:
- `lag_02__CT_flashed_players`: contribution `-0.018177`
- `lag_00__T_kills_last_3s`: contribution `-0.017179`
- `lag_00__kill_diff_last_3s`: contribution `-0.011013`
- `lag_00__T_damage_last_5s`: contribution `-0.010010`
- `lag_00__damage_diff_last_5s`: contribution `-0.009903`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `170658`, seconds `15.00`, LSTM delta `-0.1943`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.017179`
- `lag_00__kill_diff_last_3s`: contribution `-0.011013`
- `lag_00__CT_place_BRIDGE`: contribution `-0.008218`
- `lag_00__T_damage_last_5s`: contribution `-0.007508`
- `lag_00__damage_diff_last_5s`: contribution `-0.007427`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.003208`
- `lag_08__CT1__flash_duration`: contribution `-0.002853`

### tick `171618`, seconds `30.00`, LSTM delta `+0.1872`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.022027`
- `lag_00__T_kills_last_3s`: contribution `+0.017179`
- `lag_00__damage_diff_last_5s`: contribution `+0.009506`
- `lag_13__T5__flash_duration`: contribution `+0.008298`
- `lag_05__CT5__is_walking`: contribution `+0.007501`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.008298`
- `lag_14__T1__flash_duration`: contribution `+0.005165`

### tick `174594`, seconds `76.50`, LSTM delta `-0.1140`

Top all feature movements:
- `lag_01__T_kills_last_3s`: contribution `-0.008527`
- `lag_07__T_place_TSTAIRS`: contribution `-0.006493`
- `lag_01__CT_place_LOWERTUNNEL`: contribution `-0.005926`
- `lag_12__CT_place_CTSIDEUPPER`: contribution `-0.005216`
- `lag_02__T_shots_fired_sum`: contribution `-0.004946`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `-0.002363`
