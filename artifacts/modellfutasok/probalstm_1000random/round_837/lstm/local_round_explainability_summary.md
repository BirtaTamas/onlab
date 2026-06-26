# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `20`

## Largest probability jumps

- tick `148082`, seconds `65.00`, LSTM `0.2311`, delta `+0.1585`
- tick `148338`, seconds `69.00`, LSTM `0.2362`, delta `-0.1259`
- tick `148370`, seconds `69.50`, LSTM `0.1152`, delta `-0.1210`
- tick `143954`, seconds `0.50`, LSTM `0.1010`, delta `-0.0665`
- tick `145874`, seconds `30.50`, LSTM `0.1664`, delta `-0.0518`
- tick `143986`, seconds `1.00`, LSTM `0.1433`, delta `+0.0423`
- tick `148114`, seconds `65.50`, LSTM `0.2715`, delta `+0.0404`
- tick `148562`, seconds `72.50`, LSTM `0.0139`, delta `-0.0395`
- tick `146002`, seconds `32.50`, LSTM `0.1162`, delta `-0.0391`
- tick `144018`, seconds `1.50`, LSTM `0.1727`, delta `+0.0294`

## Top 15 local ridge features

- `lag_10__CT2__flash_duration`: coefficient `0.001486`, |coef| `0.001486`
- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.001339`, |coef| `0.001339`
- `lag_12__CT_place_TSPAWN`: coefficient `-0.001319`, |coef| `0.001319`
- `lag_11__CT_place_TSPAWN`: coefficient `-0.001295`, |coef| `0.001295`
- `lag_03__CT_place_TSPAWN`: coefficient `0.001247`, |coef| `0.001247`
- `lag_09__CT2__flash_duration`: coefficient `0.001167`, |coef| `0.001167`
- `lag_00__damage_diff_last_5s`: coefficient `0.001037`, |coef| `0.001037`
- `lag_00__CT2__flash_duration`: coefficient `-0.001018`, |coef| `0.001018`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001000`, |coef| `0.001000`
- `lag_08__CT2__flash_duration`: coefficient `0.000933`, |coef| `0.000933`
- `lag_06__CT3__duck_amount`: coefficient `-0.000928`, |coef| `0.000928`
- `lag_01__CT_place_TRUCK`: coefficient `0.000924`, |coef| `0.000924`
- `lag_00__CT_place_TRUCK`: coefficient `0.000903`, |coef| `0.000903`
- `lag_07__CT3__is_walking`: coefficient `0.000889`, |coef| `0.000889`
- `lag_04__CT_place_PALACEINTERIOR`: coefficient `-0.000887`, |coef| `0.000887`

## Top 10 utility ridge features

- `lag_10__CT2__flash_duration`: coefficient `0.001486` (raises CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.001167` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001018` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.000933` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000807` (raises CT win probability)
- `lag_14__CT_smokes_last_5s`: coefficient `-0.000796` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.000762` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000748` (raises CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `0.000688` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000546` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_PALACEINTERIOR`: coefficient `-0.001339` (lowers CT win probability)
- `lag_12__CT_place_TSPAWN`: coefficient `-0.001319` (lowers CT win probability)
- `lag_11__CT_place_TSPAWN`: coefficient `-0.001295` (lowers CT win probability)
- `lag_03__CT_place_TSPAWN`: coefficient `0.001247` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001037` (raises CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001000` (lowers CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `-0.000928` (lowers CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `0.000924` (raises CT win probability)
- `lag_00__CT_place_TRUCK`: coefficient `0.000903` (raises CT win probability)
- `lag_07__CT3__is_walking`: coefficient `0.000889` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `148082`, seconds `65.00`, LSTM delta `+0.1585`

Top all feature movements:
- `lag_03__CT_place_TSPAWN`: contribution `+0.009335`
- `lag_10__CT2__flash_duration`: contribution `+0.008430`
- `lag_00__CT2__flash_duration`: contribution `+0.005772`
- `lag_03__CT_place_PALACEINTERIOR`: contribution `+0.005455`
- `lag_10__T_place_CONNECTOR`: contribution `+0.004177`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.008430`
- `lag_00__CT2__flash_duration`: contribution `+0.005772`
- `lag_10__CT_flash_duration_sum`: contribution `+0.001771`

### tick `148338`, seconds `69.00`, LSTM delta `-0.1259`

Top all feature movements:
- `lag_11__CT_place_TSPAWN`: contribution `-0.009693`
- `lag_00__CT_place_TRUCK`: contribution `-0.005824`
- `lag_08__CT2__flash_duration`: contribution `-0.005292`
- `lag_04__CT_place_TRUCK`: contribution `-0.004955`
- `lag_07__CT_place_SHOP`: contribution `-0.003694`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.005292`

### tick `148370`, seconds `69.50`, LSTM delta `-0.1210`

Top all feature movements:
- `lag_12__CT_place_TSPAWN`: contribution `-0.009875`
- `lag_09__CT2__flash_duration`: contribution `-0.006617`
- `lag_01__CT_place_TRUCK`: contribution `-0.005958`
- `lag_05__CT_place_TRUCK`: contribution `-0.004568`
- `lag_03__T_flashed_players`: contribution `-0.003121`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.006617`
- `lag_00__T3__flash_duration`: contribution `-0.001671`
- `lag_09__CT_flash_duration_sum`: contribution `-0.001304`

### tick `143954`, seconds `0.50`, LSTM delta `-0.0665`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002819`
- `lag_01__T_place_TSPAWN`: contribution `-0.001987`
- `lag_00__T_velocity_mean`: contribution `-0.001089`
- `lag_01__T1__flash`: contribution `-0.001066`
- `lag_01__utility_inv_diff`: contribution `-0.001029`

Top utility-only movements:
- `lag_01__T1__flash`: contribution `-0.001066`
- `lag_01__utility_inv_diff`: contribution `-0.001029`
- `lag_01__flash_inv_diff`: contribution `-0.000894`
- `lag_00__T1__smoke`: contribution `-0.000672`
- `lag_01__molly_inv_diff`: contribution `-0.000570`

### tick `145874`, seconds `30.50`, LSTM delta `-0.0518`

Top all feature movements:
- `lag_03__CT_place_PALACEINTERIOR`: contribution `-0.005455`
- `lag_00__CT1__duck_amount`: contribution `+0.002294`
- `lag_07__CT3__is_walking`: contribution `-0.002122`
- `lag_02__CT_place_APARTMENTS`: contribution `-0.001954`
- `lag_15__T2__duck_amount`: contribution `-0.001816`

Top utility-only movements:
- No utility movement among the top local contributors.
