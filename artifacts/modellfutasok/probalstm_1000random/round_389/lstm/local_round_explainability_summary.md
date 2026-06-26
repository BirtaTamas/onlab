# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `67077`, seconds `48.00`, LSTM `0.0635`, delta `-0.5832`
- tick `66757`, seconds `43.00`, LSTM `0.5195`, delta `+0.2773`
- tick `66405`, seconds `37.50`, LSTM `0.2814`, delta `-0.2199`
- tick `66725`, seconds `42.50`, LSTM `0.2422`, delta `+0.1953`
- tick `65605`, seconds `25.00`, LSTM `0.5329`, delta `-0.1699`
- tick `66213`, seconds `34.50`, LSTM `0.4203`, delta `+0.1649`
- tick `65669`, seconds `26.00`, LSTM `0.3085`, delta `-0.1294`
- tick `66437`, seconds `38.00`, LSTM `0.1840`, delta `-0.0974`
- tick `65637`, seconds `25.50`, LSTM `0.4379`, delta `-0.0950`
- tick `66981`, seconds `46.50`, LSTM `0.6820`, delta `+0.0835`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004909`, |coef| `0.004909`
- `lag_00__damage_diff_last_5s`: coefficient `0.004430`, |coef| `0.004430`
- `lag_00__T_kills_last_3s`: coefficient `-0.003944`, |coef| `0.003944`
- `lag_00__T1__is_scoped`: coefficient `0.003934`, |coef| `0.003934`
- `lag_03__T1__is_scoped`: coefficient `-0.003142`, |coef| `0.003142`
- `lag_12__T_bomb_zone_count`: coefficient `-0.003026`, |coef| `0.003026`
- `lag_13__CT_place_LONGA`: coefficient `0.003013`, |coef| `0.003013`
- `lag_01__damage_diff_last_5s`: coefficient `0.002956`, |coef| `0.002956`
- `lag_00__CT_damage_last_5s`: coefficient `0.002913`, |coef| `0.002913`
- `lag_10__T_shots_fired_sum`: coefficient `0.002911`, |coef| `0.002911`
- `lag_11__T5__shots_fired`: coefficient `-0.002754`, |coef| `0.002754`
- `lag_07__CT_place_LONGA`: coefficient `0.002435`, |coef| `0.002435`
- `lag_01__CT_damage_last_5s`: coefficient `0.002411`, |coef| `0.002411`
- `lag_10__CT_kills_last_3s`: coefficient `-0.002320`, |coef| `0.002320`
- `lag_00__CT_kills_last_3s`: coefficient `0.002294`, |coef| `0.002294`

## Top 10 utility ridge features

- `lag_00__CT5__molly`: coefficient `0.001650` (raises CT win probability)
- `lag_11__T2__smoke`: coefficient `0.001313` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.001078` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001024` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.000931` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000912` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000876` (raises CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.000868` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `-0.000812` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000806` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004909` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004430` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003944` (lowers CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.003934` (raises CT win probability)
- `lag_03__T1__is_scoped`: coefficient `-0.003142` (lowers CT win probability)
- `lag_12__T_bomb_zone_count`: coefficient `-0.003026` (lowers CT win probability)
- `lag_13__CT_place_LONGA`: coefficient `0.003013` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.002956` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002913` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `0.002911` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `67077`, seconds `48.00`, LSTM delta `-0.5832`

Top all feature movements:
- `lag_03__T_place_HOLE`: contribution `-0.044319`
- `lag_00__T_kills_last_3s`: contribution `-0.024988`
- `lag_00__kill_diff_last_3s`: contribution `-0.023630`
- `lag_00__T1__is_scoped`: contribution `-0.022473`
- `lag_10__T_shots_fired_sum`: contribution `-0.019639`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66757`, seconds `43.00`, LSTM delta `+0.2773`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.014554`
- `lag_13__CT_place_ARAMP`: contribution `+0.014114`
- `lag_00__kill_diff_last_3s`: contribution `+0.011815`
- `lag_14__T_place_TUNNELSTAIRS`: contribution `+0.010267`
- `lag_00__damage_diff_last_5s`: contribution `+0.008795`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66405`, seconds `37.50`, LSTM delta `-0.2199`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.023630`
- `lag_09__CT_place_HOLE`: contribution `-0.017374`
- `lag_00__T_kills_last_3s`: contribution `-0.012494`
- `lag_11__CT_place_HOLE`: contribution `-0.011180`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `-0.008575`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66725`, seconds `42.50`, LSTM delta `+0.1953`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011815`
- `lag_12__CT_place_ARAMP`: contribution `+0.011741`
- `lag_00__damage_diff_last_5s`: contribution `+0.010994`
- `lag_10__kill_diff_last_3s`: contribution `+0.010075`
- `lag_00__T_shots_fired_sum`: contribution `-0.008085`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65605`, seconds `25.00`, LSTM delta `-0.1699`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.012494`
- `lag_00__kill_diff_last_3s`: contribution `-0.011815`
- `lag_00__damage_diff_last_5s`: contribution `-0.009995`
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.007524`
- `lag_14__CT_place_SHORTSTAIRS`: contribution `-0.006274`

Top utility-only movements:
- No utility movement among the top local contributors.
