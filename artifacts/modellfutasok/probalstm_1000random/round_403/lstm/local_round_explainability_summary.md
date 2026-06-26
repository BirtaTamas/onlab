# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `16`

## Largest probability jumps

- tick `126432`, seconds `22.50`, LSTM `0.8342`, delta `+0.0911`
- tick `126496`, seconds `23.50`, LSTM `0.8880`, delta `+0.0677`
- tick `127264`, seconds `35.50`, LSTM `0.8590`, delta `+0.0500`
- tick `128928`, seconds `61.50`, LSTM `0.9757`, delta `+0.0367`
- tick `128000`, seconds `47.00`, LSTM `0.8716`, delta `-0.0335`
- tick `128512`, seconds `55.00`, LSTM `0.9201`, delta `+0.0318`
- tick `125920`, seconds `14.50`, LSTM `0.7579`, delta `-0.0222`
- tick `127232`, seconds `35.00`, LSTM `0.8090`, delta `-0.0208`
- tick `125984`, seconds `15.50`, LSTM `0.7414`, delta `-0.0202`
- tick `128320`, seconds `52.00`, LSTM `0.8833`, delta `+0.0189`

## Top 15 local ridge features

- `lag_00__CT_place_IVY`: coefficient `0.001265`, |coef| `0.001265`
- `lag_07__T2__flash_duration`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.000902`, |coef| `0.000902`
- `lag_00__T_place_TMAIN`: coefficient `-0.000722`, |coef| `0.000722`
- `lag_11__T1__flash_duration`: coefficient `-0.000666`, |coef| `0.000666`
- `lag_00__T4__duck_amount`: coefficient `0.000637`, |coef| `0.000637`
- `lag_00__CT_kills_last_3s`: coefficient `0.000634`, |coef| `0.000634`
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `0.000628`, |coef| `0.000628`
- `lag_09__T2__flash_duration`: coefficient `-0.000547`, |coef| `0.000547`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000528`, |coef| `0.000528`
- `lag_00__kill_diff_last_3s`: coefficient `0.000528`, |coef| `0.000528`
- `lag_00__CT_place_DUMPSTER`: coefficient `0.000510`, |coef| `0.000510`
- `lag_08__CT5__duck_amount`: coefficient `-0.000507`, |coef| `0.000507`
- `lag_00__CT1__is_walking`: coefficient `-0.000492`, |coef| `0.000492`
- `lag_08__CT_place_TSPAWN`: coefficient `0.000477`, |coef| `0.000477`

## Top 10 utility ridge features

- `lag_07__T2__flash_duration`: coefficient `-0.000962` (lowers CT win probability)
- `lag_11__T1__flash_duration`: coefficient `-0.000666` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `-0.000547` (lowers CT win probability)
- `lag_12__CT_A_site_active_infernos`: coefficient `-0.000470` (lowers CT win probability)
- `lag_13__T1__flash_duration`: coefficient `-0.000446` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000390` (lowers CT win probability)
- `lag_07__T_flash_duration_sum`: coefficient `-0.000383` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000371` (lowers CT win probability)
- `lag_05__CT_A_site_active_infernos`: coefficient `0.000354` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000343` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_IVY`: coefficient `0.001265` (raises CT win probability)
- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.000902` (lowers CT win probability)
- `lag_00__T_place_TMAIN`: coefficient `-0.000722` (lowers CT win probability)
- `lag_00__T4__duck_amount`: coefficient `0.000637` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000634` (raises CT win probability)
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `0.000628` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000528` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000528` (raises CT win probability)
- `lag_00__CT_place_DUMPSTER`: coefficient `0.000510` (raises CT win probability)
- `lag_08__CT5__duck_amount`: coefficient `-0.000507` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `126432`, seconds `22.50`, LSTM delta `+0.0911`

Top all feature movements:
- `lag_01__CT_place_ELECTRICALBOX`: contribution `+0.010489`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.007299`
- `lag_07__T2__flash_duration`: contribution `+0.007194`
- `lag_11__T1__flash_duration`: contribution `+0.004119`
- `lag_00__T_place_TMAIN`: contribution `+0.002801`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `+0.007194`
- `lag_11__T1__flash_duration`: contribution `+0.004119`
- `lag_07__T_flash_duration_sum`: contribution `+0.001171`
- `lag_11__T_A_site_active_infernos`: contribution `+0.001160`
- `lag_11__T_B_site_active_infernos`: contribution `+0.001050`

### tick `126496`, seconds `23.50`, LSTM delta `+0.0677`

Top all feature movements:
- `lag_09__T2__flash_duration`: contribution `+0.004091`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `+0.003908`
- `lag_00__T_place_TMAIN`: contribution `+0.002801`
- `lag_13__T1__flash_duration`: contribution `+0.002761`
- `lag_14__CT_place_ELECTRICALBOX`: contribution `+0.002060`

Top utility-only movements:
- `lag_09__T2__flash_duration`: contribution `+0.004091`
- `lag_13__T1__flash_duration`: contribution `+0.002761`
- `lag_05__CT_A_site_active_infernos`: contribution `+0.001249`

### tick `127264`, seconds `35.50`, LSTM delta `+0.0500`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `+0.014443`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `+0.010489`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `+0.007299`
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.002971`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.002289`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `+0.000982`
- `lag_10__CT_A_site_active_infernos`: contribution `+0.000503`

### tick `128928`, seconds `61.50`, LSTM delta `+0.0367`

Top all feature movements:
- `lag_13__CT_place_DUMPSTER`: contribution `+0.017438`
- `lag_05__CT_place_TSPAWN`: contribution `+0.001888`
- `lag_00__CT_kills_last_3s`: contribution `+0.001830`
- `lag_05__CT_place_DUMPSTER`: contribution `+0.001754`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001468`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `128000`, seconds `47.00`, LSTM delta `-0.0335`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `-0.014443`
- `lag_12__T_place_LONGDOG`: contribution `-0.001577`
- `lag_15__CT5__duck_amount`: contribution `-0.001514`
- `lag_05__CT4__is_walking`: contribution `-0.001018`
- `lag_10__CT4__is_walking`: contribution `-0.000954`

Top utility-only movements:
- No utility movement among the top local contributors.
