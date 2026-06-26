# Local Round Explainability

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `9517`, seconds `61.50`, LSTM `0.0522`, delta `-0.1959`
- tick `8045`, seconds `38.50`, LSTM `0.2735`, delta `-0.0770`
- tick `8109`, seconds `39.50`, LSTM `0.2070`, delta `-0.0594`
- tick `9101`, seconds `55.00`, LSTM `0.2838`, delta `+0.0533`
- tick `8237`, seconds `41.50`, LSTM `0.1292`, delta `-0.0420`
- tick `6541`, seconds `15.00`, LSTM `0.3479`, delta `-0.0412`
- tick `9485`, seconds `61.00`, LSTM `0.2481`, delta `-0.0404`
- tick `8333`, seconds `43.00`, LSTM `0.1594`, delta `+0.0392`
- tick `8461`, seconds `45.00`, LSTM `0.1318`, delta `-0.0385`
- tick `11021`, seconds `85.00`, LSTM `0.0563`, delta `-0.0383`

## Top 15 local ridge features

- `lag_10__CT_place_HUT`: coefficient `-0.002315`, |coef| `0.002315`
- `lag_00__CT_place_HUT`: coefficient `0.002209`, |coef| `0.002209`
- `lag_03__CT_place_VENTS`: coefficient `-0.002020`, |coef| `0.002020`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001523`, |coef| `0.001523`
- `lag_14__CT_place_GARAGE`: coefficient `0.001458`, |coef| `0.001458`
- `lag_00__T_place_TROPHY`: coefficient `0.001415`, |coef| `0.001415`
- `lag_01__CT_place_GARAGE`: coefficient `-0.001297`, |coef| `0.001297`
- `lag_02__CT_place_VENTS`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_07__bomb_events_last_5s`: coefficient `0.001120`, |coef| `0.001120`
- `lag_00__CT2__is_walking`: coefficient `-0.001055`, |coef| `0.001055`
- `lag_09__CT_place_HUT`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_01__T3__duck_amount`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001022`, |coef| `0.001022`
- `lag_06__CT_place_HUTROOF`: coefficient `-0.000984`, |coef| `0.000984`
- `lag_04__CT_place_VENTS`: coefficient `-0.000953`, |coef| `0.000953`

## Top 10 utility ridge features

- `lag_06__T_A_site_active_infernos`: coefficient `-0.000929` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000825` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000620` (raises CT win probability)
- `lag_10__T2__molly`: coefficient `0.000615` (raises CT win probability)
- `lag_10__T1__smoke`: coefficient `0.000604` (raises CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.000589` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.000558` (lowers CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000553` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000507` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000477` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_HUT`: coefficient `-0.002315` (lowers CT win probability)
- `lag_00__CT_place_HUT`: coefficient `0.002209` (raises CT win probability)
- `lag_03__CT_place_VENTS`: coefficient `-0.002020` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001523` (lowers CT win probability)
- `lag_14__CT_place_GARAGE`: coefficient `0.001458` (raises CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `0.001415` (raises CT win probability)
- `lag_01__CT_place_GARAGE`: coefficient `-0.001297` (lowers CT win probability)
- `lag_02__CT_place_VENTS`: coefficient `-0.001137` (lowers CT win probability)
- `lag_07__bomb_events_last_5s`: coefficient `0.001120` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001055` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `9517`, seconds `61.50`, LSTM delta `-0.1959`

Top all feature movements:
- `lag_10__CT_place_HUT`: contribution `-0.022573`
- `lag_00__CT_place_HUT`: contribution `-0.021547`
- `lag_03__CT_place_VENTS`: contribution `-0.016947`
- `lag_14__CT_place_GARAGE`: contribution `-0.010482`
- `lag_00__T_bomb_zone_count`: contribution `-0.008868`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `-0.002765`
- `lag_06__T_B_site_active_infernos`: contribution `-0.002332`

### tick `8045`, seconds `38.50`, LSTM delta `-0.0770`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `-0.017951`
- `lag_00__T_place_CONTROL`: contribution `-0.009486`
- `lag_08__T_place_TROPHY`: contribution `-0.004074`
- `lag_03__CT_place_HUTROOF`: contribution `-0.003668`
- `lag_06__T_place_TROPHY`: contribution `-0.003516`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8109`, seconds `39.50`, LSTM delta `-0.0594`

Top all feature movements:
- `lag_02__T_place_CONTROL`: contribution `-0.010468`
- `lag_00__T_place_TROPHY`: contribution `-0.008976`
- `lag_02__T_place_TROPHY`: contribution `-0.008392`
- `lag_05__CT_place_HUTROOF`: contribution `-0.006591`
- `lag_00__T_place_CONTROL`: contribution `-0.004743`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.001140`

### tick `9101`, seconds `55.00`, LSTM delta `+0.0533`

Top all feature movements:
- `lag_01__CT_place_GARAGE`: contribution `+0.009320`
- `lag_06__CT_place_HUTROOF`: contribution `+0.006884`
- `lag_07__CT_place_VENTS`: contribution `+0.006721`
- `lag_04__bomb_events_last_5s`: contribution `+0.003189`
- `lag_00__CT2__is_walking`: contribution `+0.002491`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `+0.001662`
- `lag_12__T_B_site_active_infernos`: contribution `+0.001433`

### tick `8237`, seconds `41.50`, LSTM delta `-0.0420`

Top all feature movements:
- `lag_06__T_place_CONTROL`: contribution `-0.011681`
- `lag_06__T_place_TROPHY`: contribution `+0.007031`
- `lag_02__T_place_CONTROL`: contribution `-0.005234`
- `lag_00__T_place_CONTROL`: contribution `+0.004743`
- `lag_02__T_place_TROPHY`: contribution `-0.004196`

Top utility-only movements:
- `lag_03__CT2__flash_duration`: contribution `-0.001360`
