# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `5`

## Largest probability jumps

- tick `48090`, seconds `49.50`, LSTM `0.8272`, delta `+0.1952`
- tick `48314`, seconds `53.00`, LSTM `0.9326`, delta `+0.0610`
- tick `47802`, seconds `45.00`, LSTM `0.6644`, delta `+0.0429`
- tick `46586`, seconds `26.00`, LSTM `0.5396`, delta `-0.0385`
- tick `47962`, seconds `47.50`, LSTM `0.6335`, delta `-0.0354`
- tick `45626`, seconds `11.00`, LSTM `0.4965`, delta `-0.0294`
- tick `46138`, seconds `19.00`, LSTM `0.5892`, delta `+0.0289`
- tick `45434`, seconds `8.00`, LSTM `0.5534`, delta `+0.0282`
- tick `48058`, seconds `49.00`, LSTM `0.6320`, delta `-0.0267`
- tick `48282`, seconds `52.50`, LSTM `0.8716`, delta `+0.0263`

## Top 15 local ridge features

- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.001555`, |coef| `0.001555`
- `lag_03__CT_place_ELECTRICALBOX`: coefficient `-0.001505`, |coef| `0.001505`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001410`, |coef| `0.001410`
- `lag_00__CT4__shots_fired`: coefficient `0.001092`, |coef| `0.001092`
- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000947`, |coef| `0.000947`
- `lag_00__CT4__duck_amount`: coefficient `0.000942`, |coef| `0.000942`
- `lag_04__T_place_TMAIN`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_01__T_A_site_active_infernos`: coefficient `0.000919`, |coef| `0.000919`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000915`, |coef| `0.000915`
- `lag_12__CT3__duck_amount`: coefficient `0.000876`, |coef| `0.000876`
- `lag_01__T_shots_fired_sum`: coefficient `0.000853`, |coef| `0.000853`
- `lag_01__T3__shots_fired`: coefficient `0.000849`, |coef| `0.000849`
- `lag_00__CT5__shots_fired`: coefficient `0.000822`, |coef| `0.000822`
- `lag_00__CT2__duck_amount`: coefficient `0.000788`, |coef| `0.000788`
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.000780`, |coef| `0.000780`

## Top 10 utility ridge features

- `lag_03__CT_utility_damage_last_5s`: coefficient `0.000947` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000919` (raises CT win probability)
- `lag_03__utility_damage_diff_last_5s`: coefficient `0.000780` (raises CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000744` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000688` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000637` (raises CT win probability)
- `lag_11__CT5__molly`: coefficient `-0.000612` (lowers CT win probability)
- `lag_12__T2__molly`: coefficient `-0.000566` (lowers CT win probability)
- `lag_11__CT4__smoke`: coefficient `-0.000552` (lowers CT win probability)
- `lag_05__T1__molly`: coefficient `-0.000548` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_ELECTRICALBOX`: coefficient `0.001555` (raises CT win probability)
- `lag_03__CT_place_ELECTRICALBOX`: coefficient `-0.001505` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001410` (raises CT win probability)
- `lag_00__CT4__shots_fired`: coefficient `0.001092` (raises CT win probability)
- `lag_00__CT4__duck_amount`: coefficient `0.000942` (raises CT win probability)
- `lag_04__T_place_TMAIN`: coefficient `-0.000926` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.000915` (raises CT win probability)
- `lag_12__CT3__duck_amount`: coefficient `0.000876` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `0.000853` (raises CT win probability)
- `lag_01__T3__shots_fired`: coefficient `0.000849` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `48090`, seconds `49.50`, LSTM delta `+0.1952`

Top all feature movements:
- `lag_09__CT_place_ELECTRICALBOX`: contribution `+0.018076`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `+0.017495`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007835`
- `lag_15__T_place_TSTAIRS`: contribution `+0.004306`
- `lag_00__CT_duck_amount_mean`: contribution `+0.003616`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.003544`
- `lag_01__T_A_site_active_infernos`: contribution `+0.002735`
- `lag_09__CT_A_site_active_infernos`: contribution `+0.002627`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.002395`

### tick `48314`, seconds `53.00`, LSTM delta `+0.0610`

Top all feature movements:
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.003672`
- `lag_11__T_place_TMAIN`: contribution `-0.002997`
- `lag_06__CT_shots_fired_sum`: contribution `+0.002907`
- `lag_02__CT_place_LONGDOG`: contribution `+0.002843`
- `lag_05__CT_shots_fired_sum`: contribution `+0.002325`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `47802`, seconds `45.00`, LSTM delta `+0.0429`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.003092`
- `lag_15__T_place_TMAIN`: contribution `+0.002820`
- `lag_01__T_A_site_active_infernos`: contribution `+0.002735`
- `lag_12__T_place_TSTAIRS`: contribution `+0.001782`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.001691`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `+0.002735`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.001691`
- `lag_01__T_active_infernos`: contribution `+0.001326`
- `lag_01__active_infernos_total`: contribution `+0.000755`
- `lag_02__CT5__molly`: contribution `+0.000699`

### tick `46586`, seconds `26.00`, LSTM delta `-0.0385`

Top all feature movements:
- `lag_01__T_place_DUMPSTER`: contribution `-0.003853`
- `lag_00__CT2__flash_duration`: contribution `-0.003739`
- `lag_14__CT2__flash_duration`: contribution `-0.003323`
- `lag_13__CT_flashed_players`: contribution `-0.001449`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.001126`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.003739`
- `lag_14__CT2__flash_duration`: contribution `-0.003323`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.001126`
- `lag_14__CT_flash_duration_sum`: contribution `-0.000804`
- `lag_05__CT_active_infernos`: contribution `-0.000595`

### tick `47962`, seconds `47.50`, LSTM delta `-0.0354`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `-0.004730`
- `lag_11__T_place_TSTAIRS`: contribution `-0.001890`
- `lag_12__CT3__duck_amount`: contribution `-0.001744`
- `lag_00__CT_place_TUNNELS`: contribution `-0.001351`
- `lag_05__CT_A_site_active_infernos`: contribution `-0.001126`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `-0.001126`
- `lag_05__CT_active_infernos`: contribution `-0.000595`
