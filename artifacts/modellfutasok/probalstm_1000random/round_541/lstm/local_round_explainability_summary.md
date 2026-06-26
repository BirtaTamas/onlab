# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `116032`, seconds `0.50`, LSTM `0.0102`, delta `-0.0267`
- tick `116992`, seconds `15.50`, LSTM `0.0550`, delta `-0.0137`
- tick `118368`, seconds `37.00`, LSTM `0.0445`, delta `-0.0128`
- tick `116800`, seconds `12.50`, LSTM `0.0476`, delta `-0.0126`
- tick `116864`, seconds `13.50`, LSTM `0.0658`, delta `+0.0109`
- tick `116480`, seconds `7.50`, LSTM `0.0454`, delta `+0.0107`
- tick `119520`, seconds `55.00`, LSTM `0.0081`, delta `-0.0101`
- tick `117216`, seconds `19.00`, LSTM `0.0490`, delta `-0.0101`
- tick `116928`, seconds `14.50`, LSTM `0.0720`, delta `+0.0098`
- tick `117088`, seconds `17.00`, LSTM `0.0551`, delta `-0.0095`

## Top 15 local ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000303`, |coef| `0.000303`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000273`, |coef| `0.000273`
- `lag_06__CT_smokes_last_5s`: coefficient `0.000244`, |coef| `0.000244`
- `lag_00__CT_place_JUNGLE`: coefficient `0.000225`, |coef| `0.000225`
- `lag_00__CT_velocity_mean`: coefficient `-0.000214`, |coef| `0.000214`
- `lag_01__CT_place_PALACEALLEY`: coefficient `-0.000207`, |coef| `0.000207`
- `lag_12__CT_smokes_last_5s`: coefficient `0.000206`, |coef| `0.000206`
- `lag_01__CT2__is_walking`: coefficient `0.000203`, |coef| `0.000203`
- `lag_00__T_velocity_mean`: coefficient `-0.000188`, |coef| `0.000188`
- `lag_15__CT_smokes_last_5s`: coefficient `0.000178`, |coef| `0.000178`
- `lag_03__CT_place_PALACEALLEY`: coefficient `-0.000178`, |coef| `0.000178`
- `lag_06__CT_place_BACKALLEY`: coefficient `0.000174`, |coef| `0.000174`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000170`, |coef| `0.000170`
- `lag_09__CT_place_STAIRS`: coefficient `-0.000169`, |coef| `0.000169`
- `lag_01__CT_place_LADDER`: coefficient `-0.000169`, |coef| `0.000169`

## Top 10 utility ridge features

- `lag_06__CT_smokes_last_5s`: coefficient `0.000244` (raises CT win probability)
- `lag_12__CT_smokes_last_5s`: coefficient `0.000206` (raises CT win probability)
- `lag_15__CT_smokes_last_5s`: coefficient `0.000178` (raises CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `-0.000152` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000115` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000105` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000101` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000100` (raises CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000099` (lowers CT win probability)
- `lag_11__CT_smokes_last_5s`: coefficient `0.000096` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000303` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000273` (lowers CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.000225` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000214` (lowers CT win probability)
- `lag_01__CT_place_PALACEALLEY`: coefficient `-0.000207` (lowers CT win probability)
- `lag_01__CT2__is_walking`: coefficient `0.000203` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000188` (lowers CT win probability)
- `lag_03__CT_place_PALACEALLEY`: coefficient `-0.000178` (lowers CT win probability)
- `lag_06__CT_place_BACKALLEY`: coefficient `0.000174` (raises CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000170` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `116032`, seconds `0.50`, LSTM delta `-0.0267`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001450`
- `lag_01__T_place_TSPAWN`: contribution `-0.001210`
- `lag_00__CT_velocity_mean`: contribution `-0.000741`
- `lag_00__T_velocity_mean`: contribution `-0.000539`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000469`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000253`
- `lag_01__T1__utility_total`: contribution `-0.000236`
- `lag_01__T1__flash`: contribution `-0.000208`
- `lag_01__smoke_inv_diff`: contribution `-0.000191`
- `lag_01__flash_inv_diff`: contribution `-0.000176`

### tick `116992`, seconds `15.50`, LSTM delta `-0.0137`

Top all feature movements:
- `lag_12__CT_smokes_last_5s`: contribution `-0.003556`
- `lag_09__CT_place_STAIRS`: contribution `-0.001317`
- `lag_15__CT_place_JUNGLE`: contribution `-0.000797`
- `lag_13__CT_place_JUNGLE`: contribution `-0.000699`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `-0.000492`

Top utility-only movements:
- `lag_12__CT_smokes_last_5s`: contribution `-0.003556`

### tick `118368`, seconds `37.00`, LSTM delta `-0.0128`

Top all feature movements:
- `lag_00__CT_place_SIDEALLEY`: contribution `-0.002303`
- `lag_01__CT_place_LADDER`: contribution `-0.001756`
- `lag_01__CT_place_UNDERPASS`: contribution `-0.000959`
- `lag_10__CT_place_JUNGLE`: contribution `-0.000802`
- `lag_01__CT2__is_walking`: contribution `-0.000478`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `116800`, seconds `12.50`, LSTM delta `-0.0126`

Top all feature movements:
- `lag_06__CT_smokes_last_5s`: contribution `-0.004211`
- `lag_12__CT_place_SNIPERSNEST`: contribution `-0.000548`
- `lag_11__CT1__duck_amount`: contribution `-0.000400`
- `lag_14__T4__is_walking`: contribution `-0.000362`
- `lag_07__T_place_SIDEALLEY`: contribution `-0.000360`

Top utility-only movements:
- `lag_06__CT_smokes_last_5s`: contribution `-0.004211`

### tick `116864`, seconds `13.50`, LSTM delta `+0.0109`

Top all feature movements:
- `lag_08__CT_smokes_last_5s`: contribution `+0.001656`
- `lag_05__CT_place_STAIRS`: contribution `+0.000637`
- `lag_12__CT_place_SNIPERSNEST`: contribution `+0.000548`
- `lag_12__CT_place_UNDERPASS`: contribution `+0.000496`
- `lag_13__CT1__duck_amount`: contribution `+0.000440`

Top utility-only movements:
- `lag_08__CT_smokes_last_5s`: contribution `+0.001656`
