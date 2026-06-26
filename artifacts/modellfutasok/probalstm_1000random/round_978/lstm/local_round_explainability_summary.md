# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-heroic-vs-aurora-bo3-QigxwcikBDdlIOkrYDpY7y/heroic-vs-aurora-m2-dust2.csv`
- round_num: `21`

## Largest probability jumps

- tick `158640`, seconds `16.00`, LSTM `0.1663`, delta `-0.2692`
- tick `158288`, seconds `10.50`, LSTM `0.2943`, delta `-0.2503`
- tick `158608`, seconds `15.50`, LSTM `0.4355`, delta `+0.1634`
- tick `158192`, seconds `9.00`, LSTM `0.5642`, delta `-0.0639`
- tick `158672`, seconds `16.50`, LSTM `0.1093`, delta `-0.0570`
- tick `158320`, seconds `11.00`, LSTM `0.2424`, delta `-0.0519`
- tick `158128`, seconds `8.00`, LSTM `0.6304`, delta `+0.0455`
- tick `158576`, seconds `15.00`, LSTM `0.2722`, delta `+0.0377`
- tick `158384`, seconds `12.00`, LSTM `0.2501`, delta `+0.0325`
- tick `158352`, seconds `11.50`, LSTM `0.2176`, delta `-0.0248`

## Top 15 local ridge features

- `lag_15__T_he_last_5s`: coefficient `-0.003054`, |coef| `0.003054`
- `lag_14__CT_place_HOLE`: coefficient `0.002681`, |coef| `0.002681`
- `lag_11__CT2__is_scoped`: coefficient `0.001862`, |coef| `0.001862`
- `lag_15__CT_place_HOLE`: coefficient `0.001775`, |coef| `0.001775`
- `lag_03__CT_place_HOLE`: coefficient `0.001587`, |coef| `0.001587`
- `lag_05__CT_place_HOLE`: coefficient `-0.001540`, |coef| `0.001540`
- `lag_03__CT_place_EXTENDEDA`: coefficient `0.001528`, |coef| `0.001528`
- `lag_08__CT_place_BDOORS`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_00__T_flashed_players`: coefficient `0.001425`, |coef| `0.001425`
- `lag_05__T_he_last_5s`: coefficient `0.001401`, |coef| `0.001401`
- `lag_00__CT4__flash_duration`: coefficient `0.001380`, |coef| `0.001380`
- `lag_06__CT_place_LONGDOORS`: coefficient `-0.001268`, |coef| `0.001268`
- `lag_04__CT_place_SHORTSTAIRS`: coefficient `0.001238`, |coef| `0.001238`
- `lag_01__CT4__shots_fired`: coefficient `-0.001215`, |coef| `0.001215`
- `lag_00__T_kills_last_3s`: coefficient `-0.001210`, |coef| `0.001210`

## Top 10 utility ridge features

- `lag_15__T_he_last_5s`: coefficient `-0.003054` (lowers CT win probability)
- `lag_05__T_he_last_5s`: coefficient `0.001401` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001380` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000904` (raises CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.000802` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000773` (lowers CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000744` (raises CT win probability)
- `lag_11__CT2__flash`: coefficient `0.000724` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000717` (raises CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000707` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_HOLE`: coefficient `0.002681` (raises CT win probability)
- `lag_11__CT2__is_scoped`: coefficient `0.001862` (raises CT win probability)
- `lag_15__CT_place_HOLE`: coefficient `0.001775` (raises CT win probability)
- `lag_03__CT_place_HOLE`: coefficient `0.001587` (raises CT win probability)
- `lag_05__CT_place_HOLE`: coefficient `-0.001540` (lowers CT win probability)
- `lag_03__CT_place_EXTENDEDA`: coefficient `0.001528` (raises CT win probability)
- `lag_08__CT_place_BDOORS`: coefficient `-0.001437` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `0.001425` (raises CT win probability)
- `lag_06__CT_place_LONGDOORS`: coefficient `-0.001268` (lowers CT win probability)
- `lag_04__CT_place_SHORTSTAIRS`: coefficient `0.001238` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `158640`, seconds `16.00`, LSTM delta `-0.2692`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `-0.029926`
- `lag_11__CT2__is_scoped`: contribution `-0.011396`
- `lag_03__CT_place_EXTENDEDA`: contribution `-0.008580`
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.006903`
- `lag_00__CT4__flash_duration`: contribution `-0.006640`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.006640`
- `lag_11__CT_B_site_active_infernos`: contribution `-0.002655`
- `lag_11__CT2__flash`: contribution `-0.002619`

### tick `158288`, seconds `10.50`, LSTM delta `-0.2503`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `-0.039857`
- `lag_05__T_he_last_5s`: contribution `-0.018290`
- `lag_03__CT_place_HOLE`: contribution `-0.017722`
- `lag_05__CT_place_HOLE`: contribution `-0.017196`
- `lag_08__CT_place_BDOORS`: contribution `-0.013823`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `-0.039857`
- `lag_05__T_he_last_5s`: contribution `-0.018290`
- `lag_07__T_flashes_last_5s`: contribution `-0.005790`
- `lag_00__CT2__flash`: contribution `-0.003269`

### tick `158608`, seconds `15.50`, LSTM delta `+0.1634`

Top all feature movements:
- `lag_15__T_he_last_5s`: contribution `+0.039857`
- `lag_15__CT_place_HOLE`: contribution `+0.019819`
- `lag_11__CT2__is_scoped`: contribution `+0.011396`
- `lag_13__CT_place_HOLE`: contribution `+0.010114`
- `lag_03__CT_place_EXTENDEDA`: contribution `+0.008580`

Top utility-only movements:
- `lag_15__T_he_last_5s`: contribution `+0.039857`
- `lag_00__CT4__flash_duration`: contribution `+0.005543`

### tick `158192`, seconds `9.00`, LSTM delta `-0.0639`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.011964`
- `lag_02__T_he_last_5s`: contribution `-0.008543`
- `lag_12__T_he_last_5s`: contribution `-0.006121`
- `lag_02__CT_place_HOLE`: contribution `-0.005721`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `-0.005250`

Top utility-only movements:
- `lag_02__T_he_last_5s`: contribution `-0.008543`
- `lag_12__T_he_last_5s`: contribution `-0.006121`
- `lag_04__T_flashes_last_5s`: contribution `-0.002902`

### tick `158672`, seconds `16.50`, LSTM delta `-0.0570`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `-0.019819`
- `lag_12__CT2__is_scoped`: contribution `+0.005229`
- `lag_04__CT_place_EXTENDEDA`: contribution `+0.004499`
- `lag_07__CT_place_LONGDOORS`: contribution `-0.003308`
- `lag_07__CT_place_PIT`: contribution `-0.002869`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.002315`
