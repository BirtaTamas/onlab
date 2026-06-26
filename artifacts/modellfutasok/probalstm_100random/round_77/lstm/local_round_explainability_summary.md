# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `120336`, seconds `36.00`, LSTM `0.3070`, delta `-0.2074`
- tick `118896`, seconds `13.50`, LSTM `0.3541`, delta `-0.0673`
- tick `120656`, seconds `41.00`, LSTM `0.2917`, delta `+0.0588`
- tick `120496`, seconds `38.50`, LSTM `0.2604`, delta `-0.0529`
- tick `120880`, seconds `44.50`, LSTM `0.2706`, delta `-0.0495`
- tick `118064`, seconds `0.50`, LSTM `0.3278`, delta `-0.0474`
- tick `119024`, seconds `15.50`, LSTM `0.4472`, delta `+0.0461`
- tick `119376`, seconds `21.00`, LSTM `0.5125`, delta `+0.0447`
- tick `120560`, seconds `39.50`, LSTM `0.1829`, delta `-0.0438`
- tick `121616`, seconds `56.00`, LSTM `0.2271`, delta `-0.0437`

## Top 15 local ridge features

- `lag_13__CT_place_RAFTERS`: coefficient `0.001638`, |coef| `0.001638`
- `lag_11__CT_place_HUTROOF`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_13__CT_place_HUTROOF`: coefficient `-0.001380`, |coef| `0.001380`
- `lag_00__CT5__duck_amount`: coefficient `-0.001095`, |coef| `0.001095`
- `lag_07__T_place_SECRET`: coefficient `-0.001022`, |coef| `0.001022`
- `lag_01__CT5__shots_fired`: coefficient `-0.001000`, |coef| `0.001000`
- `lag_00__CT_place_TUNNELS`: coefficient `0.000997`, |coef| `0.000997`
- `lag_08__T4__duck_amount`: coefficient `0.000988`, |coef| `0.000988`
- `lag_00__CT_place_ADMIN`: coefficient `0.000966`, |coef| `0.000966`
- `lag_04__T5__duck_amount`: coefficient `-0.000941`, |coef| `0.000941`
- `lag_12__T4__is_walking`: coefficient `0.000899`, |coef| `0.000899`
- `lag_00__CT_velocity_mean`: coefficient `-0.000863`, |coef| `0.000863`
- `lag_10__CT5__is_walking`: coefficient `0.000857`, |coef| `0.000857`
- `lag_06__T3__duck_amount`: coefficient `0.000851`, |coef| `0.000851`
- `lag_06__T5__is_walking`: coefficient `0.000846`, |coef| `0.000846`

## Top 10 utility ridge features

- `lag_11__CT4__smoke`: coefficient `-0.000724` (lowers CT win probability)
- `lag_11__CT3__smoke`: coefficient `0.000631` (raises CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `0.000594` (raises CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `0.000570` (raises CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.000525` (raises CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.000504` (raises CT win probability)
- `lag_03__active_smokes_total`: coefficient `0.000501` (raises CT win probability)
- `lag_03__T_active_smokes`: coefficient `0.000490` (raises CT win probability)
- `lag_09__T_active_smokes`: coefficient `0.000475` (raises CT win probability)
- `lag_09__active_smokes_total`: coefficient `0.000464` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_RAFTERS`: coefficient `0.001638` (raises CT win probability)
- `lag_11__CT_place_HUTROOF`: coefficient `-0.001465` (lowers CT win probability)
- `lag_13__CT_place_HUTROOF`: coefficient `-0.001380` (lowers CT win probability)
- `lag_00__CT5__duck_amount`: coefficient `-0.001095` (lowers CT win probability)
- `lag_07__T_place_SECRET`: coefficient `-0.001022` (lowers CT win probability)
- `lag_01__CT5__shots_fired`: coefficient `-0.001000` (lowers CT win probability)
- `lag_00__CT_place_TUNNELS`: coefficient `0.000997` (raises CT win probability)
- `lag_08__T4__duck_amount`: coefficient `0.000988` (raises CT win probability)
- `lag_00__CT_place_ADMIN`: coefficient `0.000966` (raises CT win probability)
- `lag_04__T5__duck_amount`: coefficient `-0.000941` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `120336`, seconds `36.00`, LSTM delta `-0.2074`

Top all feature movements:
- `lag_11__CT_place_HUTROOF`: contribution `-0.010252`
- `lag_13__CT_place_HUTROOF`: contribution `-0.009660`
- `lag_13__CT_place_RAFTERS`: contribution `-0.008754`
- `lag_07__T_place_SECRET`: contribution `-0.005375`
- `lag_12__T_place_SECRET`: contribution `-0.004235`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118896`, seconds `13.50`, LSTM delta `-0.0673`

Top all feature movements:
- `lag_10__CT_place_HEAVEN`: contribution `-0.007907`
- `lag_10__CT_place_RAFTERS`: contribution `-0.005148`
- `lag_13__CT_place_HELL`: contribution `-0.003811`
- `lag_13__CT_place_ADMIN`: contribution `-0.003605`
- `lag_09__T_flash_duration_sum`: contribution `-0.003399`

Top utility-only movements:
- `lag_09__T_flash_duration_sum`: contribution `-0.003399`
- `lag_09__T5__flash_duration`: contribution `-0.002384`
- `lag_09__T3__flash_duration`: contribution `-0.002203`
- `lag_09__T4__flash_duration`: contribution `-0.002089`

### tick `120656`, seconds `41.00`, LSTM delta `+0.0588`

Top all feature movements:
- `lag_01__CT_place_LOCKERROOM`: contribution `+0.008081`
- `lag_00__CT_place_VENTS`: contribution `+0.006251`
- `lag_04__CT1__is_scoped`: contribution `+0.003300`
- `lag_00__CT1__is_scoped`: contribution `+0.003009`
- `lag_11__T_place_SECRET`: contribution `+0.002496`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120496`, seconds `38.50`, LSTM delta `-0.0529`

Top all feature movements:
- `lag_02__CT_place_HUTROOF`: contribution `-0.005483`
- `lag_06__T_place_SECRET`: contribution `-0.004305`
- `lag_12__T_place_SECRET`: contribution `-0.004235`
- `lag_02__T5__duck_amount`: contribution `+0.002619`
- `lag_04__T3__shots_fired`: contribution `-0.002019`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120880`, seconds `44.50`, LSTM delta `-0.0495`

Top all feature movements:
- `lag_07__CT_place_VENTS`: contribution `-0.006919`
- `lag_07__CT_place_MINI`: contribution `-0.004577`
- `lag_08__CT_place_LOCKERROOM`: contribution `-0.004363`
- `lag_14__CT_place_HUTROOF`: contribution `-0.003560`
- `lag_00__T_place_SECRET`: contribution `-0.003328`

Top utility-only movements:
- No utility movement among the top local contributors.
