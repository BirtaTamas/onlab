# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `40007`, seconds `83.00`, LSTM `0.8529`, delta `+0.3241`
- tick `39655`, seconds `77.50`, LSTM `0.8165`, delta `+0.2081`
- tick `39303`, seconds `72.00`, LSTM `0.3358`, delta `+0.1739`
- tick `39751`, seconds `79.00`, LSTM `0.6572`, delta `-0.1343`
- tick `39975`, seconds `82.50`, LSTM `0.5288`, delta `-0.1178`
- tick `39367`, seconds `73.00`, LSTM `0.4329`, delta `+0.1155`
- tick `34727`, seconds `0.50`, LSTM `0.1064`, delta `-0.0839`
- tick `39943`, seconds `82.00`, LSTM `0.6465`, delta `+0.0706`
- tick `39271`, seconds `71.50`, LSTM `0.1619`, delta `+0.0656`
- tick `39463`, seconds `74.50`, LSTM `0.5604`, delta `+0.0611`

## Top 15 local ridge features

- `lag_11__T_place_HUT`: coefficient `0.003066`, |coef| `0.003066`
- `lag_13__CT_place_MINI`: coefficient `0.002673`, |coef| `0.002673`
- `lag_00__T_place_HUT`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_09__CT_place_MINI`: coefficient `-0.002119`, |coef| `0.002119`
- `lag_00__CT1__is_scoped`: coefficient `0.001999`, |coef| `0.001999`
- `lag_14__CT_place_MINI`: coefficient `-0.001892`, |coef| `0.001892`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001852`, |coef| `0.001852`
- `lag_00__CT_kills_last_3s`: coefficient `0.001778`, |coef| `0.001778`
- `lag_00__kill_diff_last_3s`: coefficient `0.001719`, |coef| `0.001719`
- `lag_00__T2__duck_amount`: coefficient `0.001682`, |coef| `0.001682`
- `lag_10__T3__duck_amount`: coefficient `0.001450`, |coef| `0.001450`
- `lag_09__T_place_HUT`: coefficient `0.001411`, |coef| `0.001411`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001372`, |coef| `0.001372`
- `lag_00__CT2__duck_amount`: coefficient `-0.001364`, |coef| `0.001364`
- `lag_11__kill_diff_last_3s`: coefficient `0.001363`, |coef| `0.001363`

## Top 10 utility ridge features

- `lag_07__T_A_site_active_infernos`: coefficient `0.001186` (raises CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `0.001120` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000957` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000949` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000882` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000841` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000815` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `0.000793` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.000770` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000757` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_HUT`: coefficient `0.003066` (raises CT win probability)
- `lag_13__CT_place_MINI`: coefficient `0.002673` (raises CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.002245` (lowers CT win probability)
- `lag_09__CT_place_MINI`: coefficient `-0.002119` (lowers CT win probability)
- `lag_00__CT1__is_scoped`: coefficient `0.001999` (raises CT win probability)
- `lag_14__CT_place_MINI`: coefficient `-0.001892` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001852` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001778` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001719` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `0.001682` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `40007`, seconds `83.00`, LSTM delta `+0.3241`

Top all feature movements:
- `lag_11__T_place_HUT`: contribution `+0.028576`
- `lag_00__T_place_HUT`: contribution `+0.020929`
- `lag_13__CT_place_MINI`: contribution `+0.016385`
- `lag_09__CT_place_MINI`: contribution `+0.012994`
- `lag_14__CT_place_MINI`: contribution `+0.011602`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.005250`
- `lag_15__T_B_site_active_infernos`: contribution `+0.004757`

### tick `39655`, seconds `77.50`, LSTM delta `+0.2081`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `-0.020929`
- `lag_10__CT_shots_fired_sum`: contribution `+0.012329`
- `lag_14__T_place_HUT`: contribution `+0.011850`
- `lag_14__CT_place_MINI`: contribution `+0.011602`
- `lag_07__T_shots_fired_sum`: contribution `+0.009513`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39303`, seconds `72.00`, LSTM delta `+0.1739`

Top all feature movements:
- `lag_07__T_place_HUT`: contribution `+0.008939`
- `lag_03__T_place_HUT`: contribution `+0.008187`
- `lag_05__CT_place_GARAGE`: contribution `+0.008020`
- `lag_01__T_place_HUT`: contribution `+0.007358`
- `lag_14__CT_place_SECRET`: contribution `+0.006975`

Top utility-only movements:
- `lag_08__T_A_site_active_infernos`: contribution `+0.004507`
- `lag_08__T_B_site_active_infernos`: contribution `+0.004036`
- `lag_07__T_A_site_active_infernos`: contribution `+0.003529`
- `lag_07__T_B_site_active_infernos`: contribution `+0.003167`

### tick `39751`, seconds `79.00`, LSTM delta `-0.1343`

Top all feature movements:
- `lag_11__T_place_HUT`: contribution `-0.028576`
- `lag_09__CT_place_MINI`: contribution `-0.012994`
- `lag_10__T_shots_fired_sum`: contribution `-0.012299`
- `lag_10__T3__shots_fired`: contribution `-0.009427`
- `lag_15__T_place_HUT`: contribution `-0.008230`

Top utility-only movements:
- `lag_07__T_A_site_active_infernos`: contribution `-0.007058`
- `lag_07__T_B_site_active_infernos`: contribution `-0.006335`
- `lag_07__T_active_infernos`: contribution `-0.003302`

### tick `39975`, seconds `82.50`, LSTM delta `-0.1178`

Top all feature movements:
- `lag_13__CT_place_MINI`: contribution `-0.016385`
- `lag_00__T_shots_fired_sum`: contribution `-0.006941`
- `lag_07__CT_place_ADMIN`: contribution `-0.005857`
- `lag_12__T_place_SQUEAKY`: contribution `-0.005520`
- `lag_14__T_A_site_active_infernos`: contribution `-0.004854`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.004854`
- `lag_14__T_B_site_active_infernos`: contribution `-0.004354`
- `lag_15__T_A_site_active_infernos`: contribution `+0.002625`
- `lag_15__T_B_site_active_infernos`: contribution `+0.002379`
- `lag_14__T_active_infernos`: contribution `-0.002257`
