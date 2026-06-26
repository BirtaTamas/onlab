# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `113239`, seconds `50.50`, LSTM `0.9542`, delta `+0.0617`
- tick `110103`, seconds `1.50`, LSTM `0.9200`, delta `+0.0310`
- tick `110583`, seconds `9.00`, LSTM `0.9036`, delta `-0.0182`
- tick `111159`, seconds `18.00`, LSTM `0.9520`, delta `+0.0173`
- tick `110903`, seconds `14.00`, LSTM `0.9041`, delta `-0.0160`
- tick `113367`, seconds `52.50`, LSTM `0.9736`, delta `+0.0136`
- tick `113079`, seconds `48.00`, LSTM `0.9092`, delta `-0.0128`
- tick `113047`, seconds `47.50`, LSTM `0.9221`, delta `-0.0121`
- tick `111095`, seconds `17.00`, LSTM `0.9246`, delta `-0.0119`
- tick `111063`, seconds `16.50`, LSTM `0.9365`, delta `+0.0114`

## Top 15 local ridge features

- `lag_09__T_place_IVY`: coefficient `-0.000749`, |coef| `0.000749`
- `lag_09__T_place_TUNNELS`: coefficient `0.000607`, |coef| `0.000607`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.000576`, |coef| `0.000576`
- `lag_07__CT_shots_fired_sum`: coefficient `-0.000365`, |coef| `0.000365`
- `lag_03__T_place_IVY`: coefficient `0.000351`, |coef| `0.000351`
- `lag_08__T_place_TUNNELS`: coefficient `0.000340`, |coef| `0.000340`
- `lag_05__CT_place_LONGDOG`: coefficient `-0.000330`, |coef| `0.000330`
- `lag_01__T_place_IVY`: coefficient `0.000329`, |coef| `0.000329`
- `lag_07__CT4__shots_fired`: coefficient `-0.000321`, |coef| `0.000321`
- `lag_08__T_place_IVY`: coefficient `-0.000316`, |coef| `0.000316`
- `lag_00__CT_place_ENTRANCE`: coefficient `0.000315`, |coef| `0.000315`
- `lag_05__T_place_IVY`: coefficient `0.000314`, |coef| `0.000314`
- `lag_13__CT_place_ENTRANCE`: coefficient `-0.000291`, |coef| `0.000291`
- `lag_00__bomb_events_last_5s`: coefficient `0.000283`, |coef| `0.000283`
- `lag_01__CT4__duck_amount`: coefficient `-0.000269`, |coef| `0.000269`

## Top 10 utility ridge features

- `lag_14__CT_B_site_active_smokes`: coefficient `-0.000153` (lowers CT win probability)
- `lag_03__smoke_inv_diff`: coefficient `0.000152` (raises CT win probability)
- `lag_07__CT1__smoke`: coefficient `-0.000142` (lowers CT win probability)
- `lag_14__CT_A_site_active_smokes`: coefficient `-0.000123` (lowers CT win probability)
- `lag_01__CT4__smoke`: coefficient `-0.000117` (lowers CT win probability)
- `lag_03__CT3__smoke`: coefficient `0.000111` (raises CT win probability)
- `lag_03__CT1__smoke`: coefficient `0.000109` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `-0.000104` (lowers CT win probability)
- `lag_03__CT_B_site_active_smokes`: coefficient `-0.000101` (lowers CT win probability)
- `lag_03__CT_active_smokes`: coefficient `-0.000100` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_IVY`: coefficient `-0.000749` (lowers CT win probability)
- `lag_09__T_place_TUNNELS`: coefficient `0.000607` (raises CT win probability)
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.000576` (raises CT win probability)
- `lag_07__CT_shots_fired_sum`: coefficient `-0.000365` (lowers CT win probability)
- `lag_03__T_place_IVY`: coefficient `0.000351` (raises CT win probability)
- `lag_08__T_place_TUNNELS`: coefficient `0.000340` (raises CT win probability)
- `lag_05__CT_place_LONGDOG`: coefficient `-0.000330` (lowers CT win probability)
- `lag_01__T_place_IVY`: coefficient `0.000329` (raises CT win probability)
- `lag_07__CT4__shots_fired`: coefficient `-0.000321` (lowers CT win probability)
- `lag_08__T_place_IVY`: coefficient `-0.000316` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `113239`, seconds `50.50`, LSTM delta `+0.0617`

Top all feature movements:
- `lag_09__T_place_IVY`: contribution `+0.012013`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.006692`
- `lag_09__T_place_TUNNELS`: contribution `+0.005103`
- `lag_07__CT_shots_fired_sum`: contribution `+0.002789`
- `lag_05__CT_place_LONGDOG`: contribution `+0.002151`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110103`, seconds `1.50`, LSTM delta `+0.0310`

Top all feature movements:
- `lag_00__CT_place_ENTRANCE`: contribution `+0.002799`
- `lag_03__CT_place_CTSPAWN`: contribution `+0.001168`
- `lag_03__T_place_TSPAWN`: contribution `+0.000969`
- `lag_03__CT_closest_enemy_dist`: contribution `+0.000808`
- `lag_03__T_closest_enemy_dist`: contribution `+0.000746`

Top utility-only movements:
- `lag_03__smoke_inv_diff`: contribution `+0.000393`
- `lag_01__CT4__smoke`: contribution `+0.000255`

### tick `110583`, seconds `9.00`, LSTM delta `-0.0182`

Top all feature movements:
- `lag_13__CT_place_ENTRANCE`: contribution `-0.005166`
- `lag_04__T_place_DUMPSTER`: contribution `-0.002238`
- `lag_00__T_place_DUMPSTER`: contribution `-0.001251`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.000787`
- `lag_02__T_place_ALLEY`: contribution `-0.000770`

Top utility-only movements:
- `lag_00__CT_A_site_active_smokes`: contribution `-0.000167`

### tick `111159`, seconds `18.00`, LSTM delta `+0.0173`

Top all feature movements:
- `lag_04__T_place_DUMPSTER`: contribution `+0.002238`
- `lag_03__T_place_IVY`: contribution `+0.001874`
- `lag_01__T_place_IVY`: contribution `+0.001755`
- `lag_02__CT_place_LONGDOG`: contribution `+0.000778`
- `lag_09__CT4__duck_amount`: contribution `+0.000751`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110903`, seconds `14.00`, LSTM delta `-0.0160`

Top all feature movements:
- `lag_03__T_place_IVY`: contribution `+0.001874`
- `lag_14__T_place_DUMPSTER`: contribution `-0.001301`
- `lag_12__T_place_ALLEY`: contribution `-0.001164`
- `lag_12__T_place_DUMPSTER`: contribution `-0.001139`
- `lag_01__CT4__duck_amount`: contribution `-0.000986`

Top utility-only movements:
- No utility movement among the top local contributors.
