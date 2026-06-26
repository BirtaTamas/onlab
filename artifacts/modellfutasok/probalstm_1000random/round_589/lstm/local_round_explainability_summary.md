# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `10`

## Largest probability jumps

- tick `82761`, seconds `77.00`, LSTM `0.0368`, delta `-0.0496`
- tick `77865`, seconds `0.50`, LSTM `0.0391`, delta `-0.0492`
- tick `81769`, seconds `61.50`, LSTM `0.1084`, delta `+0.0299`
- tick `83657`, seconds `91.00`, LSTM `0.0101`, delta `-0.0211`
- tick `79017`, seconds `18.50`, LSTM `0.0867`, delta `+0.0180`
- tick `82537`, seconds `73.50`, LSTM `0.0887`, delta `-0.0162`
- tick `80457`, seconds `41.00`, LSTM `0.0696`, delta `+0.0126`
- tick `79625`, seconds `28.00`, LSTM `0.0690`, delta `-0.0122`
- tick `79145`, seconds `20.50`, LSTM `0.0933`, delta `+0.0117`
- tick `82473`, seconds `72.50`, LSTM `0.0985`, delta `-0.0116`

## Top 15 local ridge features

- `lag_00__CT4__duck_amount`: coefficient `-0.000588`, |coef| `0.000588`
- `lag_07__CT_place_TMAIN`: coefficient `-0.000586`, |coef| `0.000586`
- `lag_00__CT_place_TMAIN`: coefficient `0.000471`, |coef| `0.000471`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000447`, |coef| `0.000447`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000438`, |coef| `0.000438`
- `lag_13__T_place_TMAIN`: coefficient `-0.000430`, |coef| `0.000430`
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000384`, |coef| `0.000384`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000366`, |coef| `0.000366`
- `lag_04__CT3__is_walking`: coefficient `0.000357`, |coef| `0.000357`
- `lag_12__T_place_ALLEY`: coefficient `0.000349`, |coef| `0.000349`
- `lag_01__centroid_distance_xy`: coefficient `-0.000345`, |coef| `0.000345`
- `lag_00__CT4__is_walking`: coefficient `0.000337`, |coef| `0.000337`
- `lag_00__T_place_LONGDOG`: coefficient `0.000327`, |coef| `0.000327`
- `lag_00__T_kills_last_3s`: coefficient `-0.000321`, |coef| `0.000321`
- `lag_06__CT3__duck_amount`: coefficient `0.000318`, |coef| `0.000318`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000279` (raises CT win probability)
- `lag_01__T4__smoke`: coefficient `-0.000271` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000253` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000253` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000225` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000208` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000204` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000196` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000194` (lowers CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000185` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT4__duck_amount`: coefficient `-0.000588` (lowers CT win probability)
- `lag_07__CT_place_TMAIN`: coefficient `-0.000586` (lowers CT win probability)
- `lag_00__CT_place_TMAIN`: coefficient `0.000471` (raises CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000447` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000438` (lowers CT win probability)
- `lag_13__T_place_TMAIN`: coefficient `-0.000430` (lowers CT win probability)
- `lag_01__T_closest_enemy_dist`: coefficient `-0.000384` (lowers CT win probability)
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000366` (lowers CT win probability)
- `lag_04__CT3__is_walking`: coefficient `0.000357` (raises CT win probability)
- `lag_12__T_place_ALLEY`: coefficient `0.000349` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `82761`, seconds `77.00`, LSTM delta `-0.0496`

Top all feature movements:
- `lag_07__CT_place_TMAIN`: contribution `-0.006490`
- `lag_00__CT_place_TMAIN`: contribution `-0.005222`
- `lag_13__T_place_TMAIN`: contribution `-0.001667`
- `lag_06__CT3__duck_amount`: contribution `-0.001183`
- `lag_11__bomb_events_last_5s`: contribution `-0.001108`

Top utility-only movements:
- `lag_01__T4__smoke`: contribution `-0.000590`

### tick `77865`, seconds `0.50`, LSTM delta `-0.0492`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002139`
- `lag_01__T_place_TSPAWN`: contribution `-0.001938`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001539`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001478`
- `lag_01__centroid_distance_xy`: contribution `-0.001317`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000858`
- `lag_01__molly_inv_diff`: contribution `-0.000705`
- `lag_01__smoke_inv_diff`: contribution `-0.000572`
- `lag_01__T4__utility_total`: contribution `-0.000561`
- `lag_01__flash_inv_diff`: contribution `-0.000468`

### tick `81769`, seconds `61.50`, LSTM delta `+0.0299`

Top all feature movements:
- `lag_00__CT4__duck_amount`: contribution `+0.002161`
- `lag_12__T_place_IVY`: contribution `+0.001681`
- `lag_12__T_place_ALLEY`: contribution `+0.001480`
- `lag_09__T_place_ALLEY`: contribution `+0.001235`
- `lag_06__CT3__duck_amount`: contribution `+0.001179`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `83657`, seconds `91.00`, LSTM delta `-0.0211`

Top all feature movements:
- `lag_13__T_place_TMAIN`: contribution `-0.001667`
- `lag_15__T_place_TSTAIRS`: contribution `-0.001413`
- `lag_06__T_place_IVY`: contribution `-0.001113`
- `lag_00__T_kills_last_3s`: contribution `-0.001016`
- `lag_00__T1__duck_amount`: contribution `+0.000884`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `79017`, seconds `18.50`, LSTM delta `+0.0180`

Top all feature movements:
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.002250`
- `lag_11__T_place_ALLEY`: contribution `+0.001151`
- `lag_00__T_walking_count`: contribution `+0.000987`
- `lag_12__T_place_TMAIN`: contribution `+0.000948`
- `lag_14__CT2__duck_amount`: contribution `+0.000867`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `+0.000607`
