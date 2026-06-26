# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `23`

## Largest probability jumps

- tick `182927`, seconds `58.00`, LSTM `0.1401`, delta `-0.3114`
- tick `183311`, seconds `64.00`, LSTM `0.0424`, delta `-0.1748`
- tick `183119`, seconds `61.00`, LSTM `0.1938`, delta `+0.1092`
- tick `182735`, seconds `55.00`, LSTM `0.3717`, delta `-0.0915`
- tick `183215`, seconds `62.50`, LSTM `0.1909`, delta `-0.0838`
- tick `182895`, seconds `57.50`, LSTM `0.4516`, delta `+0.0552`
- tick `182959`, seconds `58.50`, LSTM `0.0879`, delta `-0.0523`
- tick `182607`, seconds `53.00`, LSTM `0.4635`, delta `+0.0449`
- tick `183151`, seconds `61.50`, LSTM `0.2370`, delta `+0.0432`
- tick `183183`, seconds `62.00`, LSTM `0.2747`, delta `+0.0377`

## Top 15 local ridge features

- `lag_01__CT_place_LADDER`: coefficient `0.003609`, |coef| `0.003609`
- `lag_00__CT_place_JUNGLE`: coefficient `0.002217`, |coef| `0.002217`
- `lag_00__T_kills_last_3s`: coefficient `-0.001897`, |coef| `0.001897`
- `lag_02__CT_place_LADDER`: coefficient `0.001795`, |coef| `0.001795`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001793`, |coef| `0.001793`
- `lag_08__T_place_CONNECTOR`: coefficient `0.001690`, |coef| `0.001690`
- `lag_13__T_place_CONNECTOR`: coefficient `-0.001689`, |coef| `0.001689`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001640`, |coef| `0.001640`
- `lag_07__CT_place_UNDERPASS`: coefficient `-0.001615`, |coef| `0.001615`
- `lag_06__T_place_JUNGLE`: coefficient `-0.001561`, |coef| `0.001561`
- `lag_13__T5__duck_amount`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_00__kill_diff_last_3s`: coefficient `0.001476`, |coef| `0.001476`
- `lag_01__T_place_JUNGLE`: coefficient `-0.001370`, |coef| `0.001370`
- `lag_00__T_place_JUNGLE`: coefficient `-0.001316`, |coef| `0.001316`
- `lag_03__CT4__duck_amount`: coefficient `0.001309`, |coef| `0.001309`

## Top 10 utility ridge features

- `lag_10__T_active_infernos`: coefficient `0.000954` (raises CT win probability)
- `lag_03__T1__smoke`: coefficient `0.000915` (raises CT win probability)
- `lag_07__CT4__smoke`: coefficient `0.000851` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000814` (raises CT win probability)
- `lag_10__active_infernos_total`: coefficient `0.000743` (raises CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `0.000651` (raises CT win probability)
- `lag_12__T2__flash`: coefficient `0.000596` (raises CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `0.000584` (raises CT win probability)
- `lag_00__T_A_site_active_smokes`: coefficient `-0.000579` (lowers CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `-0.000568` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_LADDER`: coefficient `0.003609` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.002217` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001897` (lowers CT win probability)
- `lag_02__CT_place_LADDER`: coefficient `0.001795` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001793` (lowers CT win probability)
- `lag_08__T_place_CONNECTOR`: coefficient `0.001690` (raises CT win probability)
- `lag_13__T_place_CONNECTOR`: coefficient `-0.001689` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001640` (raises CT win probability)
- `lag_07__CT_place_UNDERPASS`: coefficient `-0.001615` (lowers CT win probability)
- `lag_06__T_place_JUNGLE`: coefficient `-0.001561` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `182927`, seconds `58.00`, LSTM delta `-0.3114`

Top all feature movements:
- `lag_01__CT_place_LADDER`: contribution `-0.037522`
- `lag_06__T_place_JUNGLE`: contribution `-0.020220`
- `lag_00__CT_place_JUNGLE`: contribution `-0.014225`
- `lag_07__CT_place_UNDERPASS`: contribution `-0.009362`
- `lag_08__T_place_CONNECTOR`: contribution `-0.008185`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `183311`, seconds `64.00`, LSTM delta `-0.1748`

Top all feature movements:
- `lag_03__CT_shots_fired_sum`: contribution `-0.010837`
- `lag_13__CT_place_LADDER`: contribution `-0.009975`
- `lag_08__T_place_JUNGLE`: contribution `-0.009955`
- `lag_00__T_place_CONNECTOR`: contribution `-0.008681`
- `lag_06__CT_shots_fired_sum`: contribution `-0.007044`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `183119`, seconds `61.00`, LSTM delta `+0.1092`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.011396`
- `lag_00__T_place_CONNECTOR`: contribution `+0.008681`
- `lag_13__T_place_CONNECTOR`: contribution `+0.008179`
- `lag_07__CT_place_LADDER`: contribution `+0.008071`
- `lag_02__T_place_JUNGLE`: contribution `+0.006523`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `182735`, seconds `55.00`, LSTM delta `-0.0915`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `-0.017041`
- `lag_01__CT_place_UNDERPASS`: contribution `-0.006185`
- `lag_13__T5__duck_amount`: contribution `-0.005657`
- `lag_07__T_place_CONNECTOR`: contribution `+0.004662`
- `lag_10__T_place_CONNECTOR`: contribution `-0.004203`

Top utility-only movements:
- `lag_04__T_B_site_active_infernos`: contribution `-0.001839`

### tick `183215`, seconds `62.50`, LSTM delta `-0.0838`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.018233`
- `lag_00__CT1__shots_fired`: contribution `-0.008256`
- `lag_15__T_place_JUNGLE`: contribution `-0.007830`
- `lag_03__CT_shots_fired_sum`: contribution `+0.006773`
- `lag_10__CT_place_LADDER`: contribution `-0.005919`

Top utility-only movements:
- No utility movement among the top local contributors.
