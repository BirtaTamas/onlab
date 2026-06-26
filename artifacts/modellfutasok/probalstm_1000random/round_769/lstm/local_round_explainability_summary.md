# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `8`

## Largest probability jumps

- tick `61454`, seconds `61.50`, LSTM `0.4689`, delta `-0.2555`
- tick `60078`, seconds `40.00`, LSTM `0.8643`, delta `+0.2052`
- tick `61582`, seconds `63.50`, LSTM `0.0500`, delta `-0.2020`
- tick `59950`, seconds `38.00`, LSTM `0.6296`, delta `+0.1489`
- tick `61422`, seconds `61.00`, LSTM `0.7245`, delta `-0.1271`
- tick `61518`, seconds `62.50`, LSTM `0.2866`, delta `-0.0958`
- tick `61486`, seconds `62.00`, LSTM `0.3824`, delta `-0.0866`
- tick `58222`, seconds `11.00`, LSTM `0.4418`, delta `-0.0380`
- tick `61550`, seconds `63.00`, LSTM `0.2521`, delta `-0.0345`
- tick `60814`, seconds `51.50`, LSTM `0.8480`, delta `+0.0292`

## Top 15 local ridge features

- `lag_15__T_place_PIPE`: coefficient `0.002736`, |coef| `0.002736`
- `lag_03__CT_place_LOBBY`: coefficient `-0.002673`, |coef| `0.002673`
- `lag_00__CT_place_UPPERPARK`: coefficient `-0.002378`, |coef| `0.002378`
- `lag_00__kill_diff_last_3s`: coefficient `0.002302`, |coef| `0.002302`
- `lag_00__damage_diff_last_5s`: coefficient `0.002239`, |coef| `0.002239`
- `lag_00__T_kills_last_3s`: coefficient `-0.002124`, |coef| `0.002124`
- `lag_11__T_place_ALLEY`: coefficient `0.001888`, |coef| `0.001888`
- `lag_02__CT_place_LOBBY`: coefficient `-0.001868`, |coef| `0.001868`
- `lag_14__T_place_PIPE`: coefficient `0.001795`, |coef| `0.001795`
- `lag_01__T_kills_last_3s`: coefficient `-0.001730`, |coef| `0.001730`
- `lag_00__T_place_UPPERPARK`: coefficient `-0.001677`, |coef| `0.001677`
- `lag_01__kill_diff_last_3s`: coefficient `0.001658`, |coef| `0.001658`
- `lag_01__damage_diff_last_5s`: coefficient `0.001570`, |coef| `0.001570`
- `lag_00__T_damage_last_5s`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_12__T_place_ALLEY`: coefficient `0.001524`, |coef| `0.001524`

## Top 10 utility ridge features

- `lag_13__CT4__molly`: coefficient `-0.001088` (lowers CT win probability)
- `lag_08__T5__molly`: coefficient `0.000923` (raises CT win probability)
- `lag_08__T1__smoke`: coefficient `0.000904` (raises CT win probability)
- `lag_10__CT4__flash`: coefficient `-0.000852` (lowers CT win probability)
- `lag_08__T1__molly`: coefficient `-0.000807` (lowers CT win probability)
- `lag_12__CT4__molly`: coefficient `-0.000769` (lowers CT win probability)
- `lag_12__CT5__flash`: coefficient `0.000768` (raises CT win probability)
- `lag_01__CT1__flash`: coefficient `0.000756` (raises CT win probability)
- `lag_07__T5__smoke`: coefficient `-0.000749` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000744` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_PIPE`: coefficient `0.002736` (raises CT win probability)
- `lag_03__CT_place_LOBBY`: coefficient `-0.002673` (lowers CT win probability)
- `lag_00__CT_place_UPPERPARK`: coefficient `-0.002378` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002302` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002239` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002124` (lowers CT win probability)
- `lag_11__T_place_ALLEY`: coefficient `0.001888` (raises CT win probability)
- `lag_02__CT_place_LOBBY`: coefficient `-0.001868` (lowers CT win probability)
- `lag_14__T_place_PIPE`: coefficient `0.001795` (raises CT win probability)
- `lag_01__T_kills_last_3s`: coefficient `-0.001730` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `61454`, seconds `61.50`, LSTM delta `-0.2555`

Top all feature movements:
- `lag_15__T_place_PIPE`: contribution `-0.034955`
- `lag_03__CT_place_LOBBY`: contribution `-0.021883`
- `lag_00__CT_place_UPPERPARK`: contribution `-0.016924`
- `lag_15__CT_place_LOBBY`: contribution `-0.008840`
- `lag_11__T_place_ALLEY`: contribution `-0.008000`

Top utility-only movements:
- `lag_13__CT4__molly`: contribution `-0.002681`

### tick `60078`, seconds `40.00`, LSTM delta `+0.2052`

Top all feature movements:
- `lag_01__CT_place_CONSTRUCTION`: contribution `+0.015949`
- `lag_00__T_place_UPPERPARK`: contribution `+0.008845`
- `lag_06__CT_place_BACKOFA`: contribution `+0.006846`
- `lag_07__CT3__is_scoped`: contribution `+0.006046`
- `lag_03__CT_place_WATER`: contribution `+0.005776`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61582`, seconds `63.50`, LSTM delta `-0.2020`

Top all feature movements:
- `lag_07__CT_place_LOBBY`: contribution `-0.009515`
- `lag_00__T_kills_last_3s`: contribution `-0.006729`
- `lag_01__CT_place_WATER`: contribution `-0.006118`
- `lag_15__T_place_ALLEY`: contribution `-0.005928`
- `lag_00__kill_diff_last_3s`: contribution `-0.005540`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `59950`, seconds `38.00`, LSTM delta `+0.1489`

Top all feature movements:
- `lag_00__T_place_UPPERPARK`: contribution `+0.008845`
- `lag_15__CT_place_CONSTRUCTION`: contribution `+0.007008`
- `lag_14__CT_place_CONSTRUCTION`: contribution `+0.005678`
- `lag_00__kill_diff_last_3s`: contribution `+0.005540`
- `lag_00__damage_diff_last_5s`: contribution `+0.005000`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.002389`

### tick `61422`, seconds `61.00`, LSTM delta `-0.1271`

Top all feature movements:
- `lag_14__T_place_PIPE`: contribution `-0.022930`
- `lag_02__CT_place_LOBBY`: contribution `-0.015292`
- `lag_00__T_kills_last_3s`: contribution `-0.006729`
- `lag_00__kill_diff_last_3s`: contribution `-0.005540`
- `lag_10__T_place_ALLEY`: contribution `-0.005441`

Top utility-only movements:
- `lag_12__CT4__molly`: contribution `-0.001894`
