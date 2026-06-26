# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-mouz-vs-virtuspro-bo3-RgsQGjmI__aLZMP1KntvtG/mouz-vs-virtus-pro-m2-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `121997`, seconds `18.50`, LSTM `0.1925`, delta `-0.1221`
- tick `121837`, seconds `16.00`, LSTM `0.4760`, delta `-0.1062`
- tick `124973`, seconds `65.00`, LSTM `0.4353`, delta `-0.0865`
- tick `121901`, seconds `17.00`, LSTM `0.3514`, delta `-0.0861`
- tick `121677`, seconds `13.50`, LSTM `0.5137`, delta `+0.0825`
- tick `126413`, seconds `87.50`, LSTM `0.0616`, delta `-0.0781`
- tick `125037`, seconds `66.00`, LSTM `0.3151`, delta `-0.0648`
- tick `122029`, seconds `19.00`, LSTM `0.2519`, delta `+0.0594`
- tick `125517`, seconds `73.50`, LSTM `0.1288`, delta `-0.0579`
- tick `121965`, seconds `18.00`, LSTM `0.3146`, delta `-0.0564`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002009`, |coef| `0.002009`
- `lag_00__damage_diff_last_5s`: coefficient `0.001772`, |coef| `0.001772`
- `lag_15__CT_place_SHOP`: coefficient `0.001772`, |coef| `0.001772`
- `lag_11__CT_place_JUNGLE`: coefficient `0.001742`, |coef| `0.001742`
- `lag_00__T_kills_last_3s`: coefficient `-0.001647`, |coef| `0.001647`
- `lag_09__CT_place_TRUCK`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_06__CT_place_UNDERPASS`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_01__CT_place_TRUCK`: coefficient `-0.001424`, |coef| `0.001424`
- `lag_07__CT_place_UNDERPASS`: coefficient `-0.001409`, |coef| `0.001409`
- `lag_14__CT3__duck_amount`: coefficient `0.001366`, |coef| `0.001366`
- `lag_14__CT_place_CONNECTOR`: coefficient `0.001351`, |coef| `0.001351`
- `lag_01__CT_place_CATWALK`: coefficient `0.001338`, |coef| `0.001338`
- `lag_05__CT_place_UNDERPASS`: coefficient `-0.001324`, |coef| `0.001324`
- `lag_13__CT3__duck_amount`: coefficient `0.001309`, |coef| `0.001309`
- `lag_02__CT_place_CATWALK`: coefficient `0.001294`, |coef| `0.001294`

## Top 10 utility ridge features

- `lag_00__CT2__smoke`: coefficient `0.000789` (raises CT win probability)
- `lag_09__T2__smoke`: coefficient `0.000656` (raises CT win probability)
- `lag_13__T5__smoke`: coefficient `0.000638` (raises CT win probability)
- `lag_10__T2__smoke`: coefficient `0.000624` (raises CT win probability)
- `lag_01__CT2__smoke`: coefficient `0.000614` (raises CT win probability)
- `lag_08__T2__smoke`: coefficient `0.000606` (raises CT win probability)
- `lag_09__T_smoke_inv`: coefficient `0.000498` (raises CT win probability)
- `lag_10__T4__molly`: coefficient `0.000491` (raises CT win probability)
- `lag_10__T4__smoke`: coefficient `0.000491` (raises CT win probability)
- `lag_10__T_smoke_inv`: coefficient `0.000478` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002009` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001772` (raises CT win probability)
- `lag_15__CT_place_SHOP`: coefficient `0.001772` (raises CT win probability)
- `lag_11__CT_place_JUNGLE`: coefficient `0.001742` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001647` (lowers CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `-0.001490` (lowers CT win probability)
- `lag_06__CT_place_UNDERPASS`: coefficient `-0.001434` (lowers CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `-0.001424` (lowers CT win probability)
- `lag_07__CT_place_UNDERPASS`: coefficient `-0.001409` (lowers CT win probability)
- `lag_14__CT3__duck_amount`: coefficient `0.001366` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121997`, seconds `18.50`, LSTM delta `-0.1221`

Top all feature movements:
- `lag_09__CT_place_TRUCK`: contribution `-0.009612`
- `lag_06__CT_place_TRUCK`: contribution `-0.008056`
- `lag_00__T1__duck_amount`: contribution `-0.005034`
- `lag_00__damage_diff_last_5s`: contribution `-0.003998`
- `lag_10__CT2__duck_amount`: contribution `-0.003561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121837`, seconds `16.00`, LSTM delta `-0.1062`

Top all feature movements:
- `lag_01__CT_place_TRUCK`: contribution `-0.009183`
- `lag_00__T_kills_last_3s`: contribution `-0.005217`
- `lag_00__kill_diff_last_3s`: contribution `-0.004835`
- `lag_04__CT_place_TRUCK`: contribution `-0.003995`
- `lag_00__damage_diff_last_5s`: contribution `-0.003558`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `124973`, seconds `65.00`, LSTM delta `-0.0865`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005217`
- `lag_00__kill_diff_last_3s`: contribution `-0.004835`
- `lag_14__CT_place_CONNECTOR`: contribution `-0.004832`
- `lag_00__CT_place_CATWALK`: contribution `-0.003365`
- `lag_12__CT2__duck_amount`: contribution `-0.003267`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121901`, seconds `17.00`, LSTM delta `-0.0861`

Top all feature movements:
- `lag_06__CT_place_TRUCK`: contribution `-0.008056`
- `lag_13__CT5__duck_amount`: contribution `-0.003940`
- `lag_02__T_kills_last_3s`: contribution `-0.003260`
- `lag_07__CT2__duck_amount`: contribution `-0.002933`
- `lag_06__T5__duck_amount`: contribution `-0.002816`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `121677`, seconds `13.50`, LSTM delta `+0.0825`

Top all feature movements:
- `lag_10__T_place_HOUSE`: contribution `+0.009804`
- `lag_15__CT_place_SHOP`: contribution `+0.008886`
- `lag_11__CT_place_SHOP`: contribution `+0.005437`
- `lag_01__CT_place_CATWALK`: contribution `+0.005329`
- `lag_00__kill_diff_last_3s`: contribution `+0.004835`

Top utility-only movements:
- No utility movement among the top local contributors.
