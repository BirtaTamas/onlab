# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `85940`, seconds `67.50`, LSTM `0.8370`, delta `+0.1419`
- tick `85972`, seconds `68.00`, LSTM `0.9140`, delta `+0.0770`
- tick `85908`, seconds `67.00`, LSTM `0.6951`, delta `-0.0523`
- tick `86004`, seconds `68.50`, LSTM `0.9537`, delta `+0.0397`
- tick `82196`, seconds `9.00`, LSTM `0.7425`, delta `+0.0376`
- tick `82260`, seconds `10.00`, LSTM `0.7186`, delta `-0.0301`
- tick `84692`, seconds `48.00`, LSTM `0.7297`, delta `+0.0282`
- tick `82388`, seconds `12.00`, LSTM `0.7332`, delta `+0.0277`
- tick `84756`, seconds `49.00`, LSTM `0.7386`, delta `+0.0273`
- tick `83316`, seconds `26.50`, LSTM `0.7042`, delta `-0.0253`

## Top 15 local ridge features

- `lag_02__T_flashes_last_5s`: coefficient `-0.001681`, |coef| `0.001681`
- `lag_13__T_flashes_last_5s`: coefficient `0.001246`, |coef| `0.001246`
- `lag_03__T_place_TUNNELSTAIRS`: coefficient `-0.001228`, |coef| `0.001228`
- `lag_00__damage_diff_last_5s`: coefficient `0.001032`, |coef| `0.001032`
- `lag_00__CT_kills_last_3s`: coefficient `0.001020`, |coef| `0.001020`
- `lag_13__bomb_events_last_5s`: coefficient `0.001012`, |coef| `0.001012`
- `lag_00__T1__duck_amount`: coefficient `-0.000999`, |coef| `0.000999`
- `lag_00__CT3__shots_fired`: coefficient `0.000968`, |coef| `0.000968`
- `lag_01__CT3__shots_fired`: coefficient `0.000964`, |coef| `0.000964`
- `lag_00__CT_damage_last_5s`: coefficient `0.000852`, |coef| `0.000852`
- `lag_02__T_place_MIDDOORS`: coefficient `0.000843`, |coef| `0.000843`
- `lag_15__CT5__duck_amount`: coefficient `-0.000829`, |coef| `0.000829`
- `lag_03__T_place_LOWERTUNNEL`: coefficient `-0.000817`, |coef| `0.000817`
- `lag_00__CT_place_HOLE`: coefficient `0.000783`, |coef| `0.000783`
- `lag_14__CT4__is_walking`: coefficient `0.000776`, |coef| `0.000776`

## Top 10 utility ridge features

- `lag_02__T_flashes_last_5s`: coefficient `-0.001681` (lowers CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `0.001246` (raises CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000717` (lowers CT win probability)
- `lag_14__T_flashes_last_5s`: coefficient `0.000650` (raises CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `-0.000619` (lowers CT win probability)
- `lag_09__T2__smoke`: coefficient `-0.000573` (lowers CT win probability)
- `lag_06__T_smokes_last_5s`: coefficient `-0.000539` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `0.000465` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000390` (lowers CT win probability)
- `lag_02__T_smokes_last_5s`: coefficient `-0.000390` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_TUNNELSTAIRS`: coefficient `-0.001228` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001032` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001020` (raises CT win probability)
- `lag_13__bomb_events_last_5s`: coefficient `0.001012` (raises CT win probability)
- `lag_00__T1__duck_amount`: coefficient `-0.000999` (lowers CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `0.000968` (raises CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `0.000964` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.000852` (raises CT win probability)
- `lag_02__T_place_MIDDOORS`: coefficient `0.000843` (raises CT win probability)
- `lag_15__CT5__duck_amount`: coefficient `-0.000829` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `85940`, seconds `67.50`, LSTM delta `+0.1419`

Top all feature movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.015230`
- `lag_13__T_flashes_last_5s`: contribution `+0.011286`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `+0.008573`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.004985`
- `lag_13__bomb_events_last_5s`: contribution `+0.004231`

Top utility-only movements:
- `lag_02__T_flashes_last_5s`: contribution `+0.015230`
- `lag_13__T_flashes_last_5s`: contribution `+0.011286`

### tick `85972`, seconds `68.00`, LSTM delta `+0.0770`

Top all feature movements:
- `lag_14__T_flashes_last_5s`: contribution `+0.005885`
- `lag_03__T_flashes_last_5s`: contribution `+0.005611`
- `lag_04__T_place_TUNNELSTAIRS`: contribution `+0.003017`
- `lag_00__CT_kills_last_3s`: contribution `+0.002946`
- `lag_04__T_place_LOWERTUNNEL`: contribution `+0.002845`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `+0.005885`
- `lag_03__T_flashes_last_5s`: contribution `+0.005611`

### tick `85908`, seconds `67.00`, LSTM delta `-0.0523`

Top all feature movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.006499`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.004449`
- `lag_01__T_flashes_last_5s`: contribution `-0.004217`
- `lag_01__T_place_MIDDOORS`: contribution `-0.003067`
- `lag_00__T1__duck_amount`: contribution `-0.002810`

Top utility-only movements:
- `lag_12__T_flashes_last_5s`: contribution `-0.006499`
- `lag_01__T_flashes_last_5s`: contribution `-0.004217`

### tick `86004`, seconds `68.50`, LSTM delta `+0.0397`

Top all feature movements:
- `lag_01__T_place_MIDDOORS`: contribution `+0.003067`
- `lag_00__CT_kills_last_3s`: contribution `+0.002946`
- `lag_04__T_flashes_last_5s`: contribution `-0.002916`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002656`
- `lag_01__CT3__shots_fired`: contribution `+0.002479`

Top utility-only movements:
- `lag_04__T_flashes_last_5s`: contribution `-0.002916`
- `lag_15__T_flashes_last_5s`: contribution `+0.001906`

### tick `82196`, seconds `9.00`, LSTM delta `+0.0376`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.008743`
- `lag_06__T_smokes_last_5s`: contribution `+0.007902`
- `lag_04__CT_place_BDOORS`: contribution `+0.003956`
- `lag_01__T_place_OUTSIDETUNNEL`: contribution `+0.003060`
- `lag_04__T_flashes_last_5s`: contribution `+0.002916`

Top utility-only movements:
- `lag_06__T_smokes_last_5s`: contribution `+0.007902`
- `lag_04__T_flashes_last_5s`: contribution `+0.002916`
