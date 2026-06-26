# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `13`

## Largest probability jumps

- tick `113759`, seconds `54.00`, LSTM `0.2421`, delta `-0.2481`
- tick `113791`, seconds `54.50`, LSTM `0.1603`, delta `-0.0818`
- tick `113919`, seconds `56.50`, LSTM `0.0739`, delta `-0.0372`
- tick `113823`, seconds `55.00`, LSTM `0.1273`, delta `-0.0331`
- tick `112095`, seconds `28.00`, LSTM `0.4815`, delta `+0.0182`
- tick `113855`, seconds `55.50`, LSTM `0.1096`, delta `-0.0176`
- tick `113567`, seconds `51.00`, LSTM `0.4750`, delta `-0.0142`
- tick `111935`, seconds `25.50`, LSTM `0.4748`, delta `-0.0141`
- tick `110975`, seconds `10.50`, LSTM `0.4976`, delta `+0.0140`
- tick `110463`, seconds `2.50`, LSTM `0.5336`, delta `+0.0138`

## Top 15 local ridge features

- `lag_04__T_place_CONNECTOR`: coefficient `-0.003664`, |coef| `0.003664`
- `lag_00__CT_place_LOWERPARK`: coefficient `0.003553`, |coef| `0.003553`
- `lag_00__T_place_ALLEY`: coefficient `0.002885`, |coef| `0.002885`
- `lag_00__T_place_LOWERPARK`: coefficient `-0.002790`, |coef| `0.002790`
- `lag_00__T_place_TSTAIRS`: coefficient `-0.002625`, |coef| `0.002625`
- `lag_07__T_place_FOUNTAIN`: coefficient `-0.002590`, |coef| `0.002590`
- `lag_01__T2__duck_amount`: coefficient `-0.002487`, |coef| `0.002487`
- `lag_14__T_place_ALLEY`: coefficient `-0.002300`, |coef| `0.002300`
- `lag_00__CT2__alive`: coefficient `0.002133`, |coef| `0.002133`
- `lag_00__T_kills_last_3s`: coefficient `-0.002131`, |coef| `0.002131`
- `lag_00__CT2__hp`: coefficient `0.002109`, |coef| `0.002109`
- `lag_00__T_place_FOUNTAIN`: coefficient `0.002098`, |coef| `0.002098`
- `lag_00__CT2__armor`: coefficient `0.001997`, |coef| `0.001997`
- `lag_07__T_place_TUNNELS`: coefficient `0.001957`, |coef| `0.001957`
- `lag_01__CT_place_LOWERPARK`: coefficient `0.001942`, |coef| `0.001942`

## Top 10 utility ridge features

- `lag_00__T1__smoke`: coefficient `-0.000406` (lowers CT win probability)
- `lag_15__T1__smoke`: coefficient `0.000354` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000290` (lowers CT win probability)
- `lag_08__T1__smoke`: coefficient `0.000280` (raises CT win probability)
- `lag_14__T1__smoke`: coefficient `0.000255` (raises CT win probability)
- `lag_03__T2__molly`: coefficient `0.000238` (raises CT win probability)
- `lag_10__T2__molly`: coefficient `0.000236` (raises CT win probability)
- `lag_11__T1__smoke`: coefficient `0.000234` (raises CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.000231` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000211` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_CONNECTOR`: coefficient `-0.003664` (lowers CT win probability)
- `lag_00__CT_place_LOWERPARK`: coefficient `0.003553` (raises CT win probability)
- `lag_00__T_place_ALLEY`: coefficient `0.002885` (raises CT win probability)
- `lag_00__T_place_LOWERPARK`: coefficient `-0.002790` (lowers CT win probability)
- `lag_00__T_place_TSTAIRS`: coefficient `-0.002625` (lowers CT win probability)
- `lag_07__T_place_FOUNTAIN`: coefficient `-0.002590` (lowers CT win probability)
- `lag_01__T2__duck_amount`: coefficient `-0.002487` (lowers CT win probability)
- `lag_14__T_place_ALLEY`: coefficient `-0.002300` (lowers CT win probability)
- `lag_00__CT2__alive`: coefficient `0.002133` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002131` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `113759`, seconds `54.00`, LSTM delta `-0.2481`

Top all feature movements:
- `lag_04__T_place_CONNECTOR`: contribution `-0.017743`
- `lag_00__CT_place_LOWERPARK`: contribution `-0.015872`
- `lag_00__T_place_TSTAIRS`: contribution `-0.014882`
- `lag_07__T_place_FOUNTAIN`: contribution `-0.012241`
- `lag_00__T_place_ALLEY`: contribution `-0.012223`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113791`, seconds `54.50`, LSTM delta `-0.0818`

Top all feature movements:
- `lag_01__CT_place_LOWERPARK`: contribution `-0.008675`
- `lag_05__T_place_CONNECTOR`: contribution `-0.007435`
- `lag_01__T_place_TSTAIRS`: contribution `-0.006435`
- `lag_01__T_place_LOWERPARK`: contribution `-0.006120`
- `lag_15__T_place_ALLEY`: contribution `-0.004831`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113919`, seconds `56.50`, LSTM delta `-0.0372`

Top all feature movements:
- `lag_00__T_place_LOWERPARK`: contribution `-0.011251`
- `lag_00__T_place_FOUNTAIN`: contribution `-0.009918`
- `lag_05__T_place_LOWERPARK`: contribution `-0.003867`
- `lag_09__T_place_CONNECTOR`: contribution `-0.003742`
- `lag_04__T1__duck_amount`: contribution `+0.003689`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `113823`, seconds `55.00`, LSTM delta `-0.0331`

Top all feature movements:
- `lag_02__CT_place_LOWERPARK`: contribution `-0.004108`
- `lag_06__T_place_CONNECTOR`: contribution `-0.003869`
- `lag_02__T_place_LOWERPARK`: contribution `-0.003838`
- `lag_09__T_place_TUNNELS`: contribution `-0.002541`
- `lag_07__T1__duck_amount`: contribution `+0.002477`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `112095`, seconds `28.00`, LSTM delta `+0.0182`

Top all feature movements:
- `lag_00__T_place_PLAYGROUND`: contribution `+0.015709`
- `lag_00__T_place_FOUNTAIN`: contribution `-0.009918`
- `lag_10__T5__duck_amount`: contribution `-0.002436`
- `lag_00__CT2__is_walking`: contribution `+0.001922`
- `lag_13__T3__is_walking`: contribution `-0.001875`

Top utility-only movements:
- No utility movement among the top local contributors.
