# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `65319`, seconds `15.00`, LSTM `0.3925`, delta `+0.1774`
- tick `65191`, seconds `13.00`, LSTM `0.3621`, delta `-0.1634`
- tick `66023`, seconds `26.00`, LSTM `0.1337`, delta `-0.1541`
- tick `65223`, seconds `13.50`, LSTM `0.2534`, delta `-0.1088`
- tick `65703`, seconds `21.00`, LSTM `0.3355`, delta `+0.0569`
- tick `65511`, seconds `18.00`, LSTM `0.3971`, delta `-0.0557`
- tick `65607`, seconds `19.50`, LSTM `0.3153`, delta `-0.0505`
- tick `66279`, seconds `30.00`, LSTM `0.0131`, delta `-0.0422`
- tick `65991`, seconds `25.50`, LSTM `0.2878`, delta `-0.0416`
- tick `66055`, seconds `26.50`, LSTM `0.0934`, delta `-0.0403`

## Top 15 local ridge features

- `lag_15__CT_place_SHOP`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_14__T_he_last_5s`: coefficient `-0.001531`, |coef| `0.001531`
- `lag_09__T_place_HOUSE`: coefficient `-0.001344`, |coef| `0.001344`
- `lag_08__T_place_HOUSE`: coefficient `-0.001266`, |coef| `0.001266`
- `lag_00__kill_diff_last_3s`: coefficient `0.001223`, |coef| `0.001223`
- `lag_00__CT5__flash_duration`: coefficient `-0.001206`, |coef| `0.001206`
- `lag_07__CT4__flash_duration`: coefficient `0.001133`, |coef| `0.001133`
- `lag_00__T_kills_last_3s`: coefficient `-0.001113`, |coef| `0.001113`
- `lag_10__T2__flash_duration`: coefficient `-0.001111`, |coef| `0.001111`
- `lag_10__T4__flash_duration`: coefficient `-0.001106`, |coef| `0.001106`
- `lag_10__T_flash_duration_sum`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_15__T_place_HOUSE`: coefficient `0.001032`, |coef| `0.001032`
- `lag_08__CT_place_JUNGLE`: coefficient `0.001000`, |coef| `0.001000`
- `lag_10__T_he_last_5s`: coefficient `0.000999`, |coef| `0.000999`
- `lag_01__T2__flash_duration`: coefficient `0.000995`, |coef| `0.000995`

## Top 10 utility ridge features

- `lag_14__T_he_last_5s`: coefficient `-0.001531` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.001206` (lowers CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001133` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `-0.001111` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.001106` (lowers CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `-0.001063` (lowers CT win probability)
- `lag_10__T_he_last_5s`: coefficient `0.000999` (raises CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.000995` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.000895` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.000893` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SHOP`: coefficient `-0.001568` (lowers CT win probability)
- `lag_09__T_place_HOUSE`: coefficient `-0.001344` (lowers CT win probability)
- `lag_08__T_place_HOUSE`: coefficient `-0.001266` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001223` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001113` (lowers CT win probability)
- `lag_15__T_place_HOUSE`: coefficient `0.001032` (raises CT win probability)
- `lag_08__CT_place_JUNGLE`: coefficient `0.001000` (raises CT win probability)
- `lag_04__CT_place_CATWALK`: coefficient `-0.000975` (lowers CT win probability)
- `lag_14__CT_place_JUNGLE`: coefficient `-0.000943` (lowers CT win probability)
- `lag_00__T_place_CATWALK`: coefficient `-0.000874` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65319`, seconds `15.00`, LSTM delta `+0.1774`

Top all feature movements:
- `lag_14__T_he_last_5s`: contribution `+0.019982`
- `lag_07__CT4__flash_duration`: contribution `+0.008823`
- `lag_15__CT_place_SHOP`: contribution `+0.007867`
- `lag_09__T_place_HOUSE`: contribution `+0.005908`
- `lag_08__T_place_HOUSE`: contribution `+0.005568`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `+0.019982`
- `lag_07__CT4__flash_duration`: contribution `+0.008823`
- `lag_14__T1__flash_duration`: contribution `+0.003961`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003526`

### tick `65191`, seconds `13.00`, LSTM delta `-0.1634`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `-0.013039`
- `lag_15__CT_place_SHOP`: contribution `-0.007867`
- `lag_03__CT4__flash_duration`: contribution `-0.006294`
- `lag_08__T_place_HOUSE`: contribution `-0.005568`
- `lag_11__T_place_SIDEALLEY`: contribution `-0.004316`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `-0.013039`
- `lag_03__CT4__flash_duration`: contribution `-0.006294`
- `lag_10__T1__flash_duration`: contribution `-0.003824`
- `lag_00__CT5__utility_total`: contribution `-0.002266`
- `lag_03__CT_flash_duration_sum`: contribution `-0.001979`

### tick `66023`, seconds `26.00`, LSTM delta `-0.1541`

Top all feature movements:
- `lag_10__T2__flash_duration`: contribution `-0.008808`
- `lag_10__T4__flash_duration`: contribution `-0.008172`
- `lag_10__T_flash_duration_sum`: contribution `-0.007027`
- `lag_01__T2__flash_duration`: contribution `-0.006933`
- `lag_08__CT_place_JUNGLE`: contribution `-0.006413`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `-0.008808`
- `lag_10__T4__flash_duration`: contribution `-0.008172`
- `lag_10__T_flash_duration_sum`: contribution `-0.007027`
- `lag_01__T2__flash_duration`: contribution `-0.006933`
- `lag_15__CT4__flash_duration`: contribution `-0.005008`

### tick `65223`, seconds `13.50`, LSTM delta `-0.1088`

Top all feature movements:
- `lag_11__T_he_last_5s`: contribution `-0.011471`
- `lag_04__CT4__flash_duration`: contribution `-0.006759`
- `lag_00__CT5__flash_duration`: contribution `-0.006605`
- `lag_09__T_place_HOUSE`: contribution `-0.005908`
- `lag_04__CT_place_CATWALK`: contribution `-0.003886`

Top utility-only movements:
- `lag_11__T_he_last_5s`: contribution `-0.011471`
- `lag_04__CT4__flash_duration`: contribution `-0.006759`
- `lag_00__CT5__flash_duration`: contribution `-0.006605`
- `lag_11__T1__flash_duration`: contribution `-0.002216`
- `lag_04__CT_flash_duration_sum`: contribution `-0.002167`

### tick `65703`, seconds `21.00`, LSTM delta `+0.0569`

Top all feature movements:
- `lag_00__T2__flash_duration`: contribution `+0.007081`
- `lag_05__CT4__flash_duration`: contribution `+0.005278`
- `lag_00__T_flash_duration_sum`: contribution `+0.004337`
- `lag_00__T4__flash_duration`: contribution `+0.003874`
- `lag_04__CT_place_TRUCK`: contribution `-0.003435`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.007081`
- `lag_05__CT4__flash_duration`: contribution `+0.005278`
- `lag_00__T_flash_duration_sum`: contribution `+0.004337`
- `lag_00__T4__flash_duration`: contribution `+0.003874`
- `lag_15__CT5__flash_duration`: contribution `-0.001274`
