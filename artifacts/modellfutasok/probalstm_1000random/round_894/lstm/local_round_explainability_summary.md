# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `15`

## Largest probability jumps

- tick `143900`, seconds `59.50`, LSTM `0.9080`, delta `+0.1600`
- tick `141372`, seconds `20.00`, LSTM `0.8779`, delta `+0.1056`
- tick `142492`, seconds `37.50`, LSTM `0.7509`, delta `-0.0748`
- tick `141308`, seconds `19.00`, LSTM `0.7390`, delta `+0.0620`
- tick `140956`, seconds `13.50`, LSTM `0.6370`, delta `+0.0388`
- tick `140348`, seconds `4.00`, LSTM `0.6531`, delta `-0.0385`
- tick `141628`, seconds `24.00`, LSTM `0.8398`, delta `-0.0359`
- tick `144444`, seconds `68.00`, LSTM `0.9742`, delta `+0.0344`
- tick `141340`, seconds `19.50`, LSTM `0.7723`, delta `+0.0334`
- tick `143580`, seconds `54.50`, LSTM `0.7973`, delta `+0.0333`

## Top 15 local ridge features

- `lag_13__T_place_CONSTRUCTION`: coefficient `0.003134`, |coef| `0.003134`
- `lag_00__CT_place_RESTROOM`: coefficient `0.001749`, |coef| `0.001749`
- `lag_13__T_place_WATER`: coefficient `-0.001597`, |coef| `0.001597`
- `lag_00__kill_diff_last_3s`: coefficient `0.001440`, |coef| `0.001440`
- `lag_00__CT_kills_last_3s`: coefficient `0.001376`, |coef| `0.001376`
- `lag_10__CT_place_WATER`: coefficient `0.001343`, |coef| `0.001343`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001322`, |coef| `0.001322`
- `lag_06__CT_place_LOBBY`: coefficient `0.001188`, |coef| `0.001188`
- `lag_00__T2__utility_total`: coefficient `-0.001114`, |coef| `0.001114`
- `lag_00__T2__flash`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_09__T_place_PIPE`: coefficient `-0.001035`, |coef| `0.001035`
- `lag_03__T_place_CONSTRUCTION`: coefficient `0.000986`, |coef| `0.000986`
- `lag_00__T2__has_bomb`: coefficient `-0.000971`, |coef| `0.000971`
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.000935`, |coef| `0.000935`
- `lag_10__CT2__duck_amount`: coefficient `0.000921`, |coef| `0.000921`

## Top 10 utility ridge features

- `lag_00__T2__utility_total`: coefficient `-0.001114` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001059` (lowers CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.000935` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000790` (lowers CT win probability)
- `lag_08__CT1__smoke`: coefficient `0.000723` (raises CT win probability)
- `lag_12__CT_he_last_5s`: coefficient `-0.000696` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.000682` (lowers CT win probability)
- `lag_09__CT2__smoke`: coefficient `-0.000678` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000621` (lowers CT win probability)
- `lag_05__CT2__flash`: coefficient `-0.000561` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_CONSTRUCTION`: coefficient `0.003134` (raises CT win probability)
- `lag_00__CT_place_RESTROOM`: coefficient `0.001749` (raises CT win probability)
- `lag_13__T_place_WATER`: coefficient `-0.001597` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001440` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001376` (raises CT win probability)
- `lag_10__CT_place_WATER`: coefficient `0.001343` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001322` (lowers CT win probability)
- `lag_06__CT_place_LOBBY`: coefficient `0.001188` (raises CT win probability)
- `lag_09__T_place_PIPE`: coefficient `-0.001035` (lowers CT win probability)
- `lag_03__T_place_CONSTRUCTION`: coefficient `0.000986` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `143900`, seconds `59.50`, LSTM delta `+0.1600`

Top all feature movements:
- `lag_13__T_place_CONSTRUCTION`: contribution `+0.038947`
- `lag_06__CT_place_LOBBY`: contribution `+0.009722`
- `lag_13__T_place_WATER`: contribution `+0.009117`
- `lag_10__CT_place_WATER`: contribution `+0.008163`
- `lag_00__T_place_CONNECTOR`: contribution `+0.006402`

Top utility-only movements:
- `lag_00__T2__utility_total`: contribution `+0.003656`
- `lag_09__CT_A_site_active_infernos`: contribution `+0.003298`
- `lag_00__T2__flash`: contribution `+0.003119`

### tick `141372`, seconds `20.00`, LSTM delta `+0.1056`

Top all feature movements:
- `lag_09__T_place_PIPE`: contribution `+0.013216`
- `lag_04__T_place_PLAYGROUND`: contribution `+0.012844`
- `lag_13__T_place_PLAYGROUND`: contribution `+0.008241`
- `lag_15__T_place_PLAYGROUND`: contribution `+0.006517`
- `lag_00__CT_kills_last_3s`: contribution `+0.003973`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.001074`

### tick `142492`, seconds `37.50`, LSTM delta `-0.0748`

Top all feature movements:
- `lag_00__CT_place_RESTROOM`: contribution `-0.024937`
- `lag_06__CT_place_WALKWAY`: contribution `-0.003886`
- `lag_00__kill_diff_last_3s`: contribution `-0.003467`
- `lag_00__T2__duck_amount`: contribution `-0.002422`
- `lag_00__T_shots_fired_sum`: contribution `-0.002417`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141308`, seconds `19.00`, LSTM delta `+0.0620`

Top all feature movements:
- `lag_11__T_place_PLAYGROUND`: contribution `+0.012669`
- `lag_13__T_place_PLAYGROUND`: contribution `+0.008241`
- `lag_07__T_place_PIPE`: contribution `+0.006634`
- `lag_15__CT_flashes_last_5s`: contribution `+0.004596`
- `lag_10__CT_place_WALKWAY`: contribution `+0.004283`

Top utility-only movements:
- `lag_15__CT_flashes_last_5s`: contribution `+0.004596`

### tick `140956`, seconds `13.50`, LSTM delta `+0.0388`

Top all feature movements:
- `lag_12__CT_he_last_5s`: contribution `+0.012778`
- `lag_04__CT_flashes_last_5s`: contribution `+0.004650`
- `lag_10__CT_place_WALKWAY`: contribution `+0.004283`
- `lag_11__CT_place_BACKOFA`: contribution `+0.003453`
- `lag_08__CT_place_BACKOFA`: contribution `-0.003373`

Top utility-only movements:
- `lag_12__CT_he_last_5s`: contribution `+0.012778`
- `lag_04__CT_flashes_last_5s`: contribution `+0.004650`
- `lag_14__CT_flashes_last_5s`: contribution `+0.002926`
- `lag_09__CT_active_infernos`: contribution `-0.001571`
