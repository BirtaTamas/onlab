# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `103566`, seconds `55.50`, LSTM `0.3341`, delta `-0.1472`
- tick `105486`, seconds `85.50`, LSTM `0.0430`, delta `-0.1382`
- tick `104110`, seconds `64.00`, LSTM `0.4357`, delta `-0.0979`
- tick `104430`, seconds `69.00`, LSTM `0.1865`, delta `-0.0645`
- tick `104174`, seconds `65.00`, LSTM `0.3607`, delta `-0.0623`
- tick `105070`, seconds `79.00`, LSTM `0.1606`, delta `-0.0581`
- tick `103630`, seconds `56.50`, LSTM `0.3951`, delta `+0.0579`
- tick `103758`, seconds `58.50`, LSTM `0.4358`, delta `+0.0453`
- tick `102446`, seconds `38.00`, LSTM `0.5808`, delta `+0.0451`
- tick `104398`, seconds `68.50`, LSTM `0.2510`, delta `-0.0432`

## Top 15 local ridge features

- `lag_00__T_place_HELL`: coefficient `-0.002377`, |coef| `0.002377`
- `lag_00__T_place_RAFTERS`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_04__T_place_HELL`: coefficient `-0.001538`, |coef| `0.001538`
- `lag_00__kill_diff_last_3s`: coefficient `0.001534`, |coef| `0.001534`
- `lag_15__T_place_HELL`: coefficient `0.001342`, |coef| `0.001342`
- `lag_01__T_place_HELL`: coefficient `-0.001325`, |coef| `0.001325`
- `lag_09__T_place_ADMIN`: coefficient `0.001228`, |coef| `0.001228`
- `lag_14__T_place_ADMIN`: coefficient `-0.001172`, |coef| `0.001172`
- `lag_00__T_kills_last_3s`: coefficient `-0.001148`, |coef| `0.001148`
- `lag_01__T_place_HUT`: coefficient `-0.001122`, |coef| `0.001122`
- `lag_13__T_place_HEAVEN`: coefficient `-0.001114`, |coef| `0.001114`
- `lag_02__CT_place_DECON`: coefficient `-0.001097`, |coef| `0.001097`
- `lag_12__T_place_ADMIN`: coefficient `0.001077`, |coef| `0.001077`
- `lag_10__T_place_ADMIN`: coefficient `0.001066`, |coef| `0.001066`
- `lag_08__T5__duck_amount`: coefficient `0.001055`, |coef| `0.001055`

## Top 10 utility ridge features

- `lag_05__T1__smoke`: coefficient `-0.000216` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `0.000141` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000139` (raises CT win probability)
- `lag_05__T1__utility_total`: coefficient `-0.000131` (lowers CT win probability)
- `lag_05__T1__flash`: coefficient `-0.000119` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `0.000102` (raises CT win probability)
- `lag_01__CT1__flash`: coefficient `0.000100` (raises CT win probability)
- `lag_10__T_utility_inv`: coefficient `0.000099` (raises CT win probability)
- `lag_10__T_flash_inv`: coefficient `0.000096` (raises CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.000094` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_HELL`: coefficient `-0.002377` (lowers CT win probability)
- `lag_00__T_place_RAFTERS`: coefficient `-0.001770` (lowers CT win probability)
- `lag_04__T_place_HELL`: coefficient `-0.001538` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001534` (raises CT win probability)
- `lag_15__T_place_HELL`: coefficient `0.001342` (raises CT win probability)
- `lag_01__T_place_HELL`: coefficient `-0.001325` (lowers CT win probability)
- `lag_09__T_place_ADMIN`: coefficient `0.001228` (raises CT win probability)
- `lag_14__T_place_ADMIN`: coefficient `-0.001172` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001148` (lowers CT win probability)
- `lag_01__T_place_HUT`: coefficient `-0.001122` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `103566`, seconds `55.50`, LSTM delta `-0.1472`

Top all feature movements:
- `lag_04__T_place_HELL`: contribution `-0.032792`
- `lag_14__T_place_ADMIN`: contribution `-0.022788`
- `lag_06__T_place_ADMIN`: contribution `-0.012313`
- `lag_04__T_place_ADMIN`: contribution `-0.011101`
- `lag_01__T_place_HUT`: contribution `-0.010463`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105486`, seconds `85.50`, LSTM delta `-0.1382`

Top all feature movements:
- `lag_00__T_place_RAFTERS`: contribution `-0.046317`
- `lag_09__T_place_ADMIN`: contribution `-0.023865`
- `lag_13__T_place_HEAVEN`: contribution `-0.013672`
- `lag_03__T_place_HEAVEN`: contribution `-0.008033`
- `lag_05__T_place_HUT`: contribution `-0.006573`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `104110`, seconds `64.00`, LSTM delta `-0.0979`

Top all feature movements:
- `lag_15__T_place_HELL`: contribution `-0.028618`
- `lag_08__T5__duck_amount`: contribution `-0.003975`
- `lag_00__kill_diff_last_3s`: contribution `-0.003693`
- `lag_00__T_kills_last_3s`: contribution `-0.003638`
- `lag_04__CT2__duck_amount`: contribution `-0.003516`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `104430`, seconds `69.00`, LSTM delta `-0.0645`

Top all feature movements:
- `lag_00__T_place_HELL`: contribution `-0.050681`
- `lag_06__T4__duck_amount`: contribution `+0.001982`
- `lag_08__CT3__is_walking`: contribution `+0.001972`
- `lag_00__damage_diff_last_5s`: contribution `+0.001800`
- `lag_07__CT3__is_walking`: contribution `+0.001781`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `104174`, seconds `65.00`, LSTM delta `-0.0623`

Top all feature movements:
- `lag_10__T5__duck_amount`: contribution `-0.003687`
- `lag_07__T5__duck_amount`: contribution `-0.003231`
- `lag_12__CT5__duck_amount`: contribution `-0.003050`
- `lag_02__CT4__alive`: contribution `-0.002180`
- `lag_03__T3__alive`: contribution `-0.002120`

Top utility-only movements:
- No utility movement among the top local contributors.
