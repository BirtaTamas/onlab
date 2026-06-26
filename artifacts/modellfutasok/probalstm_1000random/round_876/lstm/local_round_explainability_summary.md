# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `18`

## Largest probability jumps

- tick `137591`, seconds `44.50`, LSTM `0.6547`, delta `+0.3498`
- tick `138967`, seconds `66.00`, LSTM `0.6051`, delta `+0.3349`
- tick `138775`, seconds `63.00`, LSTM `0.3336`, delta `-0.1627`
- tick `137687`, seconds `46.00`, LSTM `0.4295`, delta `-0.1606`
- tick `135703`, seconds `15.00`, LSTM `0.1072`, delta `-0.1536`
- tick `137527`, seconds `43.50`, LSTM `0.2153`, delta `+0.1421`
- tick `138551`, seconds `59.50`, LSTM `0.5206`, delta `+0.1292`
- tick `137655`, seconds `45.50`, LSTM `0.5900`, delta `-0.1043`
- tick `137559`, seconds `44.00`, LSTM `0.3049`, delta `+0.0897`
- tick `137975`, seconds `50.50`, LSTM `0.3611`, delta `+0.0706`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006975`, |coef| `0.006975`
- `lag_00__T_kills_last_3s`: coefficient `-0.004405`, |coef| `0.004405`
- `lag_00__CT_kills_last_3s`: coefficient `0.004352`, |coef| `0.004352`
- `lag_13__T_shots_fired_sum`: coefficient `-0.004255`, |coef| `0.004255`
- `lag_00__damage_diff_last_5s`: coefficient `0.003602`, |coef| `0.003602`
- `lag_00__CT_damage_last_5s`: coefficient `0.003185`, |coef| `0.003185`
- `lag_14__T3__shots_fired`: coefficient `0.002609`, |coef| `0.002609`
- `lag_04__T_place_OUTSIDETUNNEL`: coefficient `0.002497`, |coef| `0.002497`
- `lag_06__CT_place_CATWALK`: coefficient `0.002377`, |coef| `0.002377`
- `lag_05__CT2__is_walking`: coefficient `-0.002269`, |coef| `0.002269`
- `lag_15__T3__shots_fired`: coefficient `0.002249`, |coef| `0.002249`
- `lag_04__T_shots_fired_sum`: coefficient `0.002221`, |coef| `0.002221`
- `lag_15__T_shots_fired_sum`: coefficient `0.002216`, |coef| `0.002216`
- `lag_14__T2__duck_amount`: coefficient `-0.002207`, |coef| `0.002207`
- `lag_06__CT_place_SHORTSTAIRS`: coefficient `-0.002129`, |coef| `0.002129`

## Top 10 utility ridge features

- `lag_12__CT3__flash_duration`: coefficient `-0.001628` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.001626` (lowers CT win probability)
- `lag_13__T3__utility_total`: coefficient `-0.001568` (lowers CT win probability)
- `lag_12__T5__molly`: coefficient `0.001474` (raises CT win probability)
- `lag_13__T3__molly`: coefficient `-0.001419` (lowers CT win probability)
- `lag_13__T3__smoke`: coefficient `-0.001391` (lowers CT win probability)
- `lag_15__active_infernos_total`: coefficient `-0.001228` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `-0.001212` (lowers CT win probability)
- `lag_02__T2__utility_total`: coefficient `-0.001162` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.001107` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006975` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004405` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004352` (raises CT win probability)
- `lag_13__T_shots_fired_sum`: coefficient `-0.004255` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003602` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003185` (raises CT win probability)
- `lag_14__T3__shots_fired`: coefficient `0.002609` (raises CT win probability)
- `lag_04__T_place_OUTSIDETUNNEL`: coefficient `0.002497` (raises CT win probability)
- `lag_06__CT_place_CATWALK`: coefficient `0.002377` (raises CT win probability)
- `lag_05__CT2__is_walking`: coefficient `-0.002269` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `137591`, seconds `44.50`, LSTM delta `+0.3498`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016789`
- `lag_00__CT_kills_last_3s`: contribution `+0.012566`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `+0.012482`
- `lag_06__CT_place_SHORTSTAIRS`: contribution `+0.011870`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `+0.009843`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `+0.008158`

### tick `138967`, seconds `66.00`, LSTM delta `+0.3349`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `+0.035092`
- `lag_00__kill_diff_last_3s`: contribution `+0.033577`
- `lag_00__T_kills_last_3s`: contribution `+0.013955`
- `lag_00__CT_kills_last_3s`: contribution `+0.012566`
- `lag_15__T_shots_fired_sum`: contribution `+0.008308`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `138775`, seconds `63.00`, LSTM delta `-0.1627`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016789`
- `lag_07__T_shots_fired_sum`: contribution `-0.015819`
- `lag_00__T_kills_last_3s`: contribution `-0.013955`
- `lag_04__T_shots_fired_sum`: contribution `-0.008326`
- `lag_01__CT_kills_last_3s`: contribution `-0.004545`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `137687`, seconds `46.00`, LSTM delta `-0.1606`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016789`
- `lag_00__T_kills_last_3s`: contribution `-0.013955`
- `lag_00__CT_place_CATWALK`: contribution `-0.007205`
- `lag_00__T_shots_fired_sum`: contribution `+0.006241`
- `lag_04__T_place_LOWERTUNNEL`: contribution `-0.004981`

Top utility-only movements:
- `lag_15__CT3__flash_duration`: contribution `-0.004482`

### tick `135703`, seconds `15.00`, LSTM delta `-0.1536`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016789`
- `lag_00__T_kills_last_3s`: contribution `-0.013955`
- `lag_06__CT_place_SHORTSTAIRS`: contribution `-0.011870`
- `lag_06__CT_place_CATWALK`: contribution `-0.009467`
- `lag_00__damage_diff_last_5s`: contribution `-0.008126`

Top utility-only movements:
- No utility movement among the top local contributors.
