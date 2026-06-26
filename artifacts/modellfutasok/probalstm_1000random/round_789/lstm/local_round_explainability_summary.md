# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `9`

## Largest probability jumps

- tick `48257`, seconds `15.50`, LSTM `0.6930`, delta `+0.1744`
- tick `49313`, seconds `32.00`, LSTM `0.8502`, delta `+0.1605`
- tick `50721`, seconds `54.00`, LSTM `0.9575`, delta `+0.0397`
- tick `49217`, seconds `30.50`, LSTM `0.6661`, delta `-0.0348`
- tick `48897`, seconds `25.50`, LSTM `0.6830`, delta `-0.0321`
- tick `50625`, seconds `52.50`, LSTM `0.8836`, delta `+0.0309`
- tick `48385`, seconds `17.50`, LSTM `0.7290`, delta `+0.0306`
- tick `48833`, seconds `24.50`, LSTM `0.7368`, delta `-0.0303`
- tick `48321`, seconds `16.50`, LSTM `0.7203`, delta `+0.0272`
- tick `47969`, seconds `11.00`, LSTM `0.5387`, delta `-0.0262`

## Top 15 local ridge features

- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001555`, |coef| `0.001555`
- `lag_00__CT_kills_last_3s`: coefficient `0.001130`, |coef| `0.001130`
- `lag_02__CT_place_TRUCK`: coefficient `0.001100`, |coef| `0.001100`
- `lag_07__CT_place_TRUCK`: coefficient `0.001017`, |coef| `0.001017`
- `lag_04__CT_place_TRUCK`: coefficient `0.000993`, |coef| `0.000993`
- `lag_04__CT_shots_fired_sum`: coefficient `-0.000976`, |coef| `0.000976`
- `lag_03__CT_place_SNIPERSNEST`: coefficient `-0.000961`, |coef| `0.000961`
- `lag_00__kill_diff_last_3s`: coefficient `0.000942`, |coef| `0.000942`
- `lag_00__CT_place_CATWALK`: coefficient `-0.000928`, |coef| `0.000928`
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.000915`, |coef| `0.000915`
- `lag_01__CT_place_TRUCK`: coefficient `0.000914`, |coef| `0.000914`
- `lag_04__CT4__shots_fired`: coefficient `-0.000820`, |coef| `0.000820`
- `lag_07__T5__is_scoped`: coefficient `0.000806`, |coef| `0.000806`
- `lag_07__T5__duck_amount`: coefficient `0.000802`, |coef| `0.000802`
- `lag_03__CT_place_TRUCK`: coefficient `0.000782`, |coef| `0.000782`

## Top 10 utility ridge features

- `lag_10__CT_A_site_active_infernos`: coefficient `0.000534` (raises CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.000506` (lowers CT win probability)
- `lag_10__active_infernos_total`: coefficient `0.000499` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000494` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000481` (lowers CT win probability)
- `lag_00__T3__utility_total`: coefficient `-0.000476` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000473` (raises CT win probability)
- `lag_13__CT1__molly`: coefficient `-0.000455` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `-0.000437` (lowers CT win probability)
- `lag_11__T4__smoke`: coefficient `-0.000431` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_SNIPERSNEST`: coefficient `0.001555` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001130` (raises CT win probability)
- `lag_02__CT_place_TRUCK`: coefficient `0.001100` (raises CT win probability)
- `lag_07__CT_place_TRUCK`: coefficient `0.001017` (raises CT win probability)
- `lag_04__CT_place_TRUCK`: coefficient `0.000993` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `-0.000976` (lowers CT win probability)
- `lag_03__CT_place_SNIPERSNEST`: coefficient `-0.000961` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000942` (raises CT win probability)
- `lag_00__CT_place_CATWALK`: coefficient `-0.000928` (lowers CT win probability)
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.000915` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `48257`, seconds `15.50`, LSTM delta `+0.1744`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.012879`
- `lag_04__CT4__shots_fired`: contribution `+0.008397`
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.008326`
- `lag_02__CT_place_TRUCK`: contribution `-0.007097`
- `lag_04__CT_place_TRUCK`: contribution `+0.006404`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.002959`
- `lag_09__T5__flash_duration`: contribution `+0.001819`
- `lag_04__CT4__flash_duration`: contribution `+0.001578`

### tick `49313`, seconds `32.00`, LSTM delta `+0.1605`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `+0.008326`
- `lag_02__CT_place_TRUCK`: contribution `+0.007097`
- `lag_07__CT_place_TRUCK`: contribution `+0.006563`
- `lag_14__CT_place_UNDERPASS`: contribution `+0.005306`
- `lag_03__CT_place_SNIPERSNEST`: contribution `+0.005148`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50721`, seconds `54.00`, LSTM delta `+0.0397`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003264`
- `lag_06__T_place_CONNECTOR`: contribution `+0.002486`
- `lag_00__kill_diff_last_3s`: contribution `+0.002268`
- `lag_03__T1__flash_duration`: contribution `+0.002162`
- `lag_05__CT_place_TRUCK`: contribution `+0.002049`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.002162`
- `lag_00__T1__flash`: contribution `+0.000600`

### tick `49217`, seconds `30.50`, LSTM delta `-0.0348`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.008326`
- `lag_02__CT_place_TRUCK`: contribution `-0.007097`
- `lag_04__CT_place_TRUCK`: contribution `+0.006404`
- `lag_03__CT_place_SNIPERSNEST`: contribution `-0.005148`
- `lag_00__CT_place_CATWALK`: contribution `-0.003695`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48897`, seconds `25.50`, LSTM delta `-0.0321`

Top all feature movements:
- `lag_14__CT_place_TRUCK`: contribution `-0.002975`
- `lag_06__CT4__duck_amount`: contribution `-0.002583`
- `lag_06__CT_place_CATWALK`: contribution `+0.002223`
- `lag_13__T5__flash_duration`: contribution `-0.002015`
- `lag_06__CT_place_UNDERPASS`: contribution `-0.001740`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `-0.002015`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001448`
