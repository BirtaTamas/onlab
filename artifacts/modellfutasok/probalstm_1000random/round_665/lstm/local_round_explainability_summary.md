# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `107038`, seconds `63.00`, LSTM `0.1222`, delta `-0.2490`
- tick `106526`, seconds `55.00`, LSTM `0.4573`, delta `+0.0764`
- tick `104350`, seconds `21.00`, LSTM `0.4308`, delta `+0.0543`
- tick `106494`, seconds `54.50`, LSTM `0.3809`, delta `-0.0531`
- tick `107582`, seconds `71.50`, LSTM `0.0348`, delta `-0.0442`
- tick `106910`, seconds `61.00`, LSTM `0.4340`, delta `-0.0422`
- tick `105854`, seconds `44.50`, LSTM `0.4244`, delta `-0.0399`
- tick `106558`, seconds `55.50`, LSTM `0.4959`, delta `+0.0387`
- tick `106942`, seconds `61.50`, LSTM `0.3969`, delta `-0.0371`
- tick `107006`, seconds `62.50`, LSTM `0.3711`, delta `-0.0364`

## Top 15 local ridge features

- `lag_15__T_bomb_zone_count`: coefficient `0.002070`, |coef| `0.002070`
- `lag_01__T1__is_scoped`: coefficient `-0.001635`, |coef| `0.001635`
- `lag_10__T3__duck_amount`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_12__T_A_site_active_infernos`: coefficient `0.001524`, |coef| `0.001524`
- `lag_10__T1__is_scoped`: coefficient `-0.001503`, |coef| `0.001503`
- `lag_00__T_kills_last_3s`: coefficient `-0.001402`, |coef| `0.001402`
- `lag_11__CT5__is_scoped`: coefficient `0.001397`, |coef| `0.001397`
- `lag_15__T1__has_bomb`: coefficient `0.001355`, |coef| `0.001355`
- `lag_14__T_flashed_players`: coefficient `0.001316`, |coef| `0.001316`
- `lag_15__bomb_planted`: coefficient `-0.001306`, |coef| `0.001306`
- `lag_12__T1__duck_amount`: coefficient `0.001291`, |coef| `0.001291`
- `lag_14__T2__flash_duration`: coefficient `0.001275`, |coef| `0.001275`
- `lag_14__T5__duck_amount`: coefficient `0.001275`, |coef| `0.001275`
- `lag_04__CT5__duck_amount`: coefficient `-0.001268`, |coef| `0.001268`
- `lag_00__CT1__alive`: coefficient `0.001245`, |coef| `0.001245`

## Top 10 utility ridge features

- `lag_12__T_A_site_active_infernos`: coefficient `0.001524` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001275` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001127` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `0.001119` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001104` (raises CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.001090` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.001082` (raises CT win probability)
- `lag_09__CT5__molly`: coefficient `0.001073` (raises CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.001043` (raises CT win probability)
- `lag_04__T2__molly`: coefficient `0.001004` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_bomb_zone_count`: coefficient `0.002070` (raises CT win probability)
- `lag_01__T1__is_scoped`: coefficient `-0.001635` (lowers CT win probability)
- `lag_10__T3__duck_amount`: coefficient `-0.001568` (lowers CT win probability)
- `lag_10__T1__is_scoped`: coefficient `-0.001503` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001402` (lowers CT win probability)
- `lag_11__CT5__is_scoped`: coefficient `0.001397` (raises CT win probability)
- `lag_15__T1__has_bomb`: coefficient `0.001355` (raises CT win probability)
- `lag_14__T_flashed_players`: coefficient `0.001316` (raises CT win probability)
- `lag_15__bomb_planted`: coefficient `-0.001306` (lowers CT win probability)
- `lag_12__T1__duck_amount`: coefficient `0.001291` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `107038`, seconds `63.00`, LSTM delta `-0.2490`

Top all feature movements:
- `lag_15__T_bomb_zone_count`: contribution `-0.012049`
- `lag_01__T1__is_scoped`: contribution `-0.009342`
- `lag_10__T1__is_scoped`: contribution `-0.008588`
- `lag_10__T3__duck_amount`: contribution `-0.005911`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.005076`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.004537`

### tick `106526`, seconds `55.00`, LSTM delta `+0.0764`

Top all feature movements:
- `lag_14__T_flashed_players`: contribution `+0.007616`
- `lag_00__CT_place_LONGDOORS`: contribution `-0.005076`
- `lag_11__CT5__is_scoped`: contribution `+0.004996`
- `lag_01__CT_place_LONGDOORS`: contribution `+0.004735`
- `lag_06__T_bomb_zone_count`: contribution `+0.003649`

Top utility-only movements:
- `lag_14__T2__flash_duration`: contribution `+0.002902`
- `lag_14__T_flash_duration_sum`: contribution `+0.002548`
- `lag_11__T2__flash_duration`: contribution `+0.002277`
- `lag_11__T_A_site_active_infernos`: contribution `+0.002222`
- `lag_02__T1__flash_duration`: contribution `+0.002163`

### tick `104350`, seconds `21.00`, LSTM delta `+0.0543`

Top all feature movements:
- `lag_01__T1__is_scoped`: contribution `+0.009342`
- `lag_10__T3__duck_amount`: contribution `+0.005911`
- `lag_04__T1__is_scoped`: contribution `-0.004373`
- `lag_10__CT3__flash_duration`: contribution `+0.004183`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `+0.003687`

Top utility-only movements:
- `lag_10__CT3__flash_duration`: contribution `+0.004183`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.002138`

### tick `106494`, seconds `54.50`, LSTM delta `-0.0531`

Top all feature movements:
- `lag_00__CT_place_LONGDOORS`: contribution `+0.005076`
- `lag_05__T_place_EXTENDEDA`: contribution `-0.003419`
- `lag_02__T4__duck_amount`: contribution `-0.002936`
- `lag_00__bomb_events_last_5s`: contribution `-0.002760`
- `lag_10__T_place_EXTENDEDA`: contribution `-0.002265`

Top utility-only movements:
- `lag_13__T2__flash_duration`: contribution `+0.001838`
- `lag_13__T_flash_duration_sum`: contribution `+0.001477`

### tick `107582`, seconds `71.50`, LSTM delta `-0.0442`

Top all feature movements:
- `lag_01__T1__is_scoped`: contribution `+0.009342`
- `lag_13__T1__is_scoped`: contribution `-0.005264`
- `lag_12__T_A_site_active_infernos`: contribution `-0.004537`
- `lag_00__T_kills_last_3s`: contribution `-0.004441`
- `lag_05__T1__is_scoped`: contribution `-0.004010`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.004537`
- `lag_12__T_active_infernos`: contribution `-0.002254`
