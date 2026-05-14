# Local Round Explainability

- csv_path: `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3\the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `3263`, seconds `31.50`, LSTM `0.7536`, delta `+0.2040`
- tick `2975`, seconds `27.00`, LSTM `0.4565`, delta `+0.1257`
- tick `6047`, seconds `75.00`, LSTM `0.8810`, delta `-0.0737`
- tick `6655`, seconds `84.50`, LSTM `0.9516`, delta `+0.0720`
- tick `2527`, seconds `20.00`, LSTM `0.4460`, delta `-0.0640`
- tick `2559`, seconds `20.50`, LSTM `0.3822`, delta `-0.0638`
- tick `2719`, seconds `23.00`, LSTM `0.4143`, delta `+0.0623`
- tick `3391`, seconds `33.50`, LSTM `0.9414`, delta `+0.0602`
- tick `3359`, seconds `33.00`, LSTM `0.8812`, delta `+0.0529`
- tick `3295`, seconds `32.00`, LSTM `0.7974`, delta `+0.0438`

## Top 15 local ridge features

- `lag_09__T_place_SILO`: coefficient `-0.002237`, |coef| `0.002237`
- `lag_00__CT_place_TROPHY`: coefficient `0.002196`, |coef| `0.002196`
- `lag_13__CT_place_SQUEAKY`: coefficient `0.002074`, |coef| `0.002074`
- `lag_00__kill_diff_last_3s`: coefficient `0.001652`, |coef| `0.001652`
- `lag_03__T4__flash_duration`: coefficient `0.001628`, |coef| `0.001628`
- `lag_12__T_place_SILO`: coefficient `-0.001600`, |coef| `0.001600`
- `lag_10__CT_place_LOBBY`: coefficient `0.001504`, |coef| `0.001504`
- `lag_11__T_place_SILO`: coefficient `-0.001502`, |coef| `0.001502`
- `lag_00__T_place_SILO`: coefficient `-0.001396`, |coef| `0.001396`
- `lag_13__T_place_ROOF`: coefficient `0.001340`, |coef| `0.001340`
- `lag_14__CT_place_SQUEAKY`: coefficient `0.001225`, |coef| `0.001225`
- `lag_13__T_place_SILO`: coefficient `-0.001217`, |coef| `0.001217`
- `lag_03__CT2__flash_duration`: coefficient `0.001214`, |coef| `0.001214`
- `lag_10__T_place_SILO`: coefficient `-0.001178`, |coef| `0.001178`
- `lag_10__CT_place_HUT`: coefficient `-0.001147`, |coef| `0.001147`

## Top 10 utility ridge features

- `lag_03__T4__flash_duration`: coefficient `0.001628` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `0.001214` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000757` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000713` (lowers CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000539` (raises CT win probability)
- `lag_06__T1__flash`: coefficient `-0.000535` (lowers CT win probability)
- `lag_15__T1__smoke`: coefficient `-0.000511` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.000491` (lowers CT win probability)
- `lag_15__T4__flash_duration`: coefficient `-0.000432` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000400` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_SILO`: coefficient `-0.002237` (lowers CT win probability)
- `lag_00__CT_place_TROPHY`: coefficient `0.002196` (raises CT win probability)
- `lag_13__CT_place_SQUEAKY`: coefficient `0.002074` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001652` (raises CT win probability)
- `lag_12__T_place_SILO`: coefficient `-0.001600` (lowers CT win probability)
- `lag_10__CT_place_LOBBY`: coefficient `0.001504` (raises CT win probability)
- `lag_11__T_place_SILO`: coefficient `-0.001502` (lowers CT win probability)
- `lag_00__T_place_SILO`: coefficient `-0.001396` (lowers CT win probability)
- `lag_13__T_place_ROOF`: coefficient `0.001340` (raises CT win probability)
- `lag_14__CT_place_SQUEAKY`: coefficient `0.001225` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `3263`, seconds `31.50`, LSTM delta `+0.2040`

Top all feature movements:
- `lag_09__T_place_SILO`: contribution `+0.015199`
- `lag_03__T4__flash_duration`: contribution `+0.012849`
- `lag_10__CT_place_LOBBY`: contribution `+0.012311`
- `lag_10__CT_place_HUT`: contribution `+0.011191`
- `lag_00__CT_place_VENDING`: contribution `+0.009763`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `+0.012849`
- `lag_03__CT2__flash_duration`: contribution `+0.006520`
- `lag_03__T_flash_duration_sum`: contribution `+0.002841`

### tick `2975`, seconds `27.00`, LSTM delta `+0.1257`

Top all feature movements:
- `lag_13__CT_place_SQUEAKY`: contribution `+0.027577`
- `lag_00__T_place_SILO`: contribution `+0.009486`
- `lag_01__CT_place_HUT`: contribution `+0.008049`
- `lag_05__CT_place_ADMIN`: contribution `+0.005721`
- `lag_00__CT_place_MINI`: contribution `+0.004733`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6047`, seconds `75.00`, LSTM delta `-0.0737`

Top all feature movements:
- `lag_00__CT_place_TROPHY`: contribution `-0.032428`
- `lag_00__CT_place_VENDING`: contribution `+0.009763`
- `lag_04__T_bomb_zone_count`: contribution `-0.005596`
- `lag_00__kill_diff_last_3s`: contribution `-0.003976`
- `lag_07__T_place_HUT`: contribution `-0.003472`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6655`, seconds `84.50`, LSTM delta `+0.0720`

Top all feature movements:
- `lag_15__CT_place_VENDING`: contribution `+0.013536`
- `lag_04__T_bomb_zone_count`: contribution `+0.005596`
- `lag_15__CT_place_LOBBY`: contribution `+0.005023`
- `lag_12__CT_place_VENTS`: contribution `+0.004396`
- `lag_00__T_flash_alpha_mean`: contribution `+0.004325`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.004325`

### tick `2527`, seconds `20.00`, LSTM delta `-0.0640`

Top all feature movements:
- `lag_11__T_place_SILO`: contribution `-0.010204`
- `lag_11__CT_place_GARAGE`: contribution `-0.007179`
- `lag_04__CT_place_HUTROOF`: contribution `-0.004138`
- `lag_00__kill_diff_last_3s`: contribution `-0.003976`
- `lag_00__T_kills_last_3s`: contribution `-0.003400`

Top utility-only movements:
- No utility movement among the top local contributors.
