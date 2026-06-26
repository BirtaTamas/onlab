# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `15`

## Largest probability jumps

- tick `110832`, seconds `83.50`, LSTM `0.2620`, delta `-0.2465`
- tick `110704`, seconds `81.50`, LSTM `0.5111`, delta `+0.2448`
- tick `110672`, seconds `81.00`, LSTM `0.2663`, delta `-0.2161`
- tick `108528`, seconds `47.50`, LSTM `0.4308`, delta `+0.1879`
- tick `111440`, seconds `93.00`, LSTM `0.0381`, delta `-0.0992`
- tick `109744`, seconds `66.50`, LSTM `0.4405`, delta `+0.0647`
- tick `110768`, seconds `82.50`, LSTM `0.5561`, delta `+0.0479`
- tick `110800`, seconds `83.00`, LSTM `0.5085`, delta `-0.0477`
- tick `108848`, seconds `52.50`, LSTM `0.3503`, delta `-0.0469`
- tick `111312`, seconds `91.00`, LSTM `0.1556`, delta `+0.0437`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003063`, |coef| `0.003063`
- `lag_14__CT_place_SHOP`: coefficient `-0.002484`, |coef| `0.002484`
- `lag_02__T_flashed_players`: coefficient `-0.002394`, |coef| `0.002394`
- `lag_00__CT_kills_last_3s`: coefficient `0.002232`, |coef| `0.002232`
- `lag_10__T_place_PALACEINTERIOR`: coefficient `0.002177`, |coef| `0.002177`
- `lag_06__CT_place_JUNGLE`: coefficient `0.002075`, |coef| `0.002075`
- `lag_02__T_place_PALACEINTERIOR`: coefficient `-0.001933`, |coef| `0.001933`
- `lag_02__CT_flashed_players`: coefficient `-0.001917`, |coef| `0.001917`
- `lag_06__CT2__duck_amount`: coefficient `0.001816`, |coef| `0.001816`
- `lag_12__T_place_TRAMP`: coefficient `0.001796`, |coef| `0.001796`
- `lag_02__CT5__flash_duration`: coefficient `-0.001689`, |coef| `0.001689`
- `lag_10__T1__duck_amount`: coefficient `0.001688`, |coef| `0.001688`
- `lag_05__CT2__duck_amount`: coefficient `0.001657`, |coef| `0.001657`
- `lag_00__T_kills_last_3s`: coefficient `-0.001582`, |coef| `0.001582`
- `lag_09__T_place_PALACEINTERIOR`: coefficient `-0.001533`, |coef| `0.001533`

## Top 10 utility ridge features

- `lag_02__CT5__flash_duration`: coefficient `-0.001689` (lowers CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `-0.001223` (lowers CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000994` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000947` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.000941` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000939` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000909` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000881` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.000854` (lowers CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `-0.000814` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003063` (raises CT win probability)
- `lag_14__CT_place_SHOP`: coefficient `-0.002484` (lowers CT win probability)
- `lag_02__T_flashed_players`: coefficient `-0.002394` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002232` (raises CT win probability)
- `lag_10__T_place_PALACEINTERIOR`: coefficient `0.002177` (raises CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `0.002075` (raises CT win probability)
- `lag_02__T_place_PALACEINTERIOR`: coefficient `-0.001933` (lowers CT win probability)
- `lag_02__CT_flashed_players`: coefficient `-0.001917` (lowers CT win probability)
- `lag_06__CT2__duck_amount`: coefficient `0.001816` (raises CT win probability)
- `lag_12__T_place_TRAMP`: coefficient `0.001796` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `110832`, seconds `83.50`, LSTM delta `-0.2465`

Top all feature movements:
- `lag_02__CT5__flash_duration`: contribution `-0.009819`
- `lag_02__T_flashed_players`: contribution `-0.009239`
- `lag_02__CT_flashed_players`: contribution `-0.008395`
- `lag_00__kill_diff_last_3s`: contribution `-0.007372`
- `lag_10__T_place_PALACEINTERIOR`: contribution `-0.007302`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `-0.009819`
- `lag_02__CT_flash_duration_sum`: contribution `-0.004089`
- `lag_11__T_A_site_active_infernos`: contribution `-0.002819`

### tick `110704`, seconds `81.50`, LSTM delta `+0.2448`

Top all feature movements:
- `lag_14__CT_place_SHOP`: contribution `+0.012461`
- `lag_00__kill_diff_last_3s`: contribution `+0.007372`
- `lag_10__T_place_PALACEINTERIOR`: contribution `+0.007302`
- `lag_02__T_place_PALACEINTERIOR`: contribution `+0.006484`
- `lag_00__CT_kills_last_3s`: contribution `+0.006445`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110672`, seconds `81.00`, LSTM delta `-0.2161`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.007372`
- `lag_00__CT2__duck_amount`: contribution `-0.005699`
- `lag_09__T_place_PALACEINTERIOR`: contribution `-0.005143`
- `lag_00__T_kills_last_3s`: contribution `-0.005011`
- `lag_02__T_flashed_players`: contribution `-0.004619`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `-0.002802`

### tick `108528`, seconds `47.50`, LSTM delta `+0.1879`

Top all feature movements:
- `lag_06__CT_place_JUNGLE`: contribution `+0.013314`
- `lag_14__CT_place_SHOP`: contribution `+0.012461`
- `lag_00__kill_diff_last_3s`: contribution `+0.007372`
- `lag_00__CT_kills_last_3s`: contribution `+0.006445`
- `lag_15__T_place_PALACEINTERIOR`: contribution `+0.004247`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111440`, seconds `93.00`, LSTM delta `-0.0992`

Top all feature movements:
- `lag_04__T_place_SCAFFOLDING`: contribution `-0.023810`
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.015510`
- `lag_00__kill_diff_last_3s`: contribution `-0.007372`
- `lag_00__T_kills_last_3s`: contribution `-0.005011`
- `lag_04__T_place_PALACEINTERIOR`: contribution `-0.003726`

Top utility-only movements:
- No utility movement among the top local contributors.
