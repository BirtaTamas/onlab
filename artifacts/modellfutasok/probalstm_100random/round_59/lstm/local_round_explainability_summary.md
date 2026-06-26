# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `13`

## Largest probability jumps

- tick `91008`, seconds `5.00`, LSTM `0.1667`, delta `-0.1104`
- tick `94176`, seconds `54.50`, LSTM `0.0389`, delta `-0.0984`
- tick `91360`, seconds `10.50`, LSTM `0.1922`, delta `+0.0522`
- tick `91040`, seconds `5.50`, LSTM `0.1274`, delta `-0.0393`
- tick `91328`, seconds `10.00`, LSTM `0.1399`, delta `+0.0391`
- tick `90848`, seconds `2.50`, LSTM `0.3597`, delta `-0.0349`
- tick `90976`, seconds `4.50`, LSTM `0.2771`, delta `-0.0332`
- tick `91680`, seconds `15.50`, LSTM `0.2239`, delta `+0.0306`
- tick `91392`, seconds `11.00`, LSTM `0.2211`, delta `+0.0289`
- tick `92960`, seconds `35.50`, LSTM `0.1728`, delta `+0.0276`

## Top 15 local ridge features

- `lag_05__T_flashes_last_5s`: coefficient `-0.001098`, |coef| `0.001098`
- `lag_04__T_place_MIDDOORS`: coefficient `-0.001066`, |coef| `0.001066`
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001007`, |coef| `0.001007`
- `lag_02__T_place_MIDDOORS`: coefficient `-0.000973`, |coef| `0.000973`
- `lag_03__T_place_MIDDOORS`: coefficient `-0.000923`, |coef| `0.000923`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000918`, |coef| `0.000918`
- `lag_00__T_kills_last_3s`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_00__T_damage_last_5s`: coefficient `-0.000814`, |coef| `0.000814`
- `lag_00__damage_diff_last_5s`: coefficient `0.000739`, |coef| `0.000739`
- `lag_05__T_place_MIDDOORS`: coefficient `-0.000698`, |coef| `0.000698`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000695`, |coef| `0.000695`
- `lag_00__kill_diff_last_3s`: coefficient `0.000691`, |coef| `0.000691`
- `lag_10__T_money_sum`: coefficient `-0.000685`, |coef| `0.000685`
- `lag_10__T_start_balance_sum`: coefficient `-0.000682`, |coef| `0.000682`
- `lag_06__T_flashes_last_5s`: coefficient `-0.000662`, |coef| `0.000662`

## Top 10 utility ridge features

- `lag_05__T_flashes_last_5s`: coefficient `-0.001098` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.000662` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000597` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000544` (raises CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000540` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000526` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000511` (raises CT win probability)
- `lag_04__T_flashes_last_5s`: coefficient `-0.000510` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000496` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000488` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_MIDDOORS`: coefficient `-0.001066` (lowers CT win probability)
- `lag_00__T_place_MIDDOORS`: coefficient `-0.001007` (lowers CT win probability)
- `lag_02__T_place_MIDDOORS`: coefficient `-0.000973` (lowers CT win probability)
- `lag_03__T_place_MIDDOORS`: coefficient `-0.000923` (lowers CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.000918` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000909` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000814` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000739` (raises CT win probability)
- `lag_05__T_place_MIDDOORS`: coefficient `-0.000698` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000695` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `91008`, seconds `5.00`, LSTM delta `-0.1104`

Top all feature movements:
- `lag_05__T_flashes_last_5s`: contribution `-0.009950`
- `lag_05__T3__is_scoped`: contribution `-0.003692`
- `lag_07__CT_place_UNDERA`: contribution `-0.003340`
- `lag_00__T_kills_last_3s`: contribution `-0.002881`
- `lag_10__T_money_sum`: contribution `-0.002715`

Top utility-only movements:
- `lag_05__T_flashes_last_5s`: contribution `-0.009950`
- `lag_00__CT5__utility_total`: contribution `-0.001382`
- `lag_00__CT5__molly`: contribution `-0.001059`

### tick `94176`, seconds `54.50`, LSTM delta `-0.0984`

Top all feature movements:
- `lag_04__T_place_MIDDOORS`: contribution `-0.004531`
- `lag_00__T_shots_fired_sum`: contribution `-0.004169`
- `lag_02__CT4__flash_duration`: contribution `-0.004030`
- `lag_09__CT_place_EXTENDEDA`: contribution `-0.003299`
- `lag_01__CT_place_EXTENDEDA`: contribution `-0.003142`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.004030`

### tick `91360`, seconds `10.50`, LSTM delta `+0.0522`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.005998`
- `lag_14__T_place_OUTSIDETUNNEL`: contribution `+0.002918`
- `lag_12__CT_place_UNDERA`: contribution `+0.002761`
- `lag_12__CT_place_LONGA`: contribution `+0.002229`
- `lag_06__CT_flashed_players`: contribution `+0.002028`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `+0.005998`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001435`
- `lag_06__CT4__flash_duration`: contribution `+0.000960`

### tick `91040`, seconds `5.50`, LSTM delta `-0.0393`

Top all feature movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.005998`
- `lag_02__CT_place_UNDERA`: contribution `+0.003128`
- `lag_08__CT_place_UNDERA`: contribution `-0.002294`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `-0.001937`
- `lag_11__T_money_sum`: contribution `-0.001756`

Top utility-only movements:
- `lag_06__T_flashes_last_5s`: contribution `-0.005998`

### tick `91328`, seconds `10.00`, LSTM delta `+0.0391`

Top all feature movements:
- `lag_05__T_flashes_last_5s`: contribution `+0.009950`
- `lag_00__T_damage_last_5s`: contribution `+0.001953`
- `lag_15__T_flashes_last_5s`: contribution `+0.001768`
- `lag_09__CT_place_UNDERA`: contribution `+0.001689`
- `lag_00__damage_diff_last_5s`: contribution `+0.001667`

Top utility-only movements:
- `lag_05__T_flashes_last_5s`: contribution `+0.009950`
- `lag_15__T_flashes_last_5s`: contribution `+0.001768`
- `lag_05__CT4__flash_duration`: contribution `-0.001461`
