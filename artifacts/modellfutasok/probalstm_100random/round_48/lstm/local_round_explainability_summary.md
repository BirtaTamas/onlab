# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `40315`, seconds `105.50`, LSTM `0.0982`, delta `-0.3005`
- tick `39227`, seconds `88.50`, LSTM `0.5950`, delta `+0.2750`
- tick `40251`, seconds `104.50`, LSTM `0.4509`, delta `-0.2606`
- tick `39419`, seconds `91.50`, LSTM `0.8313`, delta `+0.2286`
- tick `40091`, seconds `102.00`, LSTM `0.8137`, delta `-0.0950`
- tick `35803`, seconds `35.00`, LSTM `0.4636`, delta `+0.0792`
- tick `38651`, seconds `79.50`, LSTM `0.3999`, delta `+0.0726`
- tick `39867`, seconds `98.50`, LSTM `0.9576`, delta `+0.0722`
- tick `34075`, seconds `8.00`, LSTM `0.3088`, delta `-0.0679`
- tick `35611`, seconds `32.00`, LSTM `0.4637`, delta `-0.0632`

## Top 15 local ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.003612`, |coef| `0.003612`
- `lag_00__kill_diff_last_3s`: coefficient `0.002294`, |coef| `0.002294`
- `lag_07__CT_place_ARAMP`: coefficient `0.002191`, |coef| `0.002191`
- `lag_10__T_place_MIDDOORS`: coefficient `-0.001943`, |coef| `0.001943`
- `lag_08__CT_place_ARAMP`: coefficient `0.001917`, |coef| `0.001917`
- `lag_00__T_kills_last_3s`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_13__CT_place_ARAMP`: coefficient `0.001819`, |coef| `0.001819`
- `lag_09__T_place_BDOORS`: coefficient `0.001808`, |coef| `0.001808`
- `lag_06__kill_diff_last_3s`: coefficient `0.001769`, |coef| `0.001769`
- `lag_11__T_place_MIDDOORS`: coefficient `-0.001765`, |coef| `0.001765`
- `lag_00__damage_diff_last_5s`: coefficient `0.001757`, |coef| `0.001757`
- `lag_10__T_place_BDOORS`: coefficient `0.001730`, |coef| `0.001730`
- `lag_12__T_place_BDOORS`: coefficient `0.001663`, |coef| `0.001663`
- `lag_01__damage_diff_last_5s`: coefficient `0.001647`, |coef| `0.001647`
- `lag_07__T_kills_last_3s`: coefficient `-0.001618`, |coef| `0.001618`

## Top 10 utility ridge features

- `lag_13__CT4__flash_duration`: coefficient `-0.001429` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001233` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.001068` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `0.001052` (raises CT win probability)
- `lag_13__CT2__flash_duration`: coefficient `0.001043` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.000965` (raises CT win probability)
- `lag_07__CT2__flash_duration`: coefficient `0.000954` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000939` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `-0.000931` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000914` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BDOORS`: coefficient `-0.003612` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002294` (raises CT win probability)
- `lag_07__CT_place_ARAMP`: coefficient `0.002191` (raises CT win probability)
- `lag_10__T_place_MIDDOORS`: coefficient `-0.001943` (lowers CT win probability)
- `lag_08__CT_place_ARAMP`: coefficient `0.001917` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001878` (lowers CT win probability)
- `lag_13__CT_place_ARAMP`: coefficient `0.001819` (raises CT win probability)
- `lag_09__T_place_BDOORS`: coefficient `0.001808` (raises CT win probability)
- `lag_06__kill_diff_last_3s`: coefficient `0.001769` (raises CT win probability)
- `lag_11__T_place_MIDDOORS`: coefficient `-0.001765` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40315`, seconds `105.50`, LSTM delta `-0.3005`

Top all feature movements:
- `lag_13__CT_place_ARAMP`: contribution `-0.011330`
- `lag_01__T_place_BDOORS`: contribution `-0.010936`
- `lag_15__CT_shots_fired_sum`: contribution `-0.007670`
- `lag_11__T_place_MIDDOORS`: contribution `-0.007500`
- `lag_00__T_kills_last_3s`: contribution `-0.005950`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39227`, seconds `88.50`, LSTM delta `+0.2750`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.045184`
- `lag_10__T_place_BDOORS`: contribution `+0.021643`
- `lag_06__T_place_BDOORS`: contribution `+0.010819`
- `lag_02__CT_place_ARAMP`: contribution `+0.010028`
- `lag_15__CT2__is_scoped`: contribution `+0.008449`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `+0.008074`
- `lag_07__CT_flash_duration_sum`: contribution `+0.005693`
- `lag_06__T5__flash_duration`: contribution `+0.005532`
- `lag_07__CT5__flash_duration`: contribution `+0.005262`
- `lag_07__CT2__flash_duration`: contribution `+0.004653`

### tick `40251`, seconds `104.50`, LSTM delta `-0.2606`

Top all feature movements:
- `lag_14__T_place_BDOORS`: contribution `-0.016856`
- `lag_07__CT_place_ARAMP`: contribution `-0.013648`
- `lag_11__CT3__shots_fired`: contribution `-0.010506`
- `lag_11__CT_shots_fired_sum`: contribution `-0.010050`
- `lag_11__CT_place_ARAMP`: contribution `-0.007919`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `39419`, seconds `91.50`, LSTM delta `+0.2286`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.045184`
- `lag_12__T_place_BDOORS`: contribution `+0.020801`
- `lag_08__CT_place_ARAMP`: contribution `+0.011940`
- `lag_06__T_place_BDOORS`: contribution `-0.010819`
- `lag_13__CT_flashed_players`: contribution `+0.005409`

Top utility-only movements:
- `lag_13__CT2__flash_duration`: contribution `+0.005085`
- `lag_12__T5__flash_duration`: contribution `+0.004427`
- `lag_13__CT5__flash_duration`: contribution `+0.004405`
- `lag_04__CT5__flash_duration`: contribution `+0.003590`

### tick `40091`, seconds `102.00`, LSTM delta `-0.0950`

Top all feature movements:
- `lag_09__T_place_BDOORS`: contribution `-0.022609`
- `lag_02__CT_place_ARAMP`: contribution `-0.010028`
- `lag_00__T_kills_last_3s`: contribution `-0.005950`
- `lag_00__kill_diff_last_3s`: contribution `-0.005522`
- `lag_05__T_place_LOWERTUNNEL`: contribution `-0.005055`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.004998`
- `lag_15__CT2__flash_duration`: contribution `-0.002523`
