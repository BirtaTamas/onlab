# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `99235`, seconds `15.00`, LSTM `0.1643`, delta `-0.3207`
- tick `101251`, seconds `46.50`, LSTM `0.2978`, delta `-0.3036`
- tick `103171`, seconds `76.50`, LSTM `0.1013`, delta `-0.2236`
- tick `101059`, seconds `43.50`, LSTM `0.4848`, delta `+0.2031`
- tick `103491`, seconds `81.50`, LSTM `0.2579`, delta `+0.1865`
- tick `100323`, seconds `32.00`, LSTM `0.3258`, delta `+0.1735`
- tick `99843`, seconds `24.50`, LSTM `0.0550`, delta `-0.1372`
- tick `101987`, seconds `58.00`, LSTM `0.3773`, delta `+0.1181`
- tick `99555`, seconds `20.00`, LSTM `0.2317`, delta `+0.1129`
- tick `99715`, seconds `22.50`, LSTM `0.2041`, delta `-0.0796`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007108`, |coef| `0.007108`
- `lag_00__T_kills_last_3s`: coefficient `-0.004546`, |coef| `0.004546`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004395`, |coef| `0.004395`
- `lag_00__CT_kills_last_3s`: coefficient `0.004383`, |coef| `0.004383`
- `lag_00__damage_diff_last_5s`: coefficient `0.003715`, |coef| `0.003715`
- `lag_05__CT_place_HOUSE`: coefficient `0.003708`, |coef| `0.003708`
- `lag_05__CT_place_ALLEY`: coefficient `-0.003417`, |coef| `0.003417`
- `lag_03__T_place_RAMP`: coefficient `-0.003320`, |coef| `0.003320`
- `lag_04__T_flashed_players`: coefficient `-0.003075`, |coef| `0.003075`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.003061`, |coef| `0.003061`
- `lag_06__CT_B_site_active_infernos`: coefficient `0.003042`, |coef| `0.003042`
- `lag_01__kill_diff_last_3s`: coefficient `0.002855`, |coef| `0.002855`
- `lag_00__T_bomb_zone_count`: coefficient `-0.002809`, |coef| `0.002809`
- `lag_00__T_damage_last_5s`: coefficient `-0.002701`, |coef| `0.002701`
- `lag_00__bomb_events_last_5s`: coefficient `0.002645`, |coef| `0.002645`

## Top 10 utility ridge features

- `lag_06__CT_B_site_active_infernos`: coefficient `0.003042` (raises CT win probability)
- `lag_06__CT_active_infernos`: coefficient `0.002166` (raises CT win probability)
- `lag_10__CT2__molly`: coefficient `-0.001883` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.001786` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `-0.001722` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.001679` (raises CT win probability)
- `lag_11__T4__smoke`: coefficient `-0.001668` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001638` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.001629` (raises CT win probability)
- `lag_06__active_infernos_total`: coefficient `0.001492` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007108` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004546` (lowers CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.004395` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004383` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003715` (raises CT win probability)
- `lag_05__CT_place_HOUSE`: coefficient `0.003708` (raises CT win probability)
- `lag_05__CT_place_ALLEY`: coefficient `-0.003417` (lowers CT win probability)
- `lag_03__T_place_RAMP`: coefficient `-0.003320` (lowers CT win probability)
- `lag_04__T_flashed_players`: coefficient `-0.003075` (lowers CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.003061` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `99235`, seconds `15.00`, LSTM delta `-0.3207`

Top all feature movements:
- `lag_04__T_flashed_players`: contribution `-0.023733`
- `lag_00__kill_diff_last_3s`: contribution `-0.017108`
- `lag_00__T_kills_last_3s`: contribution `-0.014402`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.012323`
- `lag_01__T_place_SIDEENTRANCE`: contribution `-0.012078`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `-0.009898`
- `lag_03__CT5__flash_duration`: contribution `-0.009216`
- `lag_04__T_flash_duration_sum`: contribution `-0.004232`
- `lag_05__CT_active_infernos`: contribution `-0.003869`

### tick `101251`, seconds `46.50`, LSTM delta `-0.3036`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.034215`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.033040`
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.015412`
- `lag_00__T_kills_last_3s`: contribution `-0.014402`
- `lag_00__CT_kills_last_3s`: contribution `-0.012654`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103171`, seconds `76.50`, LSTM delta `-0.2236`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.017108`
- `lag_00__T_kills_last_3s`: contribution `-0.014402`
- `lag_05__CT_place_HOUSE`: contribution `-0.013100`
- `lag_00__CT_place_SIDEENTRANCE`: contribution `-0.012323`
- `lag_00__T4__duck_amount`: contribution `-0.009515`

Top utility-only movements:
- `lag_00__CT5__smoke`: contribution `-0.003918`

### tick `101059`, seconds `43.50`, LSTM delta `+0.2031`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.017108`
- `lag_00__CT_kills_last_3s`: contribution `+0.012654`
- `lag_02__T1__duck_amount`: contribution `+0.008077`
- `lag_13__CT_place_SIDEENTRANCE`: contribution `+0.007568`
- `lag_00__CT5__duck_amount`: contribution `+0.006995`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103491`, seconds `81.50`, LSTM delta `+0.1865`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.017108`
- `lag_00__T_bomb_zone_count`: contribution `+0.016352`
- `lag_00__CT_kills_last_3s`: contribution `+0.012654`
- `lag_08__T_bomb_zone_count`: contribution `+0.009598`
- `lag_01__CT_duck_amount_mean`: contribution `+0.009597`

Top utility-only movements:
- No utility movement among the top local contributors.
