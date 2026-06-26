# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `16`

## Largest probability jumps

- tick `120936`, seconds `31.50`, LSTM `0.4057`, delta `-0.2892`
- tick `120872`, seconds `30.50`, LSTM `0.6905`, delta `+0.2864`
- tick `122376`, seconds `54.00`, LSTM `0.4837`, delta `-0.2582`
- tick `122280`, seconds `52.50`, LSTM `0.6869`, delta `+0.2538`
- tick `119880`, seconds `15.00`, LSTM `0.2424`, delta `-0.2401`
- tick `123496`, seconds `71.50`, LSTM `0.0624`, delta `-0.2365`
- tick `120776`, seconds `29.00`, LSTM `0.4122`, delta `+0.2316`
- tick `121128`, seconds `34.50`, LSTM `0.4338`, delta `+0.1000`
- tick `121864`, seconds `46.00`, LSTM `0.3949`, delta `-0.0722`
- tick `122248`, seconds `52.00`, LSTM `0.4331`, delta `+0.0680`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006888`, |coef| `0.006888`
- `lag_00__T_kills_last_3s`: coefficient `-0.005111`, |coef| `0.005111`
- `lag_00__damage_diff_last_5s`: coefficient `0.004874`, |coef| `0.004874`
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.004075`, |coef| `0.004075`
- `lag_00__CT_kills_last_3s`: coefficient `0.003605`, |coef| `0.003605`
- `lag_05__CT_place_ARAMP`: coefficient `-0.003555`, |coef| `0.003555`
- `lag_15__CT2__duck_amount`: coefficient `0.003286`, |coef| `0.003286`
- `lag_00__T_damage_last_5s`: coefficient `-0.003164`, |coef| `0.003164`
- `lag_08__CT_place_ARAMP`: coefficient `0.003062`, |coef| `0.003062`
- `lag_00__T5__duck_amount`: coefficient `-0.002787`, |coef| `0.002787`
- `lag_04__T_place_TUNNELSTAIRS`: coefficient `-0.002760`, |coef| `0.002760`
- `lag_12__T2__is_scoped`: coefficient `-0.002626`, |coef| `0.002626`
- `lag_10__CT_place_BDOORS`: coefficient `-0.002417`, |coef| `0.002417`
- `lag_13__T_B_site_active_infernos`: coefficient `0.002412`, |coef| `0.002412`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002380`, |coef| `0.002380`

## Top 10 utility ridge features

- `lag_13__T_B_site_active_infernos`: coefficient `0.002412` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.001786` (raises CT win probability)
- `lag_00__CT2__smoke`: coefficient `0.001700` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001593` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.001482` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.001302` (raises CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.001098` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001086` (raises CT win probability)
- `lag_15__T3__flash_duration`: coefficient `-0.001083` (lowers CT win probability)
- `lag_00__smoke_inv_diff`: coefficient `0.001080` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006888` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.005111` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004874` (raises CT win probability)
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `0.004075` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003605` (raises CT win probability)
- `lag_05__CT_place_ARAMP`: coefficient `-0.003555` (lowers CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `0.003286` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003164` (lowers CT win probability)
- `lag_08__CT_place_ARAMP`: coefficient `0.003062` (raises CT win probability)
- `lag_00__T5__duck_amount`: coefficient `-0.002787` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `120936`, seconds `31.50`, LSTM delta `-0.2892`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.016579`
- `lag_00__T_kills_last_3s`: contribution `-0.016192`
- `lag_00__CT_place_ARAMP`: contribution `-0.009546`
- `lag_00__T_shots_fired_sum`: contribution `-0.007916`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.007868`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120872`, seconds `30.50`, LSTM delta `+0.2864`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016579`
- `lag_00__CT_kills_last_3s`: contribution `+0.010407`
- `lag_07__T_place_TUNNELSTAIRS`: contribution `+0.009870`
- `lag_06__CT_place_BDOORS`: contribution `+0.009626`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008267`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `122376`, seconds `54.00`, LSTM delta `-0.2582`

Top all feature movements:
- `lag_12__T2__is_scoped`: contribution `-0.023150`
- `lag_08__CT_place_ARAMP`: contribution `-0.019072`
- `lag_02__CT_place_LOWERTUNNEL`: contribution `-0.017017`
- `lag_00__kill_diff_last_3s`: contribution `-0.016579`
- `lag_00__T_kills_last_3s`: contribution `-0.016192`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `-0.003680`

### tick `122280`, seconds `52.50`, LSTM delta `+0.2538`

Top all feature movements:
- `lag_05__CT_place_ARAMP`: contribution `+0.022142`
- `lag_09__T2__is_scoped`: contribution `+0.019352`
- `lag_00__kill_diff_last_3s`: contribution `+0.016579`
- `lag_15__CT2__duck_amount`: contribution `+0.012020`
- `lag_00__CT_kills_last_3s`: contribution `+0.010407`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.006820`
- `lag_13__T_active_infernos`: contribution `+0.003720`

### tick `119880`, seconds `15.00`, LSTM delta `-0.2401`

Top all feature movements:
- `lag_11__CT_place_HOLE`: contribution `-0.018497`
- `lag_00__kill_diff_last_3s`: contribution `-0.016579`
- `lag_00__T_kills_last_3s`: contribution `-0.016192`
- `lag_03__T_place_TUNNELSTAIRS`: contribution `-0.014865`
- `lag_00__damage_diff_last_5s`: contribution `-0.010996`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.007122`
- `lag_05__T3__flash_duration`: contribution `-0.005641`
- `lag_14__CT1__flash_duration`: contribution `-0.005603`
- `lag_14__CT_flash_duration_sum`: contribution `-0.004943`
- `lag_14__CT2__flash_duration`: contribution `-0.004603`
