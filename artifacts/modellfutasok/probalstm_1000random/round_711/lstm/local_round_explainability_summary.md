# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `73296`, seconds `72.00`, LSTM `0.0809`, delta `-0.1499`
- tick `73136`, seconds `69.50`, LSTM `0.2670`, delta `-0.1475`
- tick `72112`, seconds `53.50`, LSTM `0.3045`, delta `-0.0480`
- tick `72496`, seconds `59.50`, LSTM `0.4217`, delta `+0.0455`
- tick `72944`, seconds `66.50`, LSTM `0.3809`, delta `-0.0447`
- tick `73488`, seconds `75.00`, LSTM `0.0188`, delta `-0.0411`
- tick `69072`, seconds `6.00`, LSTM `0.3782`, delta `-0.0367`
- tick `72400`, seconds `58.00`, LSTM `0.3646`, delta `+0.0360`
- tick `71600`, seconds `45.50`, LSTM `0.3671`, delta `-0.0353`
- tick `71920`, seconds `50.50`, LSTM `0.4105`, delta `+0.0336`

## Top 15 local ridge features

- `lag_15__CT_place_WALKWAY`: coefficient `-0.001567`, |coef| `0.001567`
- `lag_00__T_kills_last_3s`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_00__T_damage_last_5s`: coefficient `-0.001404`, |coef| `0.001404`
- `lag_04__T_place_TSTAIRS`: coefficient `0.001399`, |coef| `0.001399`
- `lag_12__CT_place_WALKWAY`: coefficient `-0.001317`, |coef| `0.001317`
- `lag_06__T_place_TSTAIRS`: coefficient `-0.001307`, |coef| `0.001307`
- `lag_06__T_place_STREET`: coefficient `0.001306`, |coef| `0.001306`
- `lag_02__T_place_MIDDOORS`: coefficient `0.001259`, |coef| `0.001259`
- `lag_00__damage_diff_last_5s`: coefficient `0.001216`, |coef| `0.001216`
- `lag_11__T_place_TSTAIRS`: coefficient `-0.001182`, |coef| `0.001182`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001180`, |coef| `0.001180`
- `lag_07__CT_place_WALKWAY`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_00__kill_diff_last_3s`: coefficient `0.001073`, |coef| `0.001073`
- `lag_09__T_place_TSTAIRS`: coefficient `0.001063`, |coef| `0.001063`
- `lag_05__T_place_MIDDOORS`: coefficient `0.001034`, |coef| `0.001034`

## Top 10 utility ridge features

- `lag_01__CT_B_site_active_smokes`: coefficient `0.000487` (raises CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `0.000462` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000455` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `-0.000430` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000421` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `0.000375` (raises CT win probability)
- `lag_01__CT_active_smokes`: coefficient `0.000363` (raises CT win probability)
- `lag_06__CT4__smoke`: coefficient `-0.000341` (lowers CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `-0.000332` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000330` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_WALKWAY`: coefficient `-0.001567` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001527` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001404` (lowers CT win probability)
- `lag_04__T_place_TSTAIRS`: coefficient `0.001399` (raises CT win probability)
- `lag_12__CT_place_WALKWAY`: coefficient `-0.001317` (lowers CT win probability)
- `lag_06__T_place_TSTAIRS`: coefficient `-0.001307` (lowers CT win probability)
- `lag_06__T_place_STREET`: coefficient `0.001306` (raises CT win probability)
- `lag_02__T_place_MIDDOORS`: coefficient `0.001259` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001216` (raises CT win probability)
- `lag_11__T_place_TSTAIRS`: coefficient `-0.001182` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `73296`, seconds `72.00`, LSTM delta `-0.1499`

Top all feature movements:
- `lag_11__T_place_TSTAIRS`: contribution `-0.006701`
- `lag_12__CT_place_WALKWAY`: contribution `-0.006463`
- `lag_09__T_place_TSTAIRS`: contribution `-0.006024`
- `lag_00__T_kills_last_3s`: contribution `-0.004836`
- `lag_11__T_place_STREET`: contribution `-0.004430`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73136`, seconds `69.50`, LSTM delta `-0.1475`

Top all feature movements:
- `lag_04__T_place_TSTAIRS`: contribution `-0.007932`
- `lag_15__CT_place_WALKWAY`: contribution `-0.007691`
- `lag_06__T_place_TSTAIRS`: contribution `-0.007407`
- `lag_06__T_place_STREET`: contribution `-0.007178`
- `lag_02__T_place_MIDDOORS`: contribution `-0.005353`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72112`, seconds `53.50`, LSTM delta `-0.0480`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.005243`
- `lag_01__CT_place_TUNNEL`: contribution `-0.004658`
- `lag_06__CT5__duck_amount`: contribution `-0.003066`
- `lag_04__CT_place_CTSIDEUPPER`: contribution `-0.003059`
- `lag_00__T1__duck_amount`: contribution `-0.002978`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72496`, seconds `59.50`, LSTM delta `+0.0455`

Top all feature movements:
- `lag_12__CT_place_BRICKS`: contribution `+0.009102`
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `+0.005328`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.002940`
- `lag_05__CT_place_TUNNEL`: contribution `+0.002380`
- `lag_00__CT_scoped_count`: contribution `+0.001976`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `+0.001353`

### tick `72944`, seconds `66.50`, LSTM delta `-0.0447`

Top all feature movements:
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `-0.009967`
- `lag_09__CT_place_WALKWAY`: contribution `-0.004014`
- `lag_00__T_place_TSTAIRS`: contribution `-0.003432`
- `lag_14__CT_place_HEAVEN`: contribution `-0.003174`
- `lag_07__CT5__is_walking`: contribution `-0.002272`

Top utility-only movements:
- `lag_14__T_A_site_active_infernos`: contribution `-0.001353`
