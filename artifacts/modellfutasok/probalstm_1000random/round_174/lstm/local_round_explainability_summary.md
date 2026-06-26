# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `11`

## Largest probability jumps

- tick `72744`, seconds `94.00`, LSTM `0.5317`, delta `+0.3343`
- tick `71016`, seconds `67.00`, LSTM `0.3628`, delta `-0.2572`
- tick `71240`, seconds `70.50`, LSTM `0.5669`, delta `+0.2059`
- tick `72904`, seconds `96.50`, LSTM `0.8630`, delta `+0.1489`
- tick `71208`, seconds `70.00`, LSTM `0.3610`, delta `+0.1258`
- tick `68296`, seconds `24.50`, LSTM `0.7665`, delta `+0.1175`
- tick `68648`, seconds `30.00`, LSTM `0.7063`, delta `-0.1155`
- tick `70952`, seconds `66.00`, LSTM `0.6279`, delta `+0.1153`
- tick `72808`, seconds `95.00`, LSTM `0.6787`, delta `+0.1002`
- tick `71656`, seconds `77.00`, LSTM `0.3631`, delta `-0.0832`

## Top 15 local ridge features

- `lag_02__T_place_WALKWAY`: coefficient `-0.003133`, |coef| `0.003133`
- `lag_00__kill_diff_last_3s`: coefficient `0.002901`, |coef| `0.002901`
- `lag_00__T_place_HEAVEN`: coefficient `-0.002769`, |coef| `0.002769`
- `lag_14__CT_place_TUNNEL`: coefficient `0.002226`, |coef| `0.002226`
- `lag_00__CT_place_WALKWAY`: coefficient `-0.002190`, |coef| `0.002190`
- `lag_14__T_place_MAIN`: coefficient `-0.002142`, |coef| `0.002142`
- `lag_03__CT_place_TUNNEL`: coefficient `-0.002078`, |coef| `0.002078`
- `lag_00__CT_kills_last_3s`: coefficient `0.001914`, |coef| `0.001914`
- `lag_00__T_place_WALKWAY`: coefficient `-0.001866`, |coef| `0.001866`
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001732`, |coef| `0.001732`
- `lag_02__CT_place_TUNNELSTAIRS`: coefficient `0.001724`, |coef| `0.001724`
- `lag_00__T_kills_last_3s`: coefficient `-0.001719`, |coef| `0.001719`
- `lag_02__CT_place_TUNNEL`: coefficient `-0.001699`, |coef| `0.001699`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001697`, |coef| `0.001697`
- `lag_01__T_place_MAIN`: coefficient `-0.001671`, |coef| `0.001671`

## Top 10 utility ridge features

- `lag_14__CT_B_site_active_infernos`: coefficient `0.001048` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.001009` (lowers CT win probability)
- `lag_14__T_mollies_last_5s`: coefficient `0.001002` (raises CT win probability)
- `lag_09__T4__smoke`: coefficient `0.000881` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.000846` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000834` (raises CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.000815` (lowers CT win probability)
- `lag_05__T_mollies_last_5s`: coefficient `0.000801` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.000763` (lowers CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `-0.000755` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_WALKWAY`: coefficient `-0.003133` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002901` (raises CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.002769` (lowers CT win probability)
- `lag_14__CT_place_TUNNEL`: coefficient `0.002226` (raises CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `-0.002190` (lowers CT win probability)
- `lag_14__T_place_MAIN`: coefficient `-0.002142` (lowers CT win probability)
- `lag_03__CT_place_TUNNEL`: coefficient `-0.002078` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001914` (raises CT win probability)
- `lag_00__T_place_WALKWAY`: coefficient `-0.001866` (lowers CT win probability)
- `lag_03__CT_place_TUNNELSTAIRS`: coefficient `0.001732` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72744`, seconds `94.00`, LSTM delta `+0.3343`

Top all feature movements:
- `lag_02__T_place_WALKWAY`: contribution `+0.042607`
- `lag_14__CT_place_TUNNEL`: contribution `+0.035761`
- `lag_00__T_place_HEAVEN`: contribution `+0.033974`
- `lag_03__CT_place_TUNNEL`: contribution `+0.033375`
- `lag_02__CT_place_TUNNEL`: contribution `+0.027284`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `71016`, seconds `67.00`, LSTM delta `-0.2572`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `-0.031885`
- `lag_01__CT_place_BRICKS`: contribution `-0.027054`
- `lag_14__T_place_MAIN`: contribution `-0.013847`
- `lag_01__T_place_MAIN`: contribution `-0.010804`
- `lag_00__CT_place_WALKWAY`: contribution `-0.010751`

Top utility-only movements:
- `lag_00__CT2__flash`: contribution `-0.003018`

### tick `71240`, seconds `70.50`, LSTM delta `+0.2059`

Top all feature movements:
- `lag_08__CT_place_BRICKS`: contribution `+0.028593`
- `lag_10__T_place_MAIN`: contribution `+0.008501`
- `lag_00__T_place_MAIN`: contribution `+0.007727`
- `lag_00__kill_diff_last_3s`: contribution `+0.006983`
- `lag_01__CT_place_WALKWAY`: contribution `+0.006496`

Top utility-only movements:
- `lag_07__CT2__flash`: contribution `+0.002543`

### tick `72904`, seconds `96.50`, LSTM delta `+0.1489`

Top all feature movements:
- `lag_00__T_place_HEAVEN`: contribution `+0.033974`
- `lag_07__T_place_WALKWAY`: contribution `+0.015748`
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `+0.011071`
- `lag_07__CT_place_TUNNELSTAIRS`: contribution `+0.010250`
- `lag_00__kill_diff_last_3s`: contribution `+0.006983`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.006119`

### tick `71208`, seconds `70.00`, LSTM delta `+0.1258`

Top all feature movements:
- `lag_06__CT_place_BRICKS`: contribution `+0.022316`
- `lag_00__CT_place_WALKWAY`: contribution `+0.010751`
- `lag_00__kill_diff_last_3s`: contribution `+0.006983`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005893`
- `lag_00__T_kills_last_3s`: contribution `+0.005445`

Top utility-only movements:
- `lag_06__CT2__flash`: contribution `+0.001922`
