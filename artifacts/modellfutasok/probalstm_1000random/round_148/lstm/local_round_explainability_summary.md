# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `16`

## Largest probability jumps

- tick `157148`, seconds `126.50`, LSTM `0.4740`, delta `+0.3387`
- tick `153276`, seconds `66.00`, LSTM `0.0663`, delta `-0.3251`
- tick `152924`, seconds `60.50`, LSTM `0.7580`, delta `+0.2213`
- tick `156604`, seconds `118.00`, LSTM `0.2325`, delta `+0.1984`
- tick `153180`, seconds `64.50`, LSTM `0.5750`, delta `-0.1904`
- tick `156860`, seconds `122.00`, LSTM `0.2850`, delta `+0.1765`
- tick `153244`, seconds `65.50`, LSTM `0.3914`, delta `-0.1284`
- tick `151708`, seconds `41.50`, LSTM `0.6506`, delta `-0.0924`
- tick `156700`, seconds `119.50`, LSTM `0.1521`, delta `-0.0902`
- tick `157244`, seconds `128.00`, LSTM `0.5758`, delta `+0.0791`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003117`, |coef| `0.003117`
- `lag_06__T_place_SNIPERSNEST`: coefficient `0.002778`, |coef| `0.002778`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.002584`, |coef| `0.002584`
- `lag_00__T_place_STAIRS`: coefficient `0.002429`, |coef| `0.002429`
- `lag_00__damage_diff_last_5s`: coefficient `0.002388`, |coef| `0.002388`
- `lag_00__CT_place_WALKWAY`: coefficient `0.002379`, |coef| `0.002379`
- `lag_03__T_shots_fired_sum`: coefficient `0.002313`, |coef| `0.002313`
- `lag_09__T_place_STAIRS`: coefficient `0.002280`, |coef| `0.002280`
- `lag_01__CT_place_CANAL`: coefficient `0.002136`, |coef| `0.002136`
- `lag_10__T_place_CONSTRUCTION`: coefficient `0.002082`, |coef| `0.002082`
- `lag_06__T_place_UNDERA`: coefficient `-0.002049`, |coef| `0.002049`
- `lag_08__T_place_STAIRS`: coefficient `-0.002043`, |coef| `0.002043`
- `lag_08__T_shots_fired_sum`: coefficient `-0.002023`, |coef| `0.002023`
- `lag_00__T_kills_last_3s`: coefficient `-0.002011`, |coef| `0.002011`
- `lag_00__T_place_WATER`: coefficient `-0.001975`, |coef| `0.001975`

## Top 10 utility ridge features

- `lag_00__CT5__flash`: coefficient `0.001160` (raises CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `-0.001002` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000937` (lowers CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.000801` (raises CT win probability)
- `lag_11__T2__smoke`: coefficient `0.000744` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000705` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000687` (raises CT win probability)
- `lag_03__CT2__flash`: coefficient `0.000634` (raises CT win probability)
- `lag_11__T2__utility_total`: coefficient `0.000618` (raises CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000587` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003117` (raises CT win probability)
- `lag_06__T_place_SNIPERSNEST`: coefficient `0.002778` (raises CT win probability)
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.002584` (lowers CT win probability)
- `lag_00__T_place_STAIRS`: coefficient `0.002429` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002388` (raises CT win probability)
- `lag_00__CT_place_WALKWAY`: coefficient `0.002379` (raises CT win probability)
- `lag_03__T_shots_fired_sum`: coefficient `0.002313` (raises CT win probability)
- `lag_09__T_place_STAIRS`: coefficient `0.002280` (raises CT win probability)
- `lag_01__CT_place_CANAL`: coefficient `0.002136` (raises CT win probability)
- `lag_10__T_place_CONSTRUCTION`: coefficient `0.002082` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `157148`, seconds `126.50`, LSTM delta `+0.3387`

Top all feature movements:
- `lag_06__T_place_SNIPERSNEST`: contribution `+0.049365`
- `lag_00__T_place_SNIPERSNEST`: contribution `+0.045913`
- `lag_09__T_place_STAIRS`: contribution `+0.043643`
- `lag_08__T_place_STAIRS`: contribution `+0.039116`
- `lag_06__T_place_UNDERA`: contribution `+0.032016`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `153276`, seconds `66.00`, LSTM delta `-0.3251`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `-0.015820`
- `lag_15__T_place_PIPE`: contribution `-0.015541`
- `lag_11__CT_place_LOBBY`: contribution `-0.015303`
- `lag_01__CT_place_CANAL`: contribution `-0.012979`
- `lag_00__CT_place_WALKWAY`: contribution `-0.011679`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `152924`, seconds `60.50`, LSTM delta `+0.2213`

Top all feature movements:
- `lag_10__T_place_CONSTRUCTION`: contribution `+0.025874`
- `lag_13__T_place_PIPE`: contribution `+0.015824`
- `lag_10__T_place_PIPE`: contribution `+0.014235`
- `lag_00__CT_place_LOBBY`: contribution `+0.013673`
- `lag_04__T_place_PIPE`: contribution `+0.011657`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `+0.003536`

### tick `156604`, seconds `118.00`, LSTM delta `+0.1984`

Top all feature movements:
- `lag_02__T_place_BACKOFA`: contribution `+0.044461`
- `lag_13__T_place_BACKOFA`: contribution `+0.032458`
- `lag_11__T_place_BACKOFA`: contribution `+0.015525`
- `lag_01__CT_place_STAIRS`: contribution `+0.009650`
- `lag_00__kill_diff_last_3s`: contribution `+0.007503`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `153180`, seconds `64.50`, LSTM delta `-0.1904`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.029475`
- `lag_03__T3__shots_fired`: contribution `-0.019337`
- `lag_05__CT_place_LOBBY`: contribution `-0.012439`
- `lag_00__CT_place_WALKWAY`: contribution `-0.011679`
- `lag_12__T_place_PIPE`: contribution `-0.008202`

Top utility-only movements:
- No utility movement among the top local contributors.
