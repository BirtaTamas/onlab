# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `17`

## Largest probability jumps

- tick `147295`, seconds `50.50`, LSTM `0.7849`, delta `+0.2481`
- tick `147583`, seconds `55.00`, LSTM `0.5535`, delta `-0.1953`
- tick `148255`, seconds `65.50`, LSTM `0.9211`, delta `+0.1933`
- tick `148223`, seconds `65.00`, LSTM `0.7278`, delta `+0.1663`
- tick `147199`, seconds `49.00`, LSTM `0.5277`, delta `-0.1543`
- tick `147135`, seconds `48.00`, LSTM `0.6371`, delta `+0.1065`
- tick `147519`, seconds `54.00`, LSTM `0.7414`, delta `+0.0677`
- tick `147327`, seconds `51.00`, LSTM `0.7373`, delta `-0.0476`
- tick `147455`, seconds `53.00`, LSTM `0.7175`, delta `-0.0463`
- tick `147167`, seconds `48.50`, LSTM `0.6820`, delta `+0.0449`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002877`, |coef| `0.002877`
- `lag_07__T_flashes_last_5s`: coefficient `-0.002868`, |coef| `0.002868`
- `lag_00__CT_kills_last_3s`: coefficient `0.002589`, |coef| `0.002589`
- `lag_08__T_flashes_last_5s`: coefficient `-0.002310`, |coef| `0.002310`
- `lag_00__T_macro_B`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001468`, |coef| `0.001468`
- `lag_06__T_flashes_last_5s`: coefficient `-0.001386`, |coef| `0.001386`
- `lag_15__CT_place_LONGDOG`: coefficient `-0.001362`, |coef| `0.001362`
- `lag_00__damage_diff_last_5s`: coefficient `0.001345`, |coef| `0.001345`
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001317`, |coef| `0.001317`
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_14__T_kills_last_3s`: coefficient `-0.001213`, |coef| `0.001213`
- `lag_11__T5__shots_fired`: coefficient `0.001197`, |coef| `0.001197`
- `lag_12__T2__is_walking`: coefficient `-0.001188`, |coef| `0.001188`

## Top 10 utility ridge features

- `lag_07__T_flashes_last_5s`: coefficient `-0.002868` (lowers CT win probability)
- `lag_08__T_flashes_last_5s`: coefficient `-0.002310` (lowers CT win probability)
- `lag_06__T_flashes_last_5s`: coefficient `-0.001386` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001317` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.001059` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001056` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.001002` (raises CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001001` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000976` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000975` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002877` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002589` (raises CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001770` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001770` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001468` (raises CT win probability)
- `lag_15__CT_place_LONGDOG`: coefficient `-0.001362` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001345` (raises CT win probability)
- `lag_00__T_bomb_carrier_alive`: coefficient `-0.001240` (lowers CT win probability)
- `lag_14__T_kills_last_3s`: coefficient `-0.001213` (lowers CT win probability)
- `lag_11__T5__shots_fired`: coefficient `0.001197` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `147295`, seconds `50.50`, LSTM delta `+0.2481`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `+0.010672`
- `lag_00__CT_kills_last_3s`: contribution `+0.007475`
- `lag_00__kill_diff_last_3s`: contribution `+0.006925`
- `lag_02__T5__shots_fired`: contribution `+0.006326`
- `lag_02__T_shots_fired_sum`: contribution `+0.005981`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `+0.002901`

### tick `147583`, seconds `55.00`, LSTM delta `-0.1953`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.010202`
- `lag_11__T_shots_fired_sum`: contribution `-0.008342`
- `lag_11__T5__shots_fired`: contribution `-0.007357`
- `lag_00__kill_diff_last_3s`: contribution `-0.006925`
- `lag_01__T_shots_fired_sum`: contribution `-0.004902`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `148255`, seconds `65.50`, LSTM delta `+0.1933`

Top all feature movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.020931`
- `lag_00__CT_kills_last_3s`: contribution `+0.007475`
- `lag_00__kill_diff_last_3s`: contribution `+0.006925`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005943`
- `lag_01__T_shots_fired_sum`: contribution `+0.004902`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.020931`
- `lag_01__T_A_site_active_infernos`: contribution `+0.002981`
- `lag_01__T_B_site_active_infernos`: contribution `+0.002759`
- `lag_15__T_A_site_active_infernos`: contribution `+0.002367`

### tick `148223`, seconds `65.00`, LSTM delta `+0.1663`

Top all feature movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.025985`
- `lag_15__CT_place_ENTRANCE`: contribution `+0.008568`
- `lag_00__CT_shots_fired_sum`: contribution `-0.008161`
- `lag_00__CT_kills_last_3s`: contribution `+0.007475`
- `lag_00__kill_diff_last_3s`: contribution `+0.006925`

Top utility-only movements:
- `lag_07__T_flashes_last_5s`: contribution `+0.025985`
- `lag_00__T_A_site_active_infernos`: contribution `+0.003921`
- `lag_00__T_B_site_active_infernos`: contribution `+0.002995`
- `lag_14__T_A_site_active_infernos`: contribution `+0.002981`

### tick `147199`, seconds `49.00`, LSTM delta `-0.1543`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.006925`
- `lag_04__CT_place_LONGDOG`: contribution `-0.005987`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005101`
- `lag_01__T_shots_fired_sum`: contribution `-0.004085`
- `lag_01__CT_shots_fired_sum`: contribution `-0.003714`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.002901`
