# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `11`

## Largest probability jumps

- tick `80167`, seconds `68.00`, LSTM `0.9180`, delta `+0.3103`
- tick `81991`, seconds `96.50`, LSTM `0.9157`, delta `+0.2326`
- tick `81863`, seconds `94.50`, LSTM `0.6699`, delta `-0.1207`
- tick `80135`, seconds `67.50`, LSTM `0.6077`, delta `+0.1166`
- tick `77063`, seconds `19.50`, LSTM `0.3714`, delta `-0.0812`
- tick `77127`, seconds `20.50`, LSTM `0.3126`, delta `-0.0661`
- tick `77543`, seconds `27.00`, LSTM `0.3895`, delta `+0.0565`
- tick `81671`, seconds `91.50`, LSTM `0.8433`, delta `-0.0562`
- tick `77895`, seconds `32.50`, LSTM `0.4231`, delta `+0.0342`
- tick `77927`, seconds `33.00`, LSTM `0.4559`, delta `+0.0328`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003508`, |coef| `0.003508`
- `lag_00__kill_diff_last_3s`: coefficient `0.003355`, |coef| `0.003355`
- `lag_00__T_place_TUNNELS`: coefficient `-0.003064`, |coef| `0.003064`
- `lag_00__damage_diff_last_5s`: coefficient `0.003010`, |coef| `0.003010`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002697`, |coef| `0.002697`
- `lag_00__bomb_events_last_5s`: coefficient `0.002273`, |coef| `0.002273`
- `lag_13__CT_place_MINI`: coefficient `-0.001776`, |coef| `0.001776`
- `lag_00__CT_damage_last_5s`: coefficient `0.001727`, |coef| `0.001727`
- `lag_06__T4__is_scoped`: coefficient `-0.001538`, |coef| `0.001538`
- `lag_00__alive_diff`: coefficient `0.001487`, |coef| `0.001487`
- `lag_01__damage_diff_last_5s`: coefficient `0.001475`, |coef| `0.001475`
- `lag_01__T5__shots_fired`: coefficient `0.001433`, |coef| `0.001433`
- `lag_04__CT_place_RAFTERS`: coefficient `-0.001433`, |coef| `0.001433`
- `lag_01__kill_diff_last_3s`: coefficient `0.001412`, |coef| `0.001412`
- `lag_00__CT2__shots_fired`: coefficient `0.001340`, |coef| `0.001340`

## Top 10 utility ridge features

- `lag_05__CT_A_site_active_infernos`: coefficient `0.001107` (raises CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.001072` (lowers CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.001066` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.001042` (raises CT win probability)
- `lag_00__T3__molly`: coefficient `-0.001034` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.001031` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000948` (lowers CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000930` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000865` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000862` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003508` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003355` (raises CT win probability)
- `lag_00__T_place_TUNNELS`: coefficient `-0.003064` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003010` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002697` (raises CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.002273` (raises CT win probability)
- `lag_13__CT_place_MINI`: coefficient `-0.001776` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001727` (raises CT win probability)
- `lag_06__T4__is_scoped`: coefficient `-0.001538` (lowers CT win probability)
- `lag_00__alive_diff`: coefficient `0.001487` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `80167`, seconds `68.00`, LSTM delta `+0.3103`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.020258`
- `lag_01__CT_shots_fired_sum`: contribution `+0.018734`
- `lag_00__T_place_TUNNELS`: contribution `+0.017178`
- `lag_00__kill_diff_last_3s`: contribution `+0.016150`
- `lag_00__bomb_events_last_5s`: contribution `+0.014247`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `81991`, seconds `96.50`, LSTM delta `+0.2326`

Top all feature movements:
- `lag_13__CT_place_MINI`: contribution `+0.010892`
- `lag_00__damage_diff_last_5s`: contribution `+0.010320`
- `lag_00__CT_kills_last_3s`: contribution `+0.010129`
- `lag_00__T_place_TUNNELS`: contribution `+0.008589`
- `lag_00__kill_diff_last_3s`: contribution `+0.008075`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.003905`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.003580`

### tick `81863`, seconds `94.50`, LSTM delta `-0.1207`

Top all feature movements:
- `lag_00__CT_place_RAFTERS`: contribution `-0.006617`
- `lag_09__CT_place_MINI`: contribution `-0.005799`
- `lag_03__CT_place_HEAVEN`: contribution `-0.005023`
- `lag_00__damage_diff_last_5s`: contribution `-0.003734`
- `lag_00__CT_place_TUNNELS`: contribution `-0.003687`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80135`, seconds `67.50`, LSTM delta `+0.1166`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010129`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009026`
- `lag_00__T_place_TUNNELS`: contribution `+0.008589`
- `lag_00__kill_diff_last_3s`: contribution `+0.008075`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007494`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `77063`, seconds `19.50`, LSTM delta `-0.0812`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.021656`
- `lag_15__T_place_SILO`: contribution `-0.007612`
- `lag_00__CT_place_RAFTERS`: contribution `-0.006617`
- `lag_01__T3__duck_amount`: contribution `-0.003515`
- `lag_02__T_place_ROOF`: contribution `-0.003257`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `-0.001880`
- `lag_07__CT2__flash_duration`: contribution `-0.001863`
