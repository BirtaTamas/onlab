# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `9`

## Largest probability jumps

- tick `81225`, seconds `59.50`, LSTM `0.7686`, delta `+0.2132`
- tick `81129`, seconds `58.00`, LSTM `0.5532`, delta `-0.1875`
- tick `81321`, seconds `61.00`, LSTM `0.9381`, delta `+0.1626`
- tick `80169`, seconds `43.00`, LSTM `0.6651`, delta `+0.1316`
- tick `78825`, seconds `22.00`, LSTM `0.5814`, delta `+0.0629`
- tick `81289`, seconds `60.50`, LSTM `0.7755`, delta `+0.0504`
- tick `81097`, seconds `57.50`, LSTM `0.7407`, delta `+0.0474`
- tick `81257`, seconds `60.00`, LSTM `0.7251`, delta `-0.0434`
- tick `80553`, seconds `49.00`, LSTM `0.7124`, delta `+0.0334`
- tick `80297`, seconds `45.00`, LSTM `0.7007`, delta `+0.0273`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003384`, |coef| `0.003384`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002830`, |coef| `0.002830`
- `lag_05__T_place_SQUEAKY`: coefficient `0.002600`, |coef| `0.002600`
- `lag_00__CT_kills_last_3s`: coefficient `0.002281`, |coef| `0.002281`
- `lag_01__CT_place_MINI`: coefficient `0.002196`, |coef| `0.002196`
- `lag_11__T3__duck_amount`: coefficient `0.002176`, |coef| `0.002176`
- `lag_08__T_place_SQUEAKY`: coefficient `-0.001983`, |coef| `0.001983`
- `lag_00__T_kills_last_3s`: coefficient `-0.001951`, |coef| `0.001951`
- `lag_13__T_place_SQUEAKY`: coefficient `-0.001785`, |coef| `0.001785`
- `lag_00__damage_diff_last_5s`: coefficient `0.001629`, |coef| `0.001629`
- `lag_04__T1__duck_amount`: coefficient `0.001606`, |coef| `0.001606`
- `lag_01__CT2__is_walking`: coefficient `-0.001471`, |coef| `0.001471`
- `lag_05__CT2__is_walking`: coefficient `-0.001467`, |coef| `0.001467`
- `lag_05__CT_walking_count`: coefficient `-0.001465`, |coef| `0.001465`
- `lag_13__T3__duck_amount`: coefficient `-0.001454`, |coef| `0.001454`

## Top 10 utility ridge features

- `lag_15__T1__molly`: coefficient `0.000929` (raises CT win probability)
- `lag_15__T2__molly`: coefficient `-0.000848` (lowers CT win probability)
- `lag_15__T_A_site_active_smokes`: coefficient `-0.000764` (lowers CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `-0.000727` (lowers CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000676` (raises CT win probability)
- `lag_03__CT1__flash`: coefficient `-0.000615` (lowers CT win probability)
- `lag_12__T_B_site_active_smokes`: coefficient `0.000589` (raises CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000561` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000541` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000540` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003384` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002830` (raises CT win probability)
- `lag_05__T_place_SQUEAKY`: coefficient `0.002600` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002281` (raises CT win probability)
- `lag_01__CT_place_MINI`: coefficient `0.002196` (raises CT win probability)
- `lag_11__T3__duck_amount`: coefficient `0.002176` (raises CT win probability)
- `lag_08__T_place_SQUEAKY`: coefficient `-0.001983` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001951` (lowers CT win probability)
- `lag_13__T_place_SQUEAKY`: coefficient `-0.001785` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001629` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `81225`, seconds `59.50`, LSTM delta `+0.2132`

Top all feature movements:
- `lag_01__CT_place_MINI`: contribution `+0.013461`
- `lag_08__T_place_SQUEAKY`: contribution `+0.012344`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009831`
- `lag_11__T3__duck_amount`: contribution `+0.008205`
- `lag_00__kill_diff_last_3s`: contribution `+0.008144`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `81129`, seconds `58.00`, LSTM delta `-0.1875`

Top all feature movements:
- `lag_05__T_place_SQUEAKY`: contribution `-0.016189`
- `lag_13__T_place_SQUEAKY`: contribution `-0.011113`
- `lag_00__kill_diff_last_3s`: contribution `-0.008144`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007865`
- `lag_04__T1__duck_amount`: contribution `-0.006289`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `81321`, seconds `61.00`, LSTM delta `+0.1626`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016288`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011798`
- `lag_11__T3__duck_amount`: contribution `+0.006883`
- `lag_00__CT_kills_last_3s`: contribution `+0.006584`
- `lag_04__CT_place_MINI`: contribution `+0.006394`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `80169`, seconds `43.00`, LSTM delta `+0.1316`

Top all feature movements:
- `lag_03__CT_place_SECRET`: contribution `+0.011822`
- `lag_00__kill_diff_last_3s`: contribution `+0.008144`
- `lag_11__T3__duck_amount`: contribution `+0.007371`
- `lag_00__CT_kills_last_3s`: contribution `+0.006584`
- `lag_12__T_place_SQUEAKY`: contribution `+0.006376`

Top utility-only movements:
- `lag_00__T4__flash`: contribution `+0.001523`

### tick `78825`, seconds `22.00`, LSTM delta `+0.0629`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008144`
- `lag_00__CT_kills_last_3s`: contribution `+0.006584`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005899`
- `lag_00__damage_diff_last_5s`: contribution `+0.003675`
- `lag_13__T_place_LOBBY`: contribution `+0.003193`

Top utility-only movements:
- No utility movement among the top local contributors.
