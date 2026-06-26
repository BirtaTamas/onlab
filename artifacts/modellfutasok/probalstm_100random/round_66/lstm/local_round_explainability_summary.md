# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `11`

## Largest probability jumps

- tick `102129`, seconds `32.50`, LSTM `0.5776`, delta `+0.4502`
- tick `104689`, seconds `72.50`, LSTM `0.5017`, delta `+0.3775`
- tick `104369`, seconds `67.50`, LSTM `0.2064`, delta `-0.3752`
- tick `102193`, seconds `33.50`, LSTM `0.8419`, delta `+0.3377`
- tick `101937`, seconds `29.50`, LSTM `0.0431`, delta `-0.2670`
- tick `100817`, seconds `12.00`, LSTM `0.1864`, delta `-0.1447`
- tick `100849`, seconds `12.50`, LSTM `0.2991`, delta `+0.1127`
- tick `105361`, seconds `83.00`, LSTM `0.4562`, delta `-0.1111`
- tick `102225`, seconds `34.00`, LSTM `0.7351`, delta `-0.1068`
- tick `102577`, seconds `39.50`, LSTM `0.6426`, delta `-0.0864`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.009495`, |coef| `0.009495`
- `lag_00__T_kills_last_3s`: coefficient `-0.007331`, |coef| `0.007331`
- `lag_00__damage_diff_last_5s`: coefficient `0.006017`, |coef| `0.006017`
- `lag_12__CT_place_HOLE`: coefficient `0.005589`, |coef| `0.005589`
- `lag_04__CT_velocity_mean`: coefficient `0.005137`, |coef| `0.005137`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005085`, |coef| `0.005085`
- `lag_00__CT_kills_last_3s`: coefficient `0.004708`, |coef| `0.004708`
- `lag_08__CT_velocity_mean`: coefficient `-0.004505`, |coef| `0.004505`
- `lag_00__T_duck_amount_mean`: coefficient `-0.004323`, |coef| `0.004323`
- `lag_00__T_damage_last_5s`: coefficient `-0.004162`, |coef| `0.004162`
- `lag_00__CT4__alive`: coefficient `0.003774`, |coef| `0.003774`
- `lag_07__CT_place_HOLE`: coefficient `0.003736`, |coef| `0.003736`
- `lag_00__CT4__hp`: coefficient `0.003718`, |coef| `0.003718`
- `lag_00__CT_velocity_mean`: coefficient `-0.003705`, |coef| `0.003705`
- `lag_14__CT_duck_amount_mean`: coefficient `0.003560`, |coef| `0.003560`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005085` (lowers CT win probability)
- `lag_08__CT4__molly`: coefficient `-0.002565` (lowers CT win probability)
- `lag_01__CT4__molly`: coefficient `-0.002293` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.002275` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.002252` (raises CT win probability)
- `lag_09__CT4__molly`: coefficient `0.001862` (raises CT win probability)
- `lag_07__CT3__smoke`: coefficient `-0.001850` (lowers CT win probability)
- `lag_07__T4__molly`: coefficient `-0.001784` (lowers CT win probability)
- `lag_14__T_flash_alpha_mean`: coefficient `-0.001524` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001507` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.009495` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.007331` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.006017` (raises CT win probability)
- `lag_12__CT_place_HOLE`: coefficient `0.005589` (raises CT win probability)
- `lag_04__CT_velocity_mean`: coefficient `0.005137` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004708` (raises CT win probability)
- `lag_08__CT_velocity_mean`: coefficient `-0.004505` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.004323` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.004162` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.003774` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `102129`, seconds `32.50`, LSTM delta `+0.4502`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `+0.062397`
- `lag_00__kill_diff_last_3s`: contribution `+0.045709`
- `lag_00__T_kills_last_3s`: contribution `+0.023226`
- `lag_11__CT_place_SHORTSTAIRS`: contribution `+0.014360`
- `lag_00__CT_kills_last_3s`: contribution `+0.013594`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `104689`, seconds `72.50`, LSTM delta `+0.3775`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.030851`
- `lag_00__kill_diff_last_3s`: contribution `+0.022854`
- `lag_09__T_duck_amount_mean`: contribution `+0.018636`
- `lag_00__CT_kills_last_3s`: contribution `+0.013594`
- `lag_08__CT_velocity_mean`: contribution `+0.013543`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.030851`

### tick `104369`, seconds `67.50`, LSTM delta `-0.3752`

Top all feature movements:
- `lag_12__CT_place_HOLE`: contribution `-0.062397`
- `lag_08__CT_place_HOLE`: contribution `-0.037336`
- `lag_04__CT_velocity_mean`: contribution `-0.025769`
- `lag_00__T_duck_amount_mean`: contribution `-0.025140`
- `lag_00__T_kills_last_3s`: contribution `-0.023226`

Top utility-only movements:
- `lag_01__CT4__molly`: contribution `-0.005648`
- `lag_00__CT4__molly`: contribution `-0.005547`

### tick `102193`, seconds `33.50`, LSTM delta `+0.3377`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.022854`
- `lag_00__damage_diff_last_5s`: contribution `+0.022668`
- `lag_13__CT_place_HOLE`: contribution `+0.013634`
- `lag_00__CT_kills_last_3s`: contribution `+0.013594`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.012066`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101937`, seconds `29.50`, LSTM delta `-0.2670`

Top all feature movements:
- `lag_08__CT_place_HOLE`: contribution `-0.037336`
- `lag_00__T_kills_last_3s`: contribution `-0.023226`
- `lag_00__kill_diff_last_3s`: contribution `-0.022854`
- `lag_00__CT2__is_scoped`: contribution `-0.016435`
- `lag_00__damage_diff_last_5s`: contribution `-0.013574`

Top utility-only movements:
- No utility movement among the top local contributors.
