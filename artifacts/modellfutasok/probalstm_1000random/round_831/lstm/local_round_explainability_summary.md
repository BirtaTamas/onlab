# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `14`

## Largest probability jumps

- tick `127467`, seconds `78.50`, LSTM `0.0694`, delta `-0.2813`
- tick `126699`, seconds `66.50`, LSTM `0.4905`, delta `+0.2149`
- tick `126667`, seconds `66.00`, LSTM `0.2756`, delta `+0.1371`
- tick `123467`, seconds `16.00`, LSTM `0.0973`, delta `-0.1108`
- tick `126603`, seconds `65.00`, LSTM `0.1150`, delta `+0.0820`
- tick `127883`, seconds `85.00`, LSTM `0.0147`, delta `-0.0723`
- tick `123403`, seconds `15.00`, LSTM `0.2029`, delta `+0.0591`
- tick `123115`, seconds `10.50`, LSTM `0.3045`, delta `-0.0583`
- tick `123787`, seconds `21.00`, LSTM `0.0264`, delta `-0.0517`
- tick `126795`, seconds `68.00`, LSTM `0.3917`, delta `-0.0467`

## Top 15 local ridge features

- `lag_15__T_place_PIPE`: coefficient `-0.003309`, |coef| `0.003309`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002514`, |coef| `0.002514`
- `lag_03__T_place_PIPE`: coefficient `0.002447`, |coef| `0.002447`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002186`, |coef| `0.002186`
- `lag_04__T_place_PLAYGROUND`: coefficient `0.002104`, |coef| `0.002104`
- `lag_02__CT_shots_fired_sum`: coefficient `0.002034`, |coef| `0.002034`
- `lag_05__T_place_CONNECTOR`: coefficient `-0.002027`, |coef| `0.002027`
- `lag_03__T_place_CONNECTOR`: coefficient `-0.001994`, |coef| `0.001994`
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_00__kill_diff_last_3s`: coefficient `0.001846`, |coef| `0.001846`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001799`, |coef| `0.001799`
- `lag_06__CT_place_CONSTRUCTION`: coefficient `-0.001779`, |coef| `0.001779`
- `lag_00__damage_diff_last_5s`: coefficient `0.001578`, |coef| `0.001578`
- `lag_04__T_place_CONNECTOR`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_00__CT_place_LOWERPARK`: coefficient `0.001535`, |coef| `0.001535`

## Top 10 utility ridge features

- `lag_06__CT5__flash_duration`: coefficient `0.001341` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.001101` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.001048` (raises CT win probability)
- `lag_06__CT1__flash_duration`: coefficient `0.000946` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.000823` (raises CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000716` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000695` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000596` (lowers CT win probability)
- `lag_03__T5__molly`: coefficient `-0.000507` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.000504` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_PIPE`: coefficient `-0.003309` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002514` (raises CT win probability)
- `lag_03__T_place_PIPE`: coefficient `0.002447` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002186` (raises CT win probability)
- `lag_04__T_place_PLAYGROUND`: coefficient `0.002104` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.002034` (raises CT win probability)
- `lag_05__T_place_CONNECTOR`: coefficient `-0.002027` (lowers CT win probability)
- `lag_03__T_place_CONNECTOR`: coefficient `-0.001994` (lowers CT win probability)
- `lag_02__T_place_CONNECTOR`: coefficient `-0.001871` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001846` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `127467`, seconds `78.50`, LSTM delta `-0.2813`

Top all feature movements:
- `lag_15__T_place_PIPE`: contribution `-0.042266`
- `lag_03__T_place_PIPE`: contribution `-0.031252`
- `lag_04__T_place_PLAYGROUND`: contribution `-0.030899`
- `lag_06__CT_place_CONSTRUCTION`: contribution `-0.022380`
- `lag_10__CT_place_CONSTRUCTION`: contribution `-0.017951`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126699`, seconds `66.50`, LSTM delta `+0.2149`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.013671`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010479`
- `lag_05__T_place_CONNECTOR`: contribution `+0.009817`
- `lag_03__T_place_CONNECTOR`: contribution `+0.009655`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008480`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.006554`
- `lag_06__CT_flash_duration_sum`: contribution `+0.003992`
- `lag_06__CT1__flash_duration`: contribution `+0.003273`

### tick `126667`, seconds `66.00`, LSTM delta `+0.1371`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.015719`
- `lag_01__CT_shots_fired_sum`: contribution `+0.009114`
- `lag_02__T_place_CONNECTOR`: contribution `+0.009059`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008480`
- `lag_04__T_place_CONNECTOR`: contribution `+0.007445`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `+0.005381`
- `lag_05__CT_flash_duration_sum`: contribution `+0.003136`
- `lag_05__CT1__flash_duration`: contribution `+0.002404`

### tick `123467`, seconds `16.00`, LSTM delta `-0.1108`

Top all feature movements:
- `lag_03__T_place_CONNECTOR`: contribution `-0.009655`
- `lag_02__T_shots_fired_sum`: contribution `-0.008019`
- `lag_02__T3__shots_fired`: contribution `-0.006226`
- `lag_07__CT_place_BRIDGE`: contribution `-0.004551`
- `lag_00__kill_diff_last_3s`: contribution `-0.004443`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `126603`, seconds `65.00`, LSTM delta `+0.0820`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010479`
- `lag_02__T_place_CONNECTOR`: contribution `+0.009059`
- `lag_00__T_place_CONNECTOR`: contribution `+0.008714`
- `lag_00__kill_diff_last_3s`: contribution `+0.004443`
- `lag_00__T2__shots_fired`: contribution `+0.003900`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `+0.003497`
- `lag_03__CT_flash_duration_sum`: contribution `+0.001807`
- `lag_03__CT1__flash_duration`: contribution `+0.001099`
