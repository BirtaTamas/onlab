# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `14110`, seconds `93.50`, LSTM `0.4821`, delta `+0.2644`
- tick `14302`, seconds `96.50`, LSTM `0.2945`, delta `-0.2043`
- tick `13982`, seconds `91.50`, LSTM `0.2606`, delta `-0.1950`
- tick `13694`, seconds `87.00`, LSTM `0.3411`, delta `+0.1433`
- tick `15550`, seconds `116.00`, LSTM `0.0505`, delta `-0.0991`
- tick `12094`, seconds `62.00`, LSTM `0.2931`, delta `+0.0982`
- tick `14430`, seconds `98.50`, LSTM `0.1760`, delta `-0.0830`
- tick `13470`, seconds `83.50`, LSTM `0.2410`, delta `-0.0624`
- tick `14846`, seconds `105.00`, LSTM `0.2445`, delta `-0.0563`
- tick `14814`, seconds `104.50`, LSTM `0.3008`, delta `+0.0554`

## Top 15 local ridge features

- `lag_03__T_place_EXTENDEDA`: coefficient `0.002454`, |coef| `0.002454`
- `lag_14__CT_place_HOLE`: coefficient `-0.002309`, |coef| `0.002309`
- `lag_00__T_place_UNDERA`: coefficient `-0.002270`, |coef| `0.002270`
- `lag_00__kill_diff_last_3s`: coefficient `0.002139`, |coef| `0.002139`
- `lag_01__T_place_ARAMP`: coefficient `0.002002`, |coef| `0.002002`
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001950`, |coef| `0.001950`
- `lag_01__CT_place_HOLE`: coefficient `-0.001663`, |coef| `0.001663`
- `lag_00__T_place_ARAMP`: coefficient `-0.001620`, |coef| `0.001620`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.001586`, |coef| `0.001586`
- `lag_00__bomb_events_last_5s`: coefficient `0.001507`, |coef| `0.001507`
- `lag_04__T_place_EXTENDEDA`: coefficient `0.001443`, |coef| `0.001443`
- `lag_00__damage_diff_last_5s`: coefficient `0.001418`, |coef| `0.001418`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001415`, |coef| `0.001415`
- `lag_00__CT_kills_last_3s`: coefficient `0.001388`, |coef| `0.001388`
- `lag_14__T_flashes_last_5s`: coefficient `-0.001314`, |coef| `0.001314`

## Top 10 utility ridge features

- `lag_14__T_flashes_last_5s`: coefficient `-0.001314` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `0.001008` (raises CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.000940` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000818` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `0.000716` (raises CT win probability)
- `lag_07__CT_flash_alpha_mean`: coefficient `0.000697` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000683` (lowers CT win probability)
- `lag_01__T_smokes_last_5s`: coefficient `0.000675` (raises CT win probability)
- `lag_15__T_flashes_last_5s`: coefficient `-0.000648` (lowers CT win probability)
- `lag_10__T_flashes_last_5s`: coefficient `0.000637` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_EXTENDEDA`: coefficient `0.002454` (raises CT win probability)
- `lag_14__CT_place_HOLE`: coefficient `-0.002309` (lowers CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `-0.002270` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002139` (raises CT win probability)
- `lag_01__T_place_ARAMP`: coefficient `0.002002` (raises CT win probability)
- `lag_00__T_place_EXTENDEDA`: coefficient `-0.001950` (lowers CT win probability)
- `lag_01__CT_place_HOLE`: coefficient `-0.001663` (lowers CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.001620` (lowers CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.001586` (lowers CT win probability)
- `lag_00__bomb_events_last_5s`: coefficient `0.001507` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `14110`, seconds `93.50`, LSTM delta `+0.2644`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `+0.025783`
- `lag_01__T_place_ARAMP`: contribution `+0.018111`
- `lag_00__T_place_ARAMP`: contribution `+0.014662`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.012167`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009665`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14302`, seconds `96.50`, LSTM delta `-0.2043`

Top all feature movements:
- `lag_00__T_place_UNDERA`: contribution `-0.035473`
- `lag_07__T_place_ARAMP`: contribution `-0.011415`
- `lag_00__kill_diff_last_3s`: contribution `-0.010297`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.009665`
- `lag_09__T_shots_fired_sum`: contribution `-0.006293`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13982`, seconds `91.50`, LSTM delta `-0.1950`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `-0.025783`
- `lag_10__CT_place_HOLE`: contribution `-0.012633`
- `lag_03__T_place_EXTENDEDA`: contribution `-0.012167`
- `lag_00__T_shots_fired_sum`: contribution `-0.006363`
- `lag_05__CT_place_BDOORS`: contribution `-0.005678`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `13694`, seconds `87.00`, LSTM delta `+0.1433`

Top all feature movements:
- `lag_01__CT_place_HOLE`: contribution `+0.018563`
- `lag_03__T_place_EXTENDEDA`: contribution `+0.012167`
- `lag_00__T_place_CTSPAWN`: contribution `+0.007564`
- `lag_04__T_place_EXTENDEDA`: contribution `+0.007154`
- `lag_05__CT_place_BDOORS`: contribution `+0.005678`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15550`, seconds `116.00`, LSTM delta `-0.0991`

Top all feature movements:
- `lag_01__T_place_ARAMP`: contribution `-0.018111`
- `lag_08__T_place_ARAMP`: contribution `-0.006799`
- `lag_00__kill_diff_last_3s`: contribution `-0.005148`
- `lag_00__T_shots_fired_sum`: contribution `-0.004242`
- `lag_00__T_kills_last_3s`: contribution `-0.004094`

Top utility-only movements:
- `lag_00__CT_flash_alpha_mean`: contribution `-0.002579`
