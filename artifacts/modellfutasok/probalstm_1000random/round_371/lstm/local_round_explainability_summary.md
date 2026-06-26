# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `26`

## Largest probability jumps

- tick `228383`, seconds `14.00`, LSTM `0.7899`, delta `+0.2196`
- tick `228575`, seconds `17.00`, LSTM `0.9158`, delta `+0.0448`
- tick `228415`, seconds `14.50`, LSTM `0.8320`, delta `+0.0421`
- tick `228543`, seconds `16.50`, LSTM `0.8711`, delta `+0.0283`
- tick `232063`, seconds `71.50`, LSTM `0.9711`, delta `+0.0252`
- tick `231615`, seconds `64.50`, LSTM `0.9046`, delta `-0.0196`
- tick `231551`, seconds `63.50`, LSTM `0.9315`, delta `+0.0187`
- tick `227839`, seconds `5.50`, LSTM `0.5865`, delta `+0.0186`
- tick `227999`, seconds `8.00`, LSTM `0.6024`, delta `+0.0175`
- tick `227871`, seconds `6.00`, LSTM `0.6023`, delta `+0.0158`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001576`, |coef| `0.001576`
- `lag_00__kill_diff_last_3s`: coefficient `0.001484`, |coef| `0.001484`
- `lag_00__damage_diff_last_5s`: coefficient `0.001459`, |coef| `0.001459`
- `lag_14__T_place_LOWERMID`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_00__CT_damage_last_5s`: coefficient `0.001348`, |coef| `0.001348`
- `lag_07__T2__duck_amount`: coefficient `-0.001242`, |coef| `0.001242`
- `lag_12__CT3__is_scoped`: coefficient `0.001160`, |coef| `0.001160`
- `lag_05__CT3__is_scoped`: coefficient `-0.001091`, |coef| `0.001091`
- `lag_00__T_place_SECONDMID`: coefficient `-0.001049`, |coef| `0.001049`
- `lag_03__CT_place_BANANA`: coefficient `0.000993`, |coef| `0.000993`
- `lag_15__T_place_SECONDMID`: coefficient `0.000992`, |coef| `0.000992`
- `lag_09__CT_place_BANANA`: coefficient `0.000962`, |coef| `0.000962`
- `lag_14__T_place_SECONDMID`: coefficient `0.000950`, |coef| `0.000950`
- `lag_08__T2__duck_amount`: coefficient `0.000946`, |coef| `0.000946`
- `lag_05__CT_burning_players`: coefficient `0.000939`, |coef| `0.000939`

## Top 10 utility ridge features

- `lag_00__T1__utility_total`: coefficient `-0.000875` (lowers CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000873` (raises CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000824` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000803` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000794` (raises CT win probability)
- `lag_14__T5__smoke`: coefficient `0.000780` (raises CT win probability)
- `lag_06__active_infernos_total`: coefficient `0.000759` (raises CT win probability)
- `lag_10__CT1__molly`: coefficient `-0.000750` (lowers CT win probability)
- `lag_10__T4__molly`: coefficient `-0.000676` (lowers CT win probability)
- `lag_12__T3__molly`: coefficient `-0.000671` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001576` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001484` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001459` (raises CT win probability)
- `lag_14__T_place_LOWERMID`: coefficient `-0.001406` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001348` (raises CT win probability)
- `lag_07__T2__duck_amount`: coefficient `-0.001242` (lowers CT win probability)
- `lag_12__CT3__is_scoped`: coefficient `0.001160` (raises CT win probability)
- `lag_05__CT3__is_scoped`: coefficient `-0.001091` (lowers CT win probability)
- `lag_00__T_place_SECONDMID`: coefficient `-0.001049` (lowers CT win probability)
- `lag_03__CT_place_BANANA`: coefficient `0.000993` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `228383`, seconds `14.00`, LSTM delta `+0.2196`

Top all feature movements:
- `lag_14__T_place_LOWERMID`: contribution `+0.009350`
- `lag_00__CT_kills_last_3s`: contribution `+0.009103`
- `lag_00__kill_diff_last_3s`: contribution `+0.007143`
- `lag_00__damage_diff_last_5s`: contribution `+0.006584`
- `lag_00__CT_damage_last_5s`: contribution `+0.005875`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `228575`, seconds `17.00`, LSTM delta `+0.0448`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `-0.004551`
- `lag_00__kill_diff_last_3s`: contribution `-0.003571`
- `lag_04__CT1__flash_duration`: contribution `+0.003333`
- `lag_09__CT_place_BANANA`: contribution `+0.002848`
- `lag_00__damage_diff_last_5s`: contribution `+0.002667`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.003333`
- `lag_00__T2__utility_total`: contribution `+0.001152`
- `lag_00__T2__flash`: contribution `+0.000927`

### tick `228415`, seconds `14.50`, LSTM delta `+0.0421`

Top all feature movements:
- `lag_15__T_place_LOWERMID`: contribution `+0.004844`
- `lag_14__T_place_LOWERMID`: contribution `+0.004675`
- `lag_08__T2__duck_amount`: contribution `-0.003619`
- `lag_00__T_place_SECONDMID`: contribution `+0.003434`
- `lag_15__T_place_SECONDMID`: contribution `+0.003248`

Top utility-only movements:
- `lag_15__T5__smoke`: contribution `-0.001053`

### tick `228543`, seconds `16.50`, LSTM delta `+0.0283`

Top all feature movements:
- `lag_05__CT3__is_scoped`: contribution `+0.004961`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001720`
- `lag_10__CT3__is_scoped`: contribution `-0.001655`
- `lag_05__CT_scoped_count`: contribution `+0.001177`
- `lag_00__T5__is_scoped`: contribution `+0.001176`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `232063`, seconds `71.50`, LSTM delta `+0.0252`

Top all feature movements:
- `lag_05__CT3__is_scoped`: contribution `+0.004961`
- `lag_00__CT_kills_last_3s`: contribution `+0.004551`
- `lag_00__kill_diff_last_3s`: contribution `+0.003571`
- `lag_03__CT_place_BANANA`: contribution `-0.002940`
- `lag_00__CT_damage_last_5s`: contribution `+0.002938`

Top utility-only movements:
- No utility movement among the top local contributors.
