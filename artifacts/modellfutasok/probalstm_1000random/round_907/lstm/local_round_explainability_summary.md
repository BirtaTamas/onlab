# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `18`

## Largest probability jumps

- tick `151986`, seconds `80.50`, LSTM `0.5095`, delta `+0.3398`
- tick `152274`, seconds `85.00`, LSTM `0.1854`, delta `-0.3182`
- tick `148882`, seconds `32.00`, LSTM `0.2380`, delta `-0.2645`
- tick `151922`, seconds `79.50`, LSTM `0.2242`, delta `-0.2576`
- tick `149202`, seconds `37.00`, LSTM `0.5107`, delta `+0.1980`
- tick `151442`, seconds `72.00`, LSTM `0.4871`, delta `+0.1758`
- tick `150962`, seconds `64.50`, LSTM `0.3786`, delta `-0.1331`
- tick `147666`, seconds `13.00`, LSTM `0.3440`, delta `+0.1181`
- tick `152306`, seconds `85.50`, LSTM `0.0986`, delta `-0.0868`
- tick `146866`, seconds `0.50`, LSTM `0.1699`, delta `-0.0675`

## Top 15 local ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.004925`, |coef| `0.004925`
- `lag_00__kill_diff_last_3s`: coefficient `0.004808`, |coef| `0.004808`
- `lag_15__T_place_LIBRARY`: coefficient `0.004575`, |coef| `0.004575`
- `lag_00__T_place_LIBRARY`: coefficient `-0.003135`, |coef| `0.003135`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003123`, |coef| `0.003123`
- `lag_00__T_kills_last_3s`: coefficient `-0.003018`, |coef| `0.003018`
- `lag_00__CT_kills_last_3s`: coefficient `0.003017`, |coef| `0.003017`
- `lag_00__CT_damage_last_5s`: coefficient `0.002963`, |coef| `0.002963`
- `lag_09__CT_place_PIT`: coefficient `-0.002503`, |coef| `0.002503`
- `lag_10__T1__duck_amount`: coefficient `-0.002227`, |coef| `0.002227`
- `lag_03__T_place_ARCH`: coefficient `0.002211`, |coef| `0.002211`
- `lag_00__CT_place_ARCH`: coefficient `0.002199`, |coef| `0.002199`
- `lag_07__T2__duck_amount`: coefficient `-0.002164`, |coef| `0.002164`
- `lag_05__T_place_ARCH`: coefficient `0.002145`, |coef| `0.002145`
- `lag_01__T_place_CTSPAWN`: coefficient `0.002143`, |coef| `0.002143`

## Top 10 utility ridge features

- `lag_14__T_B_site_active_infernos`: coefficient `-0.001772` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.001513` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `0.001473` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `-0.001358` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.001318` (lowers CT win probability)
- `lag_13__T_active_infernos`: coefficient `-0.001254` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.001049` (lowers CT win probability)
- `lag_07__CT2__smoke`: coefficient `0.000994` (raises CT win probability)
- `lag_05__T1__smoke`: coefficient `0.000975` (raises CT win probability)
- `lag_14__active_infernos_total`: coefficient `-0.000934` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__damage_diff_last_5s`: coefficient `0.004925` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004808` (raises CT win probability)
- `lag_15__T_place_LIBRARY`: coefficient `0.004575` (raises CT win probability)
- `lag_00__T_place_LIBRARY`: coefficient `-0.003135` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003123` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003018` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003017` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002963` (raises CT win probability)
- `lag_09__CT_place_PIT`: coefficient `-0.002503` (lowers CT win probability)
- `lag_10__T1__duck_amount`: coefficient `-0.002227` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `151986`, seconds `80.50`, LSTM delta `+0.3398`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.011573`
- `lag_00__damage_diff_last_5s`: contribution `+0.011110`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010848`
- `lag_09__CT_place_PIT`: contribution `+0.010775`
- `lag_10__T1__duck_amount`: contribution `+0.008718`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `152274`, seconds `85.00`, LSTM delta `-0.3182`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011573`
- `lag_00__damage_diff_last_5s`: contribution `-0.010999`
- `lag_01__T_place_CTSPAWN`: contribution `-0.010221`
- `lag_00__T_kills_last_3s`: contribution `-0.009561`
- `lag_05__T_kills_last_3s`: contribution `-0.005758`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `148882`, seconds `32.00`, LSTM delta `-0.2645`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.011573`
- `lag_00__damage_diff_last_5s`: contribution `-0.011110`
- `lag_00__T_kills_last_3s`: contribution `-0.009561`
- `lag_00__CT_place_ARCH`: contribution `-0.008973`
- `lag_07__T2__duck_amount`: contribution `-0.007049`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.005009`
- `lag_13__T_B_site_active_infernos`: contribution `-0.004277`
- `lag_00__T_B_site_active_infernos`: contribution `-0.004164`
- `lag_14__T_active_infernos`: contribution `-0.002828`

### tick `151922`, seconds `79.50`, LSTM delta `-0.2576`

Top all feature movements:
- `lag_15__T_place_LIBRARY`: contribution `-0.100817`
- `lag_00__kill_diff_last_3s`: contribution `-0.011573`
- `lag_00__T_kills_last_3s`: contribution `-0.009561`
- `lag_07__CT_place_PIT`: contribution `-0.004511`
- `lag_09__T1__duck_amount`: contribution `-0.004256`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `149202`, seconds `37.00`, LSTM delta `+0.1980`

Top all feature movements:
- `lag_05__T_place_ARCH`: contribution `+0.019952`
- `lag_00__CT_shots_fired_sum`: contribution `+0.017356`
- `lag_00__damage_diff_last_5s`: contribution `+0.014221`
- `lag_00__kill_diff_last_3s`: contribution `+0.011573`
- `lag_00__CT_kills_last_3s`: contribution `+0.008711`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `+0.003725`
- `lag_10__T_B_site_active_infernos`: contribution `+0.002195`
- `lag_15__T_active_infernos`: contribution `+0.002185`
