# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `2817`, seconds `44.00`, LSTM `0.7922`, delta `+0.2296`
- tick `3425`, seconds `53.50`, LSTM `0.9051`, delta `+0.1148`
- tick `4865`, seconds `76.00`, LSTM `0.9543`, delta `+0.0533`
- tick `2497`, seconds `39.00`, LSTM `0.6360`, delta `+0.0421`
- tick `2337`, seconds `36.50`, LSTM `0.5911`, delta `-0.0361`
- tick `3457`, seconds `54.00`, LSTM `0.9401`, delta `+0.0350`
- tick `1729`, seconds `27.00`, LSTM `0.5511`, delta `+0.0337`
- tick `1793`, seconds `28.00`, LSTM `0.5961`, delta `+0.0333`
- tick `3009`, seconds `47.00`, LSTM `0.8135`, delta `-0.0318`
- tick `2593`, seconds `40.50`, LSTM `0.6056`, delta `-0.0261`

## Top 15 local ridge features

- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.001765`, |coef| `0.001765`
- `lag_01__CT_place_OUTSIDELONG`: coefficient `0.001731`, |coef| `0.001731`
- `lag_00__CT_kills_last_3s`: coefficient `0.001638`, |coef| `0.001638`
- `lag_02__T_place_LOWERTUNNEL`: coefficient `-0.001633`, |coef| `0.001633`
- `lag_15__CT_place_EXTENDEDA`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_00__kill_diff_last_3s`: coefficient `0.001366`, |coef| `0.001366`
- `lag_00__CT_damage_last_5s`: coefficient `0.001344`, |coef| `0.001344`
- `lag_00__damage_diff_last_5s`: coefficient `0.001330`, |coef| `0.001330`
- `lag_15__T_place_LOWERTUNNEL`: coefficient `0.001297`, |coef| `0.001297`
- `lag_00__T2__duck_amount`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_07__CT_place_LONGDOORS`: coefficient `0.001230`, |coef| `0.001230`
- `lag_05__T_place_LOWERTUNNEL`: coefficient `0.001109`, |coef| `0.001109`
- `lag_00__T1__alive`: coefficient `-0.001043`, |coef| `0.001043`
- `lag_00__T1__hp`: coefficient `-0.001027`, |coef| `0.001027`
- `lag_08__T5__duck_amount`: coefficient `0.001020`, |coef| `0.001020`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `-0.000673` (lowers CT win probability)
- `lag_00__T_smokes_last_5s`: coefficient `-0.000571` (lowers CT win probability)
- `lag_11__T1__smoke`: coefficient `-0.000461` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000428` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000419` (lowers CT win probability)
- `lag_01__T1__smoke`: coefficient `-0.000392` (lowers CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.000390` (lowers CT win probability)
- `lag_11__CT_smokes_last_5s`: coefficient `-0.000384` (lowers CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.000381` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000353` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_TUNNELSTAIRS`: coefficient `-0.001765` (lowers CT win probability)
- `lag_01__CT_place_OUTSIDELONG`: coefficient `0.001731` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001638` (raises CT win probability)
- `lag_02__T_place_LOWERTUNNEL`: coefficient `-0.001633` (lowers CT win probability)
- `lag_15__CT_place_EXTENDEDA`: coefficient `-0.001506` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001366` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001344` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001330` (raises CT win probability)
- `lag_15__T_place_LOWERTUNNEL`: coefficient `0.001297` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001233` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `2817`, seconds `44.00`, LSTM delta `+0.2296`

Top all feature movements:
- `lag_01__CT_place_OUTSIDELONG`: contribution `+0.017558`
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.012320`
- `lag_15__CT_place_EXTENDEDA`: contribution `+0.008456`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `+0.007082`
- `lag_02__T_place_LOWERTUNNEL`: contribution `+0.007060`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3425`, seconds `53.50`, LSTM delta `+0.1148`

Top all feature movements:
- `lag_07__CT_place_LONGDOORS`: contribution `+0.005385`
- `lag_00__CT_kills_last_3s`: contribution `+0.004730`
- `lag_11__CT_place_OUTSIDELONG`: contribution `+0.003842`
- `lag_00__kill_diff_last_3s`: contribution `+0.003287`
- `lag_04__CT_place_BDOORS`: contribution `+0.003180`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `4865`, seconds `76.00`, LSTM delta `+0.0533`

Top all feature movements:
- `lag_15__CT_place_TUNNELSTAIRS`: contribution `+0.007401`
- `lag_07__CT_place_OUTSIDELONG`: contribution `+0.006161`
- `lag_00__CT_kills_last_3s`: contribution `+0.004730`
- `lag_00__kill_diff_last_3s`: contribution `+0.003287`
- `lag_15__CT_place_LOWERTUNNEL`: contribution `+0.002554`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `-0.001155`

### tick `2497`, seconds `39.00`, LSTM delta `+0.0421`

Top all feature movements:
- `lag_05__T_place_TUNNELSTAIRS`: contribution `+0.012320`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.007082`
- `lag_05__CT_place_EXTENDEDA`: contribution `+0.004819`
- `lag_05__T_place_LOWERTUNNEL`: contribution `+0.004796`
- `lag_06__CT1__duck_amount`: contribution `-0.003128`

Top utility-only movements:
- `lag_11__T1__smoke`: contribution `+0.000996`

### tick `2337`, seconds `36.50`, LSTM delta `-0.0361`

Top all feature movements:
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.008456`
- `lag_10__T_place_TUNNELSTAIRS`: contribution `+0.007042`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.004771`
- `lag_11__T_place_TUNNELSTAIRS`: contribution `+0.004260`
- `lag_12__CT_place_EXTENDEDA`: contribution `-0.002723`

Top utility-only movements:
- No utility movement among the top local contributors.
