# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `2873`, seconds `25.00`, LSTM `0.4448`, delta `-0.2803`
- tick `4409`, seconds `49.00`, LSTM `0.7345`, delta `+0.2451`
- tick `2521`, seconds `19.50`, LSTM `0.5516`, delta `+0.1707`
- tick `2777`, seconds `23.50`, LSTM `0.6532`, delta `+0.1690`
- tick `2553`, seconds `20.00`, LSTM `0.4238`, delta `-0.1278`
- tick `2105`, seconds `13.00`, LSTM `0.6084`, delta `+0.1167`
- tick `2137`, seconds `13.50`, LSTM `0.4932`, delta `-0.1152`
- tick `2841`, seconds `24.50`, LSTM `0.7251`, delta `+0.0955`
- tick `2425`, seconds `18.00`, LSTM `0.3557`, delta `-0.0700`
- tick `2713`, seconds `22.50`, LSTM `0.4323`, delta `+0.0535`

## Top 15 local ridge features

- `lag_02__T_place_UNDERA`: coefficient `0.005463`, |coef| `0.005463`
- `lag_00__T_place_ARAMP`: coefficient `-0.004198`, |coef| `0.004198`
- `lag_01__T_place_UNDERA`: coefficient `0.003283`, |coef| `0.003283`
- `lag_00__kill_diff_last_3s`: coefficient `0.002737`, |coef| `0.002737`
- `lag_03__T_place_UNDERA`: coefficient `0.002516`, |coef| `0.002516`
- `lag_00__CT_kills_last_3s`: coefficient `0.001916`, |coef| `0.001916`
- `lag_00__T_place_UNDERA`: coefficient `0.001697`, |coef| `0.001697`
- `lag_00__damage_diff_last_5s`: coefficient `0.001627`, |coef| `0.001627`
- `lag_02__T_place_SIDE`: coefficient `-0.001541`, |coef| `0.001541`
- `lag_04__T_place_UNDERA`: coefficient `0.001532`, |coef| `0.001532`
- `lag_00__T_kills_last_3s`: coefficient `-0.001500`, |coef| `0.001500`
- `lag_14__CT_place_HOLE`: coefficient `0.001494`, |coef| `0.001494`
- `lag_14__T_place_ARAMP`: coefficient `-0.001466`, |coef| `0.001466`
- `lag_06__CT3__flash_duration`: coefficient `-0.001410`, |coef| `0.001410`
- `lag_01__T_place_ARAMP`: coefficient `-0.001407`, |coef| `0.001407`

## Top 10 utility ridge features

- `lag_06__CT3__flash_duration`: coefficient `-0.001410` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.000957` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `0.000929` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000924` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.000924` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.000810` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000807` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.000787` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `-0.000767` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.000743` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_place_UNDERA`: coefficient `0.005463` (raises CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.004198` (lowers CT win probability)
- `lag_01__T_place_UNDERA`: coefficient `0.003283` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002737` (raises CT win probability)
- `lag_03__T_place_UNDERA`: coefficient `0.002516` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001916` (raises CT win probability)
- `lag_00__T_place_UNDERA`: coefficient `0.001697` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001627` (raises CT win probability)
- `lag_02__T_place_SIDE`: coefficient `-0.001541` (lowers CT win probability)
- `lag_04__T_place_UNDERA`: coefficient `0.001532` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `2873`, seconds `25.00`, LSTM delta `-0.2803`

Top all feature movements:
- `lag_02__T_place_SIDE`: contribution `-0.029820`
- `lag_14__CT_place_HOLE`: contribution `-0.016677`
- `lag_00__T_place_SIDE`: contribution `-0.015081`
- `lag_06__CT3__flash_duration`: contribution `-0.010263`
- `lag_00__kill_diff_last_3s`: contribution `-0.006588`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.010263`
- `lag_06__T3__flash_duration`: contribution `-0.003133`

### tick `4409`, seconds `49.00`, LSTM delta `+0.2451`

Top all feature movements:
- `lag_02__T_place_UNDERA`: contribution `+0.085364`
- `lag_00__T_place_ARAMP`: contribution `+0.037980`
- `lag_14__T_place_ARAMP`: contribution `+0.013265`
- `lag_00__kill_diff_last_3s`: contribution `+0.006588`
- `lag_13__T_place_ARAMP`: contribution `+0.006229`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `2521`, seconds `19.50`, LSTM delta `+0.1707`

Top all feature movements:
- `lag_05__CT_place_HOLE`: contribution `+0.013829`
- `lag_03__CT_place_HOLE`: contribution `+0.008581`
- `lag_15__T1__flash_duration`: contribution `+0.006865`
- `lag_00__kill_diff_last_3s`: contribution `+0.006588`
- `lag_00__CT_kills_last_3s`: contribution `+0.005533`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.006865`
- `lag_13__T1__flash_duration`: contribution `+0.005313`
- `lag_15__T_flash_duration_sum`: contribution `+0.003575`
- `lag_15__CT5__flash_duration`: contribution `+0.003188`
- `lag_15__T3__flash_duration`: contribution `+0.002972`

### tick `2777`, seconds `23.50`, LSTM delta `+0.1690`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013176`
- `lag_11__CT_place_HOLE`: contribution `-0.009936`
- `lag_13__CT_place_HOLE`: contribution `+0.009125`
- `lag_03__CT3__flash_duration`: contribution `+0.006723`
- `lag_00__CT_kills_last_3s`: contribution `+0.005533`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `+0.006723`
- `lag_03__CT4__flash_duration`: contribution `+0.001957`
- `lag_08__T3__flash_duration`: contribution `+0.001910`

### tick `2553`, seconds `20.00`, LSTM delta `-0.1278`

Top all feature movements:
- `lag_04__CT_place_HOLE`: contribution `-0.009729`
- `lag_00__kill_diff_last_3s`: contribution `-0.006588`
- `lag_14__T1__flash_duration`: contribution `-0.005081`
- `lag_00__T_kills_last_3s`: contribution `-0.004751`
- `lag_10__CT1__duck_amount`: contribution `-0.004202`

Top utility-only movements:
- `lag_14__T1__flash_duration`: contribution `-0.005081`
- `lag_06__T3__flash_duration`: contribution `+0.002566`
- `lag_06__CT3__flash_duration`: contribution `-0.001709`
