# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `46013`, seconds `56.00`, LSTM `0.9421`, delta `+0.0357`
- tick `45501`, seconds `48.00`, LSTM `0.8384`, delta `-0.0257`
- tick `44093`, seconds `26.00`, LSTM `0.8504`, delta `+0.0250`
- tick `45885`, seconds `54.00`, LSTM `0.8954`, delta `+0.0202`
- tick `44829`, seconds `37.50`, LSTM `0.8535`, delta `-0.0198`
- tick `45245`, seconds `44.00`, LSTM `0.8719`, delta `+0.0186`
- tick `44285`, seconds `29.00`, LSTM `0.8895`, delta `+0.0184`
- tick `45181`, seconds `43.00`, LSTM `0.8413`, delta `-0.0184`
- tick `44381`, seconds `30.50`, LSTM `0.8614`, delta `-0.0183`
- tick `43997`, seconds `24.50`, LSTM `0.8427`, delta `-0.0178`

## Top 15 local ridge features

- `lag_15__CT_place_DECON`: coefficient `-0.000597`, |coef| `0.000597`
- `lag_00__CT2__is_scoped`: coefficient `-0.000593`, |coef| `0.000593`
- `lag_00__CT3__is_walking`: coefficient `-0.000450`, |coef| `0.000450`
- `lag_00__T5__is_walking`: coefficient `-0.000405`, |coef| `0.000405`
- `lag_05__T2__is_walking`: coefficient `-0.000373`, |coef| `0.000373`
- `lag_08__CT5__duck_amount`: coefficient `0.000369`, |coef| `0.000369`
- `lag_01__CT_place_DECON`: coefficient `-0.000345`, |coef| `0.000345`
- `lag_15__T_place_CONTROL`: coefficient `0.000334`, |coef| `0.000334`
- `lag_10__CT5__duck_amount`: coefficient `0.000326`, |coef| `0.000326`
- `lag_02__CT_place_DECON`: coefficient `-0.000323`, |coef| `0.000323`
- `lag_01__T_place_CONTROL`: coefficient `-0.000321`, |coef| `0.000321`
- `lag_04__CT_flashes_last_5s`: coefficient `0.000319`, |coef| `0.000319`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000319`, |coef| `0.000319`
- `lag_13__T_place_ROOF`: coefficient `-0.000318`, |coef| `0.000318`
- `lag_05__T_place_VENDING`: coefficient `-0.000314`, |coef| `0.000314`

## Top 10 utility ridge features

- `lag_04__CT_flashes_last_5s`: coefficient `0.000319` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000308` (raises CT win probability)
- `lag_13__T_smokes_last_5s`: coefficient `0.000271` (raises CT win probability)
- `lag_12__T_smokes_last_5s`: coefficient `-0.000230` (lowers CT win probability)
- `lag_02__T_smokes_last_5s`: coefficient `0.000227` (raises CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `-0.000160` (lowers CT win probability)
- `lag_11__T_smokes_last_5s`: coefficient `-0.000153` (lowers CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.000146` (raises CT win probability)
- `lag_05__CT_flashes_last_5s`: coefficient `0.000144` (raises CT win probability)
- `lag_13__CT_active_infernos`: coefficient `-0.000140` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_DECON`: coefficient `-0.000597` (lowers CT win probability)
- `lag_00__CT2__is_scoped`: coefficient `-0.000593` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000450` (lowers CT win probability)
- `lag_00__T5__is_walking`: coefficient `-0.000405` (lowers CT win probability)
- `lag_05__T2__is_walking`: coefficient `-0.000373` (lowers CT win probability)
- `lag_08__CT5__duck_amount`: coefficient `0.000369` (raises CT win probability)
- `lag_01__CT_place_DECON`: coefficient `-0.000345` (lowers CT win probability)
- `lag_15__T_place_CONTROL`: coefficient `0.000334` (raises CT win probability)
- `lag_10__CT5__duck_amount`: coefficient `0.000326` (raises CT win probability)
- `lag_02__CT_place_DECON`: coefficient `-0.000323` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `46013`, seconds `56.00`, LSTM delta `+0.0357`

Top all feature movements:
- `lag_15__CT_place_DECON`: contribution `+0.009496`
- `lag_15__T_place_CONTROL`: contribution `+0.002376`
- `lag_01__T_place_CONTROL`: contribution `+0.002281`
- `lag_05__T_place_CONTROL`: contribution `+0.001933`
- `lag_04__T_place_CONTROL`: contribution `+0.001810`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45501`, seconds `48.00`, LSTM delta `-0.0257`

Top all feature movements:
- `lag_12__CT_place_DECON`: contribution `-0.004393`
- `lag_00__CT2__is_scoped`: contribution `-0.003632`
- `lag_08__CT2__is_scoped`: contribution `-0.001584`
- `lag_08__CT5__duck_amount`: contribution `-0.001391`
- `lag_10__CT5__duck_amount`: contribution `-0.001035`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44093`, seconds `26.00`, LSTM delta `+0.0250`

Top all feature movements:
- `lag_13__CT_place_LOCKERROOM`: contribution `+0.002284`
- `lag_00__T_place_TROPHY`: contribution `+0.001982`
- `lag_00__CT_place_VENTS`: contribution `+0.001810`
- `lag_08__CT5__duck_amount`: contribution `+0.001391`
- `lag_04__T_place_SILO`: contribution `+0.001304`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `45885`, seconds `54.00`, LSTM delta `+0.0202`

Top all feature movements:
- `lag_11__CT_place_DECON`: contribution `+0.003330`
- `lag_01__T_place_CONTROL`: contribution `-0.002281`
- `lag_00__T_place_TROPHY`: contribution `+0.001982`
- `lag_11__T_place_TROPHY`: contribution `+0.001908`
- `lag_11__T_place_CONTROL`: contribution `+0.001789`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44829`, seconds `37.50`, LSTM delta `-0.0198`

Top all feature movements:
- `lag_00__CT2__is_scoped`: contribution `-0.003632`
- `lag_09__CT_place_HELL`: contribution `-0.001497`
- `lag_09__CT_place_ADMIN`: contribution `-0.001471`
- `lag_11__T2__duck_amount`: contribution `-0.001011`
- `lag_01__T4__duck_amount`: contribution `-0.000992`

Top utility-only movements:
- No utility movement among the top local contributors.
