# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `1`

## Largest probability jumps

- tick `7236`, seconds `44.50`, LSTM `0.1742`, delta `-0.2283`
- tick `7268`, seconds `45.00`, LSTM `0.0669`, delta `-0.1074`
- tick `7204`, seconds `44.00`, LSTM `0.4025`, delta `-0.0612`
- tick `7556`, seconds `49.50`, LSTM `0.0186`, delta `-0.0476`
- tick `7492`, seconds `48.50`, LSTM `0.0561`, delta `+0.0434`
- tick `7428`, seconds `47.50`, LSTM `0.0145`, delta `-0.0350`
- tick `6468`, seconds `32.50`, LSTM `0.5606`, delta `+0.0292`
- tick `6596`, seconds `34.50`, LSTM `0.5424`, delta `-0.0256`
- tick `7332`, seconds `46.00`, LSTM `0.0389`, delta `-0.0219`
- tick `6564`, seconds `34.00`, LSTM `0.5680`, delta `-0.0203`

## Top 15 local ridge features

- `lag_13__T_place_MIDDOORS`: coefficient `-0.002854`, |coef| `0.002854`
- `lag_15__T_place_LOWERTUNNEL`: coefficient `0.002848`, |coef| `0.002848`
- `lag_12__T_place_MIDDOORS`: coefficient `-0.002388`, |coef| `0.002388`
- `lag_00__T_place_BDOORS`: coefficient `-0.001967`, |coef| `0.001967`
- `lag_14__bomb_events_last_5s`: coefficient `0.001761`, |coef| `0.001761`
- `lag_00__CT3__duck_amount`: coefficient `0.001668`, |coef| `0.001668`
- `lag_00__CT_place_HOLE`: coefficient `0.001658`, |coef| `0.001658`
- `lag_14__T_place_MIDDOORS`: coefficient `-0.001611`, |coef| `0.001611`
- `lag_08__CT_place_EXTENDEDA`: coefficient `0.001442`, |coef| `0.001442`
- `lag_08__T_place_MIDDOORS`: coefficient `-0.001430`, |coef| `0.001430`
- `lag_02__CT_place_HOLE`: coefficient `-0.001406`, |coef| `0.001406`
- `lag_00__T_kills_last_3s`: coefficient `-0.001393`, |coef| `0.001393`
- `lag_08__T2__duck_amount`: coefficient `0.001383`, |coef| `0.001383`
- `lag_13__T2__duck_amount`: coefficient `-0.001358`, |coef| `0.001358`
- `lag_11__T_place_MIDDOORS`: coefficient `-0.001355`, |coef| `0.001355`

## Top 10 utility ridge features

- `lag_15__T5__smoke`: coefficient `0.000999` (raises CT win probability)
- `lag_14__T5__smoke`: coefficient `0.000509` (raises CT win probability)
- `lag_15__T5__utility_total`: coefficient `0.000462` (raises CT win probability)
- `lag_15__T_smoke_inv`: coefficient `0.000368` (raises CT win probability)
- `lag_07__T_active_smokes`: coefficient `-0.000346` (lowers CT win probability)
- `lag_07__T_mollies_last_5s`: coefficient `-0.000344` (lowers CT win probability)
- `lag_00__T1__flash`: coefficient `0.000324` (raises CT win probability)
- `lag_03__T_mollies_last_5s`: coefficient `0.000317` (raises CT win probability)
- `lag_14__T5__utility_total`: coefficient `0.000309` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `0.000276` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_MIDDOORS`: coefficient `-0.002854` (lowers CT win probability)
- `lag_15__T_place_LOWERTUNNEL`: coefficient `0.002848` (raises CT win probability)
- `lag_12__T_place_MIDDOORS`: coefficient `-0.002388` (lowers CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.001967` (lowers CT win probability)
- `lag_14__bomb_events_last_5s`: coefficient `0.001761` (raises CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001668` (raises CT win probability)
- `lag_00__CT_place_HOLE`: coefficient `0.001658` (raises CT win probability)
- `lag_14__T_place_MIDDOORS`: coefficient `-0.001611` (lowers CT win probability)
- `lag_08__CT_place_EXTENDEDA`: coefficient `0.001442` (raises CT win probability)
- `lag_08__T_place_MIDDOORS`: coefficient `-0.001430` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `7236`, seconds `44.50`, LSTM delta `-0.2283`

Top all feature movements:
- `lag_15__T_place_LOWERTUNNEL`: contribution `-0.024631`
- `lag_00__CT_place_HOLE`: contribution `-0.018507`
- `lag_01__CT_place_HOLE`: contribution `-0.013180`
- `lag_13__T_place_MIDDOORS`: contribution `-0.012132`
- `lag_12__T_place_MIDDOORS`: contribution `-0.010150`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7268`, seconds `45.00`, LSTM delta `-0.1074`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.024599`
- `lag_02__CT_place_HOLE`: contribution `-0.015692`
- `lag_01__CT_place_HOLE`: contribution `+0.013180`
- `lag_13__T_place_MIDDOORS`: contribution `-0.012132`
- `lag_14__T_place_MIDDOORS`: contribution `-0.006848`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7204`, seconds `44.00`, LSTM delta `-0.0612`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.018507`
- `lag_14__T_place_LOWERTUNNEL`: contribution `-0.011523`
- `lag_12__T_place_MIDDOORS`: contribution `-0.010150`
- `lag_07__CT_place_EXTENDEDA`: contribution `-0.005942`
- `lag_11__T_place_MIDDOORS`: contribution `-0.005759`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7556`, seconds `49.50`, LSTM delta `-0.0476`

Top all feature movements:
- `lag_10__CT_place_HOLE`: contribution `-0.012493`
- `lag_00__T_kills_last_3s`: contribution `-0.004414`
- `lag_07__T_place_MIDDOORS`: contribution `+0.004094`
- `lag_09__T_place_MIDDOORS`: contribution `+0.003999`
- `lag_09__T_place_BDOORS`: contribution `-0.003798`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `7492`, seconds `48.50`, LSTM delta `+0.0434`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.024599`
- `lag_09__CT_place_HOLE`: contribution `+0.011486`
- `lag_03__T_place_BDOORS`: contribution `+0.008507`
- `lag_07__T_place_MIDDOORS`: contribution `+0.004094`
- `lag_02__T3__duck_amount`: contribution `-0.003061`

Top utility-only movements:
- No utility movement among the top local contributors.
