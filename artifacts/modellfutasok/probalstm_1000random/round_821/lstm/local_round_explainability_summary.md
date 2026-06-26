# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `106766`, seconds `13.50`, LSTM `0.2112`, delta `-0.2914`
- tick `106798`, seconds `14.00`, LSTM `0.1105`, delta `-0.1007`
- tick `107630`, seconds `27.00`, LSTM `0.0087`, delta `-0.0292`
- tick `106830`, seconds `14.50`, LSTM `0.0817`, delta `-0.0288`
- tick `107598`, seconds `26.50`, LSTM `0.0380`, delta `+0.0271`
- tick `107246`, seconds `21.00`, LSTM `0.0219`, delta `-0.0140`
- tick `106862`, seconds `15.00`, LSTM `0.0683`, delta `-0.0134`
- tick `107278`, seconds `21.50`, LSTM `0.0132`, delta `-0.0087`
- tick `106926`, seconds `16.00`, LSTM `0.0593`, delta `-0.0084`
- tick `107086`, seconds `18.50`, LSTM `0.0389`, delta `-0.0076`

## Top 15 local ridge features

- `lag_05__CT_place_HUTROOF`: coefficient `-0.001958`, |coef| `0.001958`
- `lag_00__T_place_TROPHY`: coefficient `0.001832`, |coef| `0.001832`
- `lag_11__CT_place_GARAGE`: coefficient `-0.001795`, |coef| `0.001795`
- `lag_03__T_place_CONTROL`: coefficient `-0.001683`, |coef| `0.001683`
- `lag_03__T_place_VENDING`: coefficient `0.001632`, |coef| `0.001632`
- `lag_02__CT_place_RAFTERS`: coefficient `0.001515`, |coef| `0.001515`
- `lag_02__T_place_VENDING`: coefficient `-0.001514`, |coef| `0.001514`
- `lag_00__T_place_RAMP`: coefficient `-0.001405`, |coef| `0.001405`
- `lag_11__CT_place_RAFTERS`: coefficient `-0.001400`, |coef| `0.001400`
- `lag_02__T_place_TROPHY`: coefficient `0.001397`, |coef| `0.001397`
- `lag_05__CT_place_RAFTERS`: coefficient `0.001333`, |coef| `0.001333`
- `lag_14__CT_place_HELL`: coefficient `0.001254`, |coef| `0.001254`
- `lag_14__CT_place_ADMIN`: coefficient `-0.001240`, |coef| `0.001240`
- `lag_06__T_place_TROPHY`: coefficient `-0.001237`, |coef| `0.001237`
- `lag_00__T_place_CONTROL`: coefficient `-0.001228`, |coef| `0.001228`

## Top 10 utility ridge features

- `lag_00__CT2__smoke`: coefficient `0.000871` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.000550` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000493` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.000475` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `-0.000445` (lowers CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000443` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.000394` (lowers CT win probability)
- `lag_08__CT4__flash_duration`: coefficient `-0.000386` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000378` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.000353` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_HUTROOF`: coefficient `-0.001958` (lowers CT win probability)
- `lag_00__T_place_TROPHY`: coefficient `0.001832` (raises CT win probability)
- `lag_11__CT_place_GARAGE`: coefficient `-0.001795` (lowers CT win probability)
- `lag_03__T_place_CONTROL`: coefficient `-0.001683` (lowers CT win probability)
- `lag_03__T_place_VENDING`: coefficient `0.001632` (raises CT win probability)
- `lag_02__CT_place_RAFTERS`: coefficient `0.001515` (raises CT win probability)
- `lag_02__T_place_VENDING`: coefficient `-0.001514` (lowers CT win probability)
- `lag_00__T_place_RAMP`: coefficient `-0.001405` (lowers CT win probability)
- `lag_11__CT_place_RAFTERS`: coefficient `-0.001400` (lowers CT win probability)
- `lag_02__T_place_TROPHY`: coefficient `0.001397` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `106766`, seconds `13.50`, LSTM delta `-0.2914`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `-0.023233`
- `lag_03__T_place_VENDING`: contribution `-0.016549`
- `lag_05__CT_place_HUTROOF`: contribution `-0.013704`
- `lag_11__CT_place_GARAGE`: contribution `-0.012902`
- `lag_03__T_place_CONTROL`: contribution `-0.011963`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106798`, seconds `14.00`, LSTM delta `-0.1007`

Top all feature movements:
- `lag_03__T_place_CONTROL`: contribution `-0.011963`
- `lag_04__T_place_VENDING`: contribution `-0.011214`
- `lag_02__T_place_TROPHY`: contribution `-0.008862`
- `lag_01__T_place_CONTROL`: contribution `-0.008319`
- `lag_06__T_place_TROPHY`: contribution `-0.007844`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107630`, seconds `27.00`, LSTM delta `-0.0292`

Top all feature movements:
- `lag_14__CT_place_DECON`: contribution `-0.010732`
- `lag_04__CT_place_DECON`: contribution `+0.006235`
- `lag_06__CT_place_ADMIN`: contribution `-0.004301`
- `lag_04__CT_place_ADMIN`: contribution `-0.003968`
- `lag_03__T_place_RAMP`: contribution `+0.003160`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106830`, seconds `14.50`, LSTM delta `-0.0288`

Top all feature movements:
- `lag_02__T_place_TROPHY`: contribution `-0.017724`
- `lag_03__T_place_CONTROL`: contribution `-0.011963`
- `lag_06__T_place_TROPHY`: contribution `-0.007844`
- `lag_03__T_place_TROPHY`: contribution `+0.007470`
- `lag_14__CT_place_HELL`: contribution `-0.006799`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `107598`, seconds `26.50`, LSTM delta `+0.0271`

Top all feature movements:
- `lag_03__CT_place_DECON`: contribution `+0.012687`
- `lag_12__T_place_RAMP`: contribution `+0.003376`
- `lag_03__T_place_RAMP`: contribution `+0.003160`
- `lag_09__T_place_RAMP`: contribution `+0.003125`
- `lag_05__CT_place_ADMIN`: contribution `-0.003033`

Top utility-only movements:
- No utility movement among the top local contributors.
