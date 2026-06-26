# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `19`

## Largest probability jumps

- tick `186297`, seconds `94.50`, LSTM `0.9171`, delta `+0.2300`
- tick `186137`, seconds `92.00`, LSTM `0.8742`, delta `+0.1851`
- tick `186201`, seconds `93.00`, LSTM `0.7278`, delta `-0.1227`
- tick `185657`, seconds `84.50`, LSTM `0.6731`, delta `+0.0895`
- tick `187001`, seconds `105.50`, LSTM `0.9295`, delta `+0.0662`
- tick `186105`, seconds `91.50`, LSTM `0.6891`, delta `+0.0567`
- tick `186969`, seconds `105.00`, LSTM `0.8633`, delta `+0.0410`
- tick `185881`, seconds `88.00`, LSTM `0.6302`, delta `-0.0362`
- tick `186681`, seconds `100.50`, LSTM `0.8538`, delta `-0.0329`
- tick `185977`, seconds `89.50`, LSTM `0.6537`, delta `+0.0322`

## Top 15 local ridge features

- `lag_12__T_place_CONTROL`: coefficient `0.001971`, |coef| `0.001971`
- `lag_11__T_place_TROPHY`: coefficient `0.001771`, |coef| `0.001771`
- `lag_11__T_place_VENDING`: coefficient `-0.001338`, |coef| `0.001338`
- `lag_04__T_place_HUT`: coefficient `0.001302`, |coef| `0.001302`
- `lag_00__CT_kills_last_3s`: coefficient `0.001279`, |coef| `0.001279`
- `lag_05__T4__is_scoped`: coefficient `-0.001247`, |coef| `0.001247`
- `lag_14__T_place_TROPHY`: coefficient `-0.001204`, |coef| `0.001204`
- `lag_00__T_place_HUT`: coefficient `-0.001201`, |coef| `0.001201`
- `lag_14__T_place_CONTROL`: coefficient `0.001178`, |coef| `0.001178`
- `lag_00__kill_diff_last_3s`: coefficient `0.001176`, |coef| `0.001176`
- `lag_02__T_place_RAMP`: coefficient `0.001121`, |coef| `0.001121`
- `lag_13__T4__duck_amount`: coefficient `-0.001090`, |coef| `0.001090`
- `lag_10__T2__flash_duration`: coefficient `0.001049`, |coef| `0.001049`
- `lag_02__CT_place_VENTS`: coefficient `0.001027`, |coef| `0.001027`
- `lag_04__T_place_CONTROL`: coefficient `0.000987`, |coef| `0.000987`

## Top 10 utility ridge features

- `lag_10__T2__flash_duration`: coefficient `0.001049` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.000689` (raises CT win probability)
- `lag_10__T_flash_duration_sum`: coefficient `0.000470` (raises CT win probability)
- `lag_15__CT2__flash_duration`: coefficient `0.000403` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.000394` (raises CT win probability)
- `lag_05__T_flash_duration_sum`: coefficient `0.000382` (raises CT win probability)
- `lag_14__CT1__molly`: coefficient `0.000376` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.000370` (raises CT win probability)
- `lag_00__T5__molly`: coefficient `-0.000370` (lowers CT win probability)
- `lag_09__T2__flash_duration`: coefficient `0.000368` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_CONTROL`: coefficient `0.001971` (raises CT win probability)
- `lag_11__T_place_TROPHY`: coefficient `0.001771` (raises CT win probability)
- `lag_11__T_place_VENDING`: coefficient `-0.001338` (lowers CT win probability)
- `lag_04__T_place_HUT`: coefficient `0.001302` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001279` (raises CT win probability)
- `lag_05__T4__is_scoped`: coefficient `-0.001247` (lowers CT win probability)
- `lag_14__T_place_TROPHY`: coefficient `-0.001204` (lowers CT win probability)
- `lag_00__T_place_HUT`: coefficient `-0.001201` (lowers CT win probability)
- `lag_14__T_place_CONTROL`: coefficient `0.001178` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001176` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `186297`, seconds `94.50`, LSTM delta `+0.2300`

Top all feature movements:
- `lag_12__T_place_CONTROL`: contribution `+0.028007`
- `lag_04__T_place_HUT`: contribution `+0.012132`
- `lag_00__T_place_HUT`: contribution `+0.011192`
- `lag_12__T_place_TROPHY`: contribution `+0.011172`
- `lag_02__CT_place_VENTS`: contribution `+0.008621`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `+0.007816`

### tick `186137`, seconds `92.00`, LSTM delta `+0.1851`

Top all feature movements:
- `lag_11__T_place_TROPHY`: contribution `+0.022467`
- `lag_11__T_place_VENDING`: contribution `+0.013567`
- `lag_15__T_place_VENTS`: contribution `+0.012723`
- `lag_07__T_place_TROPHY`: contribution `+0.011016`
- `lag_07__T_place_CONTROL`: contribution `+0.007229`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.005131`

### tick `186201`, seconds `93.00`, LSTM delta `-0.1227`

Top all feature movements:
- `lag_11__T_place_TROPHY`: contribution `-0.011234`
- `lag_13__T_place_VENDING`: contribution `-0.009851`
- `lag_13__T_place_TROPHY`: contribution `-0.009080`
- `lag_14__T_place_TROPHY`: contribution `-0.007637`
- `lag_04__T_place_CONTROL`: contribution `-0.007015`

Top utility-only movements:
- `lag_07__T2__flash_duration`: contribution `-0.002121`

### tick `185657`, seconds `84.50`, LSTM delta `+0.0895`

Top all feature movements:
- `lag_13__CT_place_OBSERVATION`: contribution `+0.010248`
- `lag_00__T_place_VENTS`: contribution `+0.008016`
- `lag_05__T4__is_scoped`: contribution `+0.005794`
- `lag_00__CT_kills_last_3s`: contribution `+0.003693`
- `lag_12__T_place_SQUEAKY`: contribution `+0.003525`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `187001`, seconds `105.50`, LSTM delta `+0.0662`

Top all feature movements:
- `lag_02__CT_place_HUT`: contribution `+0.008921`
- `lag_13__CT_place_DECON`: contribution `+0.008886`
- `lag_10__CT_place_HUT`: contribution `+0.008073`
- `lag_04__CT_place_VENTS`: contribution `+0.006498`
- `lag_15__CT_place_DECON`: contribution `+0.004277`

Top utility-only movements:
- No utility movement among the top local contributors.
