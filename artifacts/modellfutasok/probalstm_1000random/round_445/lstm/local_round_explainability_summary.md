# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `87588`, seconds `81.50`, LSTM `0.1235`, delta `-0.1220`
- tick `87876`, seconds `86.00`, LSTM `0.0181`, delta `-0.0975`
- tick `87524`, seconds `80.50`, LSTM `0.3012`, delta `+0.0855`
- tick `87556`, seconds `81.00`, LSTM `0.2454`, delta `-0.0557`
- tick `86020`, seconds `57.00`, LSTM `0.1901`, delta `-0.0515`
- tick `87332`, seconds `77.50`, LSTM `0.2323`, delta `+0.0476`
- tick `82980`, seconds `9.50`, LSTM `0.2294`, delta `+0.0447`
- tick `87268`, seconds `76.50`, LSTM `0.2014`, delta `+0.0404`
- tick `82404`, seconds `0.50`, LSTM `0.2247`, delta `-0.0399`
- tick `82756`, seconds `6.00`, LSTM `0.1673`, delta `+0.0353`

## Top 15 local ridge features

- `lag_05__CT_place_OUTSIDELONG`: coefficient `0.001309`, |coef| `0.001309`
- `lag_00__CT3__is_scoped`: coefficient `0.001207`, |coef| `0.001207`
- `lag_00__CT_place_ARAMP`: coefficient `-0.001200`, |coef| `0.001200`
- `lag_15__T_place_ARAMP`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000867`, |coef| `0.000867`
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000828`, |coef| `0.000828`
- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.000818`, |coef| `0.000818`
- `lag_06__T_place_ARAMP`: coefficient `-0.000812`, |coef| `0.000812`
- `lag_02__T4__is_scoped`: coefficient `-0.000793`, |coef| `0.000793`
- `lag_00__T4__is_scoped`: coefficient `-0.000765`, |coef| `0.000765`
- `lag_09__T_place_EXTENDEDA`: coefficient `-0.000761`, |coef| `0.000761`
- `lag_10__T_place_EXTENDEDA`: coefficient `-0.000756`, |coef| `0.000756`
- `lag_12__CT_place_OUTSIDELONG`: coefficient `-0.000745`, |coef| `0.000745`
- `lag_11__CT3__is_scoped`: coefficient `-0.000713`, |coef| `0.000713`
- `lag_00__T_flashes_last_5s`: coefficient `-0.000689`, |coef| `0.000689`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000867` (lowers CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000828` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000689` (lowers CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `-0.000552` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000503` (raises CT win probability)
- `lag_07__T_utility_damage_last_5s`: coefficient `0.000499` (raises CT win probability)
- `lag_02__CT_flashes_last_5s`: coefficient `-0.000460` (lowers CT win probability)
- `lag_07__T_flashes_last_5s`: coefficient `-0.000456` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000443` (raises CT win probability)
- `lag_08__T_utility_damage_last_5s`: coefficient `0.000408` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_OUTSIDELONG`: coefficient `0.001309` (raises CT win probability)
- `lag_00__CT3__is_scoped`: coefficient `0.001207` (raises CT win probability)
- `lag_00__CT_place_ARAMP`: coefficient `-0.001200` (lowers CT win probability)
- `lag_15__T_place_ARAMP`: coefficient `-0.000874` (lowers CT win probability)
- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.000818` (lowers CT win probability)
- `lag_06__T_place_ARAMP`: coefficient `-0.000812` (lowers CT win probability)
- `lag_02__T4__is_scoped`: coefficient `-0.000793` (lowers CT win probability)
- `lag_00__T4__is_scoped`: coefficient `-0.000765` (lowers CT win probability)
- `lag_09__T_place_EXTENDEDA`: coefficient `-0.000761` (lowers CT win probability)
- `lag_10__T_place_EXTENDEDA`: coefficient `-0.000756` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `87588`, seconds `81.50`, LSTM delta `-0.1220`

Top all feature movements:
- `lag_05__CT_place_OUTSIDELONG`: contribution `-0.013274`
- `lag_13__CT_place_OUTSIDELONG`: contribution `-0.008296`
- `lag_06__T_place_ARAMP`: contribution `-0.007348`
- `lag_09__T_place_ARAMP`: contribution `-0.005460`
- `lag_01__T_place_ARAMP`: contribution `-0.004312`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.002121`
- `lag_10__T_A_site_active_infernos`: contribution `-0.001497`

### tick `87876`, seconds `86.00`, LSTM delta `-0.0975`

Top all feature movements:
- `lag_15__T_place_ARAMP`: contribution `-0.007906`
- `lag_07__CT_place_HOLE`: contribution `-0.005540`
- `lag_03__T_place_ARAMP`: contribution `-0.005374`
- `lag_11__T_place_ARAMP`: contribution `-0.005039`
- `lag_12__CT_place_ARAMP`: contribution `-0.003582`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `87524`, seconds `80.50`, LSTM delta `+0.0855`

Top all feature movements:
- `lag_00__CT3__is_scoped`: contribution `+0.005490`
- `lag_10__CT_place_ARAMP`: contribution `+0.003763`
- `lag_10__T_place_EXTENDEDA`: contribution `+0.003750`
- `lag_01__CT_place_ARAMP`: contribution `+0.003573`
- `lag_00__T4__is_scoped`: contribution `-0.003552`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.001373`

### tick `87556`, seconds `81.00`, LSTM delta `-0.0557`

Top all feature movements:
- `lag_12__CT_place_OUTSIDELONG`: contribution `-0.007560`
- `lag_08__T_place_ARAMP`: contribution `-0.005902`
- `lag_00__CT3__is_scoped`: contribution `-0.005490`
- `lag_01__T_place_ARAMP`: contribution `+0.004312`
- `lag_05__T_place_ARAMP`: contribution `-0.004080`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.001176`

### tick `86020`, seconds `57.00`, LSTM delta `-0.0515`

Top all feature movements:
- `lag_00__CT_place_ARAMP`: contribution `-0.007478`
- `lag_00__CT3__is_scoped`: contribution `-0.005490`
- `lag_10__CT_place_ARAMP`: contribution `-0.003763`
- `lag_06__CT_place_ARAMP`: contribution `-0.002412`
- `lag_15__T_place_TUNNELSTAIRS`: contribution `-0.002333`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.000623`
