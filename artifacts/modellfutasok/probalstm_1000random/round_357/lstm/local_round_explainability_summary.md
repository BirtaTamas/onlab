# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `9`

## Largest probability jumps

- tick `66329`, seconds `58.50`, LSTM `0.2384`, delta `-0.2067`
- tick `64281`, seconds `26.50`, LSTM `0.5244`, delta `+0.1812`
- tick `63929`, seconds `21.00`, LSTM `0.4105`, delta `-0.1014`
- tick `66713`, seconds `64.50`, LSTM `0.0184`, delta `-0.1010`
- tick `65561`, seconds `46.50`, LSTM `0.4917`, delta `-0.0707`
- tick `65593`, seconds `47.00`, LSTM `0.5624`, delta `+0.0707`
- tick `63545`, seconds `15.00`, LSTM `0.5587`, delta `+0.0653`
- tick `63097`, seconds `8.00`, LSTM `0.3410`, delta `-0.0613`
- tick `66361`, seconds `59.00`, LSTM `0.1881`, delta `-0.0504`
- tick `63961`, seconds `21.50`, LSTM `0.3603`, delta `-0.0502`

## Top 15 local ridge features

- `lag_04__CT_place_SHORTSTAIRS`: coefficient `0.003464`, |coef| `0.003464`
- `lag_07__CT3__is_scoped`: coefficient `-0.002333`, |coef| `0.002333`
- `lag_00__kill_diff_last_3s`: coefficient `0.002183`, |coef| `0.002183`
- `lag_00__T_kills_last_3s`: coefficient `-0.002173`, |coef| `0.002173`
- `lag_04__CT_place_CATWALK`: coefficient `-0.001891`, |coef| `0.001891`
- `lag_12__CT2__duck_amount`: coefficient `0.001888`, |coef| `0.001888`
- `lag_03__CT_place_CATWALK`: coefficient `-0.001772`, |coef| `0.001772`
- `lag_15__T_B_site_active_infernos`: coefficient `-0.001751`, |coef| `0.001751`
- `lag_00__CT3__duck_amount`: coefficient `0.001729`, |coef| `0.001729`
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001693`, |coef| `0.001693`
- `lag_00__CT3__alive`: coefficient `0.001570`, |coef| `0.001570`
- `lag_10__CT4__is_scoped`: coefficient `0.001539`, |coef| `0.001539`
- `lag_10__CT_scoped_count`: coefficient `0.001527`, |coef| `0.001527`
- `lag_06__CT4__duck_amount`: coefficient `-0.001507`, |coef| `0.001507`
- `lag_03__CT_place_SHORTSTAIRS`: coefficient `0.001505`, |coef| `0.001505`

## Top 10 utility ridge features

- `lag_15__T_B_site_active_infernos`: coefficient `-0.001751` (lowers CT win probability)
- `lag_05__T2__molly`: coefficient `0.001326` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `-0.001271` (lowers CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `-0.001219` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.001071` (lowers CT win probability)
- `lag_10__T1__molly`: coefficient `-0.000926` (lowers CT win probability)
- `lag_15__active_infernos_total`: coefficient `-0.000894` (lowers CT win probability)
- `lag_13__T_he_last_5s`: coefficient `-0.000836` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.000761` (lowers CT win probability)
- `lag_11__T_he_last_5s`: coefficient `-0.000748` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_SHORTSTAIRS`: coefficient `0.003464` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `-0.002333` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002183` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002173` (lowers CT win probability)
- `lag_04__CT_place_CATWALK`: coefficient `-0.001891` (lowers CT win probability)
- `lag_12__CT2__duck_amount`: coefficient `0.001888` (raises CT win probability)
- `lag_03__CT_place_CATWALK`: coefficient `-0.001772` (lowers CT win probability)
- `lag_00__CT3__duck_amount`: coefficient `0.001729` (raises CT win probability)
- `lag_01__T_place_MIDDOORS`: coefficient `-0.001693` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001570` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `66329`, seconds `58.50`, LSTM delta `-0.2067`

Top all feature movements:
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.019312`
- `lag_07__CT3__is_scoped`: contribution `-0.010613`
- `lag_03__CT_place_SHORTSTAIRS`: contribution `-0.008387`
- `lag_04__CT_place_CATWALK`: contribution `-0.007533`
- `lag_12__CT2__duck_amount`: contribution `-0.007192`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.004952`

### tick `64281`, seconds `26.50`, LSTM delta `+0.1812`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.007247`
- `lag_05__CT_place_BDOORS`: contribution `+0.006832`
- `lag_15__CT_place_EXTENDEDA`: contribution `+0.006815`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `+0.006245`
- `lag_06__T_place_OUTSIDETUNNEL`: contribution `+0.006068`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `63929`, seconds `21.00`, LSTM delta `-0.1014`

Top all feature movements:
- `lag_04__CT_place_SHORTSTAIRS`: contribution `-0.019312`
- `lag_00__T_kills_last_3s`: contribution `-0.006885`
- `lag_00__kill_diff_last_3s`: contribution `-0.005254`
- `lag_11__CT_shots_fired_sum`: contribution `-0.004017`
- `lag_12__CT1__is_walking`: contribution `-0.003285`

Top utility-only movements:
- `lag_13__CT_active_infernos`: contribution `-0.002585`

### tick `66713`, seconds `64.50`, LSTM delta `-0.1010`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `-0.016449`
- `lag_06__CT_flashes_last_5s`: contribution `-0.013403`
- `lag_07__CT_place_LOWERTUNNEL`: contribution `-0.008642`
- `lag_00__T_kills_last_3s`: contribution `-0.006885`
- `lag_15__CT_place_SHORTSTAIRS`: contribution `+0.006245`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `-0.013403`

### tick `65561`, seconds `46.50`, LSTM delta `-0.0707`

Top all feature movements:
- `lag_07__CT3__is_scoped`: contribution `+0.010613`
- `lag_03__CT_place_SHORTSTAIRS`: contribution `+0.008387`
- `lag_03__CT_place_EXTENDEDA`: contribution `-0.008131`
- `lag_01__T_place_MIDDOORS`: contribution `-0.007197`
- `lag_02__T4__is_scoped`: contribution `-0.005812`

Top utility-only movements:
- No utility movement among the top local contributors.
