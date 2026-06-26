# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `56589`, seconds `94.00`, LSTM `0.7968`, delta `+0.2459`
- tick `57421`, seconds `107.00`, LSTM `0.7993`, delta `+0.2315`
- tick `56621`, seconds `94.50`, LSTM `0.5793`, delta `-0.2174`
- tick `56781`, seconds `97.00`, LSTM `0.8054`, delta `+0.2104`
- tick `57069`, seconds `101.50`, LSTM `0.7387`, delta `-0.1429`
- tick `57229`, seconds `104.00`, LSTM `0.5950`, delta `-0.0996`
- tick `56813`, seconds `97.50`, LSTM `0.8658`, delta `+0.0604`
- tick `57325`, seconds `105.50`, LSTM `0.6165`, delta `+0.0542`
- tick `57101`, seconds `102.00`, LSTM `0.6918`, delta `-0.0469`
- tick `57293`, seconds `105.00`, LSTM `0.5623`, delta `-0.0438`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002827`, |coef| `0.002827`
- `lag_00__kill_diff_last_3s`: coefficient `0.002715`, |coef| `0.002715`
- `lag_03__T_place_BALCONY`: coefficient `0.001957`, |coef| `0.001957`
- `lag_00__damage_diff_last_5s`: coefficient `0.001892`, |coef| `0.001892`
- `lag_03__T_place_ARCH`: coefficient `0.001777`, |coef| `0.001777`
- `lag_00__T_kills_last_3s`: coefficient `-0.001713`, |coef| `0.001713`
- `lag_14__T_place_ARCH`: coefficient `0.001707`, |coef| `0.001707`
- `lag_00__CT_kills_last_3s`: coefficient `0.001696`, |coef| `0.001696`
- `lag_00__T_place_ARCH`: coefficient `-0.001686`, |coef| `0.001686`
- `lag_00__CT5__shots_fired`: coefficient `0.001658`, |coef| `0.001658`
- `lag_03__CT1__is_scoped`: coefficient `-0.001621`, |coef| `0.001621`
- `lag_00__T_place_BALCONY`: coefficient `-0.001595`, |coef| `0.001595`
- `lag_00__CT_damage_last_5s`: coefficient `0.001586`, |coef| `0.001586`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001512`, |coef| `0.001512`
- `lag_08__CT_place_LIBRARY`: coefficient `-0.001310`, |coef| `0.001310`

## Top 10 utility ridge features

- `lag_00__CT2__molly`: coefficient `0.000560` (raises CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `-0.000531` (lowers CT win probability)
- `lag_09__CT_B_site_active_smokes`: coefficient `-0.000410` (lowers CT win probability)
- `lag_06__CT2__flash`: coefficient `-0.000403` (lowers CT win probability)
- `lag_13__CT_A_site_active_smokes`: coefficient `-0.000386` (lowers CT win probability)
- `lag_06__CT2__molly`: coefficient `-0.000339` (lowers CT win probability)
- `lag_06__CT2__utility_total`: coefficient `-0.000321` (lowers CT win probability)
- `lag_05__CT5__flash`: coefficient `-0.000314` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000312` (raises CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `-0.000309` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002827` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002715` (raises CT win probability)
- `lag_03__T_place_BALCONY`: coefficient `0.001957` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001892` (raises CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.001777` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001713` (lowers CT win probability)
- `lag_14__T_place_ARCH`: coefficient `0.001707` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001696` (raises CT win probability)
- `lag_00__T_place_ARCH`: coefficient `-0.001686` (lowers CT win probability)
- `lag_00__CT5__shots_fired`: coefficient `0.001658` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `56589`, seconds `94.00`, LSTM delta `+0.2459`

Top all feature movements:
- `lag_03__T_place_ARCH`: contribution `+0.016534`
- `lag_00__T_place_ARCH`: contribution `+0.015684`
- `lag_00__kill_diff_last_3s`: contribution `+0.013072`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009819`
- `lag_00__CT_kills_last_3s`: contribution `+0.009791`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `57421`, seconds `107.00`, LSTM delta `+0.2315`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.013746`
- `lag_00__kill_diff_last_3s`: contribution `+0.013072`
- `lag_00__CT_duck_amount_mean`: contribution `+0.009053`
- `lag_14__CT_place_LIBRARY`: contribution `+0.007740`
- `lag_11__CT_place_LIBRARY`: contribution `+0.007221`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56621`, seconds `94.50`, LSTM delta `-0.2174`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.029456`
- `lag_00__CT5__shots_fired`: contribution `-0.013149`
- `lag_04__T_place_ARCH`: contribution `-0.009040`
- `lag_01__T_place_ARCH`: contribution `-0.008756`
- `lag_03__CT1__is_scoped`: contribution `-0.006942`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `56781`, seconds `97.00`, LSTM delta `+0.2104`

Top all feature movements:
- `lag_03__T_place_BALCONY`: contribution `+0.026905`
- `lag_00__T_place_BALCONY`: contribution `+0.021933`
- `lag_14__T_place_ARCH`: contribution `+0.015880`
- `lag_05__CT_shots_fired_sum`: contribution `+0.010158`
- `lag_05__CT5__shots_fired`: contribution `+0.007388`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `57069`, seconds `101.50`, LSTM delta `-0.1429`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `-0.014682`
- `lag_09__T_place_BALCONY`: contribution `-0.013570`
- `lag_15__T_place_ARCH`: contribution `-0.011472`
- `lag_08__CT_place_LIBRARY`: contribution `-0.008399`
- `lag_14__CT_shots_fired_sum`: contribution `-0.007295`

Top utility-only movements:
- No utility movement among the top local contributors.
