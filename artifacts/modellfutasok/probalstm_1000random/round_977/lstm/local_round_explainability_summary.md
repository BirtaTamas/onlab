# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `103572`, seconds `70.00`, LSTM `0.9729`, delta `+0.0555`
- tick `103316`, seconds `66.00`, LSTM `0.9033`, delta `-0.0543`
- tick `103348`, seconds `66.50`, LSTM `0.9443`, delta `+0.0411`
- tick `101044`, seconds `30.50`, LSTM `0.8729`, delta `+0.0273`
- tick `103156`, seconds `63.50`, LSTM `0.9551`, delta `+0.0272`
- tick `100372`, seconds `20.00`, LSTM `0.8716`, delta `+0.0265`
- tick `100692`, seconds `25.00`, LSTM `0.8950`, delta `+0.0254`
- tick `100244`, seconds `18.00`, LSTM `0.8601`, delta `-0.0198`
- tick `99380`, seconds `4.50`, LSTM `0.8647`, delta `+0.0174`
- tick `103092`, seconds `62.50`, LSTM `0.9225`, delta `+0.0152`

## Top 15 local ridge features

- `lag_00__T_place_BALCONY`: coefficient `0.000623`, |coef| `0.000623`
- `lag_00__CT3__is_walking`: coefficient `-0.000581`, |coef| `0.000581`
- `lag_03__T_place_BALCONY`: coefficient `0.000550`, |coef| `0.000550`
- `lag_01__T_place_BALCONY`: coefficient `0.000549`, |coef| `0.000549`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000535`, |coef| `0.000535`
- `lag_00__damage_diff_last_5s`: coefficient `0.000527`, |coef| `0.000527`
- `lag_02__T_place_BALCONY`: coefficient `0.000500`, |coef| `0.000500`
- `lag_00__T_walking_count`: coefficient `-0.000480`, |coef| `0.000480`
- `lag_00__T2__is_walking`: coefficient `-0.000467`, |coef| `0.000467`
- `lag_00__CT_place_BALCONY`: coefficient `-0.000446`, |coef| `0.000446`
- `lag_00__T3__duck_amount`: coefficient `-0.000441`, |coef| `0.000441`
- `lag_00__T4__is_walking`: coefficient `-0.000413`, |coef| `0.000413`
- `lag_04__CT_place_PIT`: coefficient `-0.000409`, |coef| `0.000409`
- `lag_05__CT_A_site_active_infernos`: coefficient `-0.000384`, |coef| `0.000384`
- `lag_00__CT_kills_last_3s`: coefficient `0.000369`, |coef| `0.000369`

## Top 10 utility ridge features

- `lag_05__CT_A_site_active_infernos`: coefficient `-0.000384` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `-0.000212` (lowers CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.000205` (lowers CT win probability)
- `lag_08__CT1__molly`: coefficient `-0.000180` (lowers CT win probability)
- `lag_12__CT4__smoke`: coefficient `-0.000180` (lowers CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.000171` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000169` (lowers CT win probability)
- `lag_13__CT_active_infernos`: coefficient `0.000161` (raises CT win probability)
- `lag_05__CT2__flash`: coefficient `-0.000157` (lowers CT win probability)
- `lag_09__CT_A_site_active_infernos`: coefficient `0.000151` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_BALCONY`: coefficient `0.000623` (raises CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000581` (lowers CT win probability)
- `lag_03__T_place_BALCONY`: coefficient `0.000550` (raises CT win probability)
- `lag_01__T_place_BALCONY`: coefficient `0.000549` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000535` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000527` (raises CT win probability)
- `lag_02__T_place_BALCONY`: coefficient `0.000500` (raises CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000480` (lowers CT win probability)
- `lag_00__T2__is_walking`: coefficient `-0.000467` (lowers CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `-0.000446` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `103572`, seconds `70.00`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `+0.002964`
- `lag_00__damage_diff_last_5s`: contribution `+0.002462`
- `lag_15__T_place_BALCONY`: contribution `+0.002409`
- `lag_13__T_place_BALCONY`: contribution `+0.002130`
- `lag_08__CT_place_QUAD`: contribution `+0.001836`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103316`, seconds `66.00`, LSTM delta `-0.0543`

Top all feature movements:
- `lag_03__T_place_BALCONY`: contribution `-0.007559`
- `lag_05__T_place_BALCONY`: contribution `-0.004137`
- `lag_00__CT_shots_fired_sum`: contribution `-0.002602`
- `lag_07__T_place_BALCONY`: contribution `-0.002359`
- `lag_00__CT_place_QUAD`: contribution `-0.002221`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103348`, seconds `66.50`, LSTM delta `+0.0411`

Top all feature movements:
- `lag_05__T_place_BALCONY`: contribution `+0.004137`
- `lag_06__T_place_BALCONY`: contribution `+0.002720`
- `lag_01__CT_place_QUAD`: contribution `+0.002604`
- `lag_00__CT_shots_fired_sum`: contribution `+0.001859`
- `lag_03__CT_shots_fired_sum`: contribution `+0.001295`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101044`, seconds `30.50`, LSTM delta `+0.0273`

Top all feature movements:
- `lag_10__T_place_BALCONY`: contribution `+0.003567`
- `lag_00__CT_place_BALCONY`: contribution `+0.002865`
- `lag_00__T3__duck_amount`: contribution `+0.001662`
- `lag_07__CT_place_BALCONY`: contribution `+0.001578`
- `lag_05__CT_place_ARCH`: contribution `+0.001422`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103156`, seconds `63.50`, LSTM delta `+0.0272`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.008568`
- `lag_02__T_place_BALCONY`: contribution `+0.006871`
- `lag_04__CT_place_PIT`: contribution `+0.001761`
- `lag_00__damage_diff_last_5s`: contribution `+0.001190`
- `lag_00__T2__is_walking`: contribution `+0.001074`

Top utility-only movements:
- `lag_08__CT1__molly`: contribution `+0.000449`
