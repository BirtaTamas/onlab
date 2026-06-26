# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `14`

## Largest probability jumps

- tick `105360`, seconds `96.00`, LSTM `0.2149`, delta `-0.2106`
- tick `105520`, seconds `98.50`, LSTM `0.0309`, delta `-0.0892`
- tick `103088`, seconds `60.50`, LSTM `0.5379`, delta `+0.0873`
- tick `101456`, seconds `35.00`, LSTM `0.3549`, delta `+0.0853`
- tick `108080`, seconds `138.50`, LSTM `0.0228`, delta `-0.0594`
- tick `105296`, seconds `95.00`, LSTM `0.4116`, delta `-0.0565`
- tick `101232`, seconds `31.50`, LSTM `0.2335`, delta `-0.0558`
- tick `107952`, seconds `136.50`, LSTM `0.0688`, delta `+0.0517`
- tick `102288`, seconds `48.00`, LSTM `0.3081`, delta `-0.0469`
- tick `101392`, seconds `34.00`, LSTM `0.2720`, delta `+0.0459`

## Top 15 local ridge features

- `lag_11__T_place_QUAD`: coefficient `-0.002218`, |coef| `0.002218`
- `lag_00__T_place_BALCONY`: coefficient `-0.001732`, |coef| `0.001732`
- `lag_03__T_place_ARCH`: coefficient `-0.001548`, |coef| `0.001548`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001241`, |coef| `0.001241`
- `lag_08__T_place_QUAD`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_07__T2__is_walking`: coefficient `0.001202`, |coef| `0.001202`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001189`, |coef| `0.001189`
- `lag_00__kill_diff_last_3s`: coefficient `0.001189`, |coef| `0.001189`
- `lag_00__CT_place_BALCONY`: coefficient `0.001145`, |coef| `0.001145`
- `lag_06__T_shots_fired_sum`: coefficient `-0.001102`, |coef| `0.001102`
- `lag_12__T_place_QUAD`: coefficient `-0.001099`, |coef| `0.001099`
- `lag_06__T_place_QUAD`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_02__T_place_BALCONY`: coefficient `-0.001017`, |coef| `0.001017`
- `lag_07__T_place_BALCONY`: coefficient `0.001010`, |coef| `0.001010`
- `lag_01__T_place_BALCONY`: coefficient `-0.000980`, |coef| `0.000980`

## Top 10 utility ridge features

- `lag_00__T3__smoke`: coefficient `-0.000663` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000623` (lowers CT win probability)
- `lag_15__T_B_site_active_smokes`: coefficient `0.000611` (raises CT win probability)
- `lag_15__CT_he_last_5s`: coefficient `0.000583` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.000574` (raises CT win probability)
- `lag_11__CT_B_site_active_infernos`: coefficient `-0.000571` (lowers CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `-0.000565` (lowers CT win probability)
- `lag_15__T5__smoke`: coefficient `-0.000564` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000516` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `-0.000511` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__T_place_QUAD`: coefficient `-0.002218` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001732` (lowers CT win probability)
- `lag_03__T_place_ARCH`: coefficient `-0.001548` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001241` (raises CT win probability)
- `lag_08__T_place_QUAD`: coefficient `-0.001227` (lowers CT win probability)
- `lag_07__T2__is_walking`: coefficient `0.001202` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001189` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001189` (raises CT win probability)
- `lag_00__CT_place_BALCONY`: coefficient `0.001145` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `-0.001102` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `105360`, seconds `96.00`, LSTM delta `-0.2106`

Top all feature movements:
- `lag_11__T_place_QUAD`: contribution `-0.053420`
- `lag_06__T_place_QUAD`: contribution `-0.024963`
- `lag_02__T_place_QUAD`: contribution `-0.019305`
- `lag_03__T_place_ARCH`: contribution `-0.014401`
- `lag_00__CT_place_BALCONY`: contribution `-0.007347`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `105520`, seconds `98.50`, LSTM delta `-0.0892`

Top all feature movements:
- `lag_11__T_place_QUAD`: contribution `-0.053420`
- `lag_14__T_place_QUAD`: contribution `+0.008237`
- `lag_00__T_shots_fired_sum`: contribution `-0.005347`
- `lag_01__T_place_ARCH`: contribution `+0.004844`
- `lag_06__T_shots_fired_sum`: contribution `+0.004133`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103088`, seconds `60.50`, LSTM delta `+0.0873`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006037`
- `lag_00__kill_diff_last_3s`: contribution `+0.002861`
- `lag_00__CT_kills_last_3s`: contribution `+0.002691`
- `lag_11__CT2__duck_amount`: contribution `+0.002417`
- `lag_11__T3__duck_amount`: contribution `+0.002406`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `101456`, seconds `35.00`, LSTM delta `+0.0853`

Top all feature movements:
- `lag_02__T_place_BALCONY`: contribution `+0.013988`
- `lag_07__T_place_BALCONY`: contribution `+0.013893`
- `lag_07__T4__flash_duration`: contribution `+0.002929`
- `lag_00__T2__flash_duration`: contribution `+0.001992`
- `lag_07__T3__is_walking`: contribution `+0.001972`

Top utility-only movements:
- `lag_07__T4__flash_duration`: contribution `+0.002929`
- `lag_00__T2__flash_duration`: contribution `+0.001992`

### tick `108080`, seconds `138.50`, LSTM delta `-0.0594`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.008624`
- `lag_06__T_shots_fired_sum`: contribution `-0.004133`
- `lag_00__kill_diff_last_3s`: contribution `-0.002861`
- `lag_04__T4__shots_fired`: contribution `-0.002696`
- `lag_01__T_duck_amount_mean`: contribution `-0.002390`

Top utility-only movements:
- No utility movement among the top local contributors.
