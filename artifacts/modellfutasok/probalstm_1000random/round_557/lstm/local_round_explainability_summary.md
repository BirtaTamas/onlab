# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `4`

## Largest probability jumps

- tick `29961`, seconds `68.50`, LSTM `0.1046`, delta `-0.3797`
- tick `28265`, seconds `42.00`, LSTM `0.8351`, delta `+0.2121`
- tick `29865`, seconds `67.00`, LSTM `0.4627`, delta `-0.1422`
- tick `29705`, seconds `64.50`, LSTM `0.5605`, delta `-0.0822`
- tick `28233`, seconds `41.50`, LSTM `0.6230`, delta `+0.0821`
- tick `28521`, seconds `46.00`, LSTM `0.8934`, delta `+0.0702`
- tick `33321`, seconds `121.00`, LSTM `0.0861`, delta `+0.0662`
- tick `28873`, seconds `51.50`, LSTM `0.7956`, delta `-0.0648`
- tick `26889`, seconds `20.50`, LSTM `0.5671`, delta `+0.0635`
- tick `28329`, seconds `43.00`, LSTM `0.7894`, delta `-0.0512`

## Top 15 local ridge features

- `lag_00__CT_place_CRANE`: coefficient `0.004655`, |coef| `0.004655`
- `lag_00__kill_diff_last_3s`: coefficient `0.003431`, |coef| `0.003431`
- `lag_00__T_kills_last_3s`: coefficient `-0.002960`, |coef| `0.002960`
- `lag_00__damage_diff_last_5s`: coefficient `0.002932`, |coef| `0.002932`
- `lag_01__CT_place_CRANE`: coefficient `0.002306`, |coef| `0.002306`
- `lag_00__T_damage_last_5s`: coefficient `-0.002079`, |coef| `0.002079`
- `lag_03__kill_diff_last_3s`: coefficient `0.002066`, |coef| `0.002066`
- `lag_08__CT4__is_walking`: coefficient `0.002045`, |coef| `0.002045`
- `lag_00__CT1__flash_duration`: coefficient `0.001990`, |coef| `0.001990`
- `lag_03__CT_place_GARAGE`: coefficient `0.001947`, |coef| `0.001947`
- `lag_05__CT_place_CRANE`: coefficient `-0.001890`, |coef| `0.001890`
- `lag_03__CT_place_RAMP`: coefficient `0.001733`, |coef| `0.001733`
- `lag_00__CT4__duck_amount`: coefficient `0.001725`, |coef| `0.001725`
- `lag_14__T4__is_walking`: coefficient `-0.001716`, |coef| `0.001716`
- `lag_03__CT2__duck_amount`: coefficient `-0.001702`, |coef| `0.001702`

## Top 10 utility ridge features

- `lag_00__CT1__flash_duration`: coefficient `0.001990` (raises CT win probability)
- `lag_05__CT_B_site_active_smokes`: coefficient `-0.001474` (lowers CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `-0.001326` (lowers CT win probability)
- `lag_03__CT2__flash`: coefficient `0.001099` (raises CT win probability)
- `lag_08__CT3__flash`: coefficient `0.001090` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001040` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.000987` (raises CT win probability)
- `lag_13__CT1__flash_duration`: coefficient `0.000946` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000921` (raises CT win probability)
- `lag_05__CT_active_smokes`: coefficient `-0.000905` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CRANE`: coefficient `0.004655` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003431` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002960` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002932` (raises CT win probability)
- `lag_01__CT_place_CRANE`: coefficient `0.002306` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002079` (lowers CT win probability)
- `lag_03__kill_diff_last_3s`: coefficient `0.002066` (raises CT win probability)
- `lag_08__CT4__is_walking`: coefficient `0.002045` (raises CT win probability)
- `lag_03__CT_place_GARAGE`: coefficient `0.001947` (raises CT win probability)
- `lag_05__CT_place_CRANE`: coefficient `-0.001890` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `29961`, seconds `68.50`, LSTM delta `-0.3797`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `-0.076367`
- `lag_05__CT_place_CRANE`: contribution `-0.031002`
- `lag_03__CT_place_GARAGE`: contribution `-0.013992`
- `lag_00__T_kills_last_3s`: contribution `-0.009378`
- `lag_00__kill_diff_last_3s`: contribution `-0.008258`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `28265`, seconds `42.00`, LSTM delta `+0.2121`

Top all feature movements:
- `lag_01__CT_place_CRANE`: contribution `+0.037827`
- `lag_00__CT1__flash_duration`: contribution `+0.013793`
- `lag_00__kill_diff_last_3s`: contribution `+0.008258`
- `lag_11__CT_shots_fired_sum`: contribution `+0.006258`
- `lag_05__CT_B_site_active_smokes`: contribution `+0.004897`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.013793`
- `lag_05__CT_B_site_active_smokes`: contribution `+0.004897`
- `lag_05__CT_A_site_active_smokes`: contribution `+0.004270`
- `lag_00__T5__flash`: contribution `+0.002952`

### tick `29865`, seconds `67.00`, LSTM delta `-0.1422`

Top all feature movements:
- `lag_00__CT_place_GARAGE`: contribution `-0.009401`
- `lag_00__T_kills_last_3s`: contribution `-0.009378`
- `lag_02__CT_place_CRANE`: contribution `-0.009256`
- `lag_00__kill_diff_last_3s`: contribution `-0.008258`
- `lag_00__damage_diff_last_5s`: contribution `-0.006615`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29705`, seconds `64.50`, LSTM delta `-0.0822`

Top all feature movements:
- `lag_12__CT_place_MINI`: contribution `-0.010058`
- `lag_00__T_kills_last_3s`: contribution `-0.009378`
- `lag_00__kill_diff_last_3s`: contribution `-0.008258`
- `lag_12__T_place_SECRET`: contribution `-0.007785`
- `lag_03__T2__is_walking`: contribution `-0.003109`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.001440`

### tick `28233`, seconds `41.50`, LSTM delta `+0.0821`

Top all feature movements:
- `lag_00__CT_place_CRANE`: contribution `+0.076367`
- `lag_00__CT_place_RAFTERS`: contribution `-0.006221`
- `lag_00__T_shots_fired_sum`: contribution `-0.004148`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003670`
- `lag_08__CT2__is_walking`: contribution `+0.003500`

Top utility-only movements:
- `lag_04__CT_B_site_active_smokes`: contribution `+0.001717`
