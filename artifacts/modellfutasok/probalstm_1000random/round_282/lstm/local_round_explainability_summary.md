# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `4`

## Largest probability jumps

- tick `25492`, seconds `93.50`, LSTM `0.8269`, delta `+0.2662`
- tick `25364`, seconds `91.50`, LSTM `0.5705`, delta `-0.0937`
- tick `20276`, seconds `12.00`, LSTM `0.6720`, delta `+0.0682`
- tick `25556`, seconds `94.50`, LSTM `0.9433`, delta `+0.0663`
- tick `25300`, seconds `90.50`, LSTM `0.7003`, delta `+0.0593`
- tick `20308`, seconds `12.50`, LSTM `0.7296`, delta `+0.0576`
- tick `25236`, seconds `89.50`, LSTM `0.6102`, delta `+0.0505`
- tick `25524`, seconds `94.00`, LSTM `0.8770`, delta `+0.0501`
- tick `24948`, seconds `85.00`, LSTM `0.5994`, delta `-0.0404`
- tick `21204`, seconds `26.50`, LSTM `0.7188`, delta `+0.0398`

## Top 15 local ridge features

- `lag_04__T_place_QUAD`: coefficient `-0.002453`, |coef| `0.002453`
- `lag_05__T_place_QUAD`: coefficient `-0.002038`, |coef| `0.002038`
- `lag_08__T_place_QUAD`: coefficient `0.001938`, |coef| `0.001938`
- `lag_00__T_place_QUAD`: coefficient `0.001550`, |coef| `0.001550`
- `lag_06__T_place_QUAD`: coefficient `0.001349`, |coef| `0.001349`
- `lag_03__T_place_ARCH`: coefficient `0.001057`, |coef| `0.001057`
- `lag_10__T_place_QUAD`: coefficient `0.000970`, |coef| `0.000970`
- `lag_11__CT_place_LIBRARY`: coefficient `0.000937`, |coef| `0.000937`
- `lag_00__damage_diff_last_5s`: coefficient `0.000830`, |coef| `0.000830`
- `lag_00__CT_flashes_last_5s`: coefficient `-0.000749`, |coef| `0.000749`
- `lag_12__CT_place_LIBRARY`: coefficient `0.000726`, |coef| `0.000726`
- `lag_00__CT3__duck_amount`: coefficient `0.000718`, |coef| `0.000718`
- `lag_09__CT_place_LIBRARY`: coefficient `0.000705`, |coef| `0.000705`
- `lag_09__T_place_QUAD`: coefficient `0.000696`, |coef| `0.000696`
- `lag_00__CT3__is_scoped`: coefficient `0.000668`, |coef| `0.000668`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000749` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `0.000467` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `0.000415` (raises CT win probability)
- `lag_07__T3__flash_duration`: coefficient `0.000410` (raises CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.000409` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000396` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000393` (raises CT win probability)
- `lag_13__T3__flash_duration`: coefficient `0.000382` (raises CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.000374` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.000358` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__T_place_QUAD`: coefficient `-0.002453` (lowers CT win probability)
- `lag_05__T_place_QUAD`: coefficient `-0.002038` (lowers CT win probability)
- `lag_08__T_place_QUAD`: coefficient `0.001938` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.001550` (raises CT win probability)
- `lag_06__T_place_QUAD`: coefficient `0.001349` (raises CT win probability)
- `lag_03__T_place_ARCH`: coefficient `0.001057` (raises CT win probability)
- `lag_10__T_place_QUAD`: coefficient `0.000970` (raises CT win probability)
- `lag_11__CT_place_LIBRARY`: coefficient `0.000937` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000830` (raises CT win probability)
- `lag_12__CT_place_LIBRARY`: coefficient `0.000726` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `25492`, seconds `93.50`, LSTM delta `+0.2662`

Top all feature movements:
- `lag_04__T_place_QUAD`: contribution `+0.059089`
- `lag_05__T_place_QUAD`: contribution `+0.049080`
- `lag_08__T_place_QUAD`: contribution `+0.046685`
- `lag_06__T_place_QUAD`: contribution `+0.032504`
- `lag_03__T_place_ARCH`: contribution `+0.009830`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.001067`

### tick `25364`, seconds `91.50`, LSTM delta `-0.0937`

Top all feature movements:
- `lag_04__T_place_QUAD`: contribution `-0.059089`
- `lag_00__T_place_QUAD`: contribution `-0.037339`
- `lag_02__T_place_QUAD`: contribution `-0.006151`
- `lag_07__CT_place_LIBRARY`: contribution `+0.003521`
- `lag_01__T_place_QUAD`: contribution `+0.003474`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20276`, seconds `12.00`, LSTM delta `+0.0682`

Top all feature movements:
- `lag_15__T_place_LOWERMID`: contribution `+0.003961`
- `lag_14__CT_place_LIBRARY`: contribution `-0.003277`
- `lag_00__CT3__is_scoped`: contribution `-0.003040`
- `lag_14__T_place_LOWERMID`: contribution `+0.002698`
- `lag_02__CT_place_BALCONY`: contribution `+0.002683`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.001572`
- `lag_04__T_A_site_active_infernos`: contribution `+0.001171`

### tick `25556`, seconds `94.50`, LSTM delta `+0.0663`

Top all feature movements:
- `lag_08__T_place_QUAD`: contribution `+0.046686`
- `lag_06__T_place_QUAD`: contribution `-0.032504`
- `lag_10__T_place_QUAD`: contribution `+0.023352`
- `lag_05__T_place_ARCH`: contribution `+0.004270`
- `lag_13__CT_place_LIBRARY`: contribution `+0.003547`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `+0.001171`

### tick `25300`, seconds `90.50`, LSTM delta `+0.0593`

Top all feature movements:
- `lag_00__T_place_QUAD`: contribution `+0.037339`
- `lag_02__T_place_QUAD`: contribution `-0.006151`
- `lag_00__CT3__is_scoped`: contribution `+0.003040`
- `lag_05__CT_place_LIBRARY`: contribution `+0.002495`
- `lag_12__CT_place_TOPOFMID`: contribution `+0.001649`

Top utility-only movements:
- No utility movement among the top local contributors.
