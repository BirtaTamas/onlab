# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-natus-vincere-vs-spirit-bo3-cW0x-KCT4cbPLaZUAvb08Z/natus-vincere-vs-spirit-m2-ancient.csv`
- round_num: `37`

## Largest probability jumps

- tick `290637`, seconds `43.00`, LSTM `0.1773`, delta `-0.3810`
- tick `290541`, seconds `41.50`, LSTM `0.5580`, delta `-0.2495`
- tick `288653`, seconds `12.00`, LSTM `0.5278`, delta `+0.1647`
- tick `288621`, seconds `11.50`, LSTM `0.3631`, delta `-0.1371`
- tick `291725`, seconds `60.00`, LSTM `0.0181`, delta `-0.1036`
- tick `288781`, seconds `14.00`, LSTM `0.6727`, delta `+0.1021`
- tick `290509`, seconds `41.00`, LSTM `0.8075`, delta `+0.0558`
- tick `288813`, seconds `14.50`, LSTM `0.7258`, delta `+0.0531`
- tick `288973`, seconds `17.00`, LSTM `0.7352`, delta `-0.0514`
- tick `290093`, seconds `34.50`, LSTM `0.7582`, delta `+0.0404`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.007140`, |coef| `0.007140`
- `lag_00__T_kills_last_3s`: coefficient `-0.004640`, |coef| `0.004640`
- `lag_00__kill_diff_last_3s`: coefficient `0.003833`, |coef| `0.003833`
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.003217`, |coef| `0.003217`
- `lag_04__CT_place_SIDEENTRANCE`: coefficient `0.003202`, |coef| `0.003202`
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.002869`, |coef| `0.002869`
- `lag_00__T_damage_last_5s`: coefficient `-0.002778`, |coef| `0.002778`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002668`, |coef| `0.002668`
- `lag_00__damage_diff_last_5s`: coefficient `0.002554`, |coef| `0.002554`
- `lag_07__CT4__is_walking`: coefficient `-0.002535`, |coef| `0.002535`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `0.002402`, |coef| `0.002402`
- `lag_14__CT1__is_walking`: coefficient `0.002201`, |coef| `0.002201`
- `lag_14__T_place_TSIDELOWER`: coefficient `0.002096`, |coef| `0.002096`
- `lag_00__T1__duck_amount`: coefficient `-0.002063`, |coef| `0.002063`
- `lag_04__CT1__is_walking`: coefficient `0.001966`, |coef| `0.001966`

## Top 10 utility ridge features

- `lag_00__CT2__smoke`: coefficient `0.001787` (raises CT win probability)
- `lag_09__CT4__smoke`: coefficient `0.001635` (raises CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `0.001325` (raises CT win probability)
- `lag_06__CT4__smoke`: coefficient `0.001305` (raises CT win probability)
- `lag_12__CT_he_last_5s`: coefficient `-0.001304` (lowers CT win probability)
- `lag_05__CT_A_site_active_smokes`: coefficient `-0.001222` (lowers CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.001130` (raises CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `0.001011` (raises CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `-0.000945` (lowers CT win probability)
- `lag_08__CT4__smoke`: coefficient `0.000848` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.007140` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004640` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003833` (raises CT win probability)
- `lag_10__CT_place_TSIDEUPPER`: coefficient `-0.003217` (lowers CT win probability)
- `lag_04__CT_place_SIDEENTRANCE`: coefficient `0.003202` (raises CT win probability)
- `lag_04__CT_place_TSIDEUPPER`: coefficient `-0.002869` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002778` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002668` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002554` (raises CT win probability)
- `lag_07__CT4__is_walking`: coefficient `-0.002535` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `290637`, seconds `43.00`, LSTM delta `-0.3810`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.053668`
- `lag_10__CT_place_TSIDEUPPER`: contribution `-0.024183`
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.021569`
- `lag_00__T_kills_last_3s`: contribution `-0.014701`
- `lag_04__CT_place_SIDEENTRANCE`: contribution `-0.012889`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `290541`, seconds `41.50`, LSTM delta `-0.2495`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.053668`
- `lag_00__T_kills_last_3s`: contribution `-0.014701`
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.012940`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `-0.009669`
- `lag_00__kill_diff_last_3s`: contribution `-0.009227`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `288653`, seconds `12.00`, LSTM delta `+0.1647`

Top all feature movements:
- `lag_12__CT_he_last_5s`: contribution `+0.023931`
- `lag_00__T_shots_fired_sum`: contribution `+0.018005`
- `lag_00__CT_flashes_last_5s`: contribution `+0.012428`
- `lag_00__kill_diff_last_3s`: contribution `+0.009227`
- `lag_03__CT_place_SIDEENTRANCE`: contribution `+0.006920`

Top utility-only movements:
- `lag_12__CT_he_last_5s`: contribution `+0.023931`
- `lag_00__CT_flashes_last_5s`: contribution `+0.012428`
- `lag_03__CT3__flash_duration`: contribution `+0.002141`

### tick `288621`, seconds `11.50`, LSTM delta `-0.1371`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.014701`
- `lag_11__CT_he_last_5s`: contribution `-0.014001`
- `lag_00__T_shots_fired_sum`: contribution `-0.010003`
- `lag_00__kill_diff_last_3s`: contribution `-0.009227`
- `lag_14__CT_place_HOUSE`: contribution `-0.007263`

Top utility-only movements:
- `lag_11__CT_he_last_5s`: contribution `-0.014001`
- `lag_02__CT3__flash_duration`: contribution `-0.001950`

### tick `291725`, seconds `60.00`, LSTM delta `-0.1036`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.014701`
- `lag_00__kill_diff_last_3s`: contribution `-0.009227`
- `lag_00__T_damage_last_5s`: contribution `-0.006660`
- `lag_00__damage_diff_last_5s`: contribution `-0.005762`
- `lag_06__T_place_TSIDELOWER`: contribution `-0.004010`

Top utility-only movements:
- `lag_00__CT4__molly`: contribution `-0.001907`
