# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `14857`, seconds `72.00`, LSTM `0.5402`, delta `+0.2347`
- tick `15241`, seconds `78.00`, LSTM `0.7291`, delta `+0.1618`
- tick `15401`, seconds `80.50`, LSTM `0.8528`, delta `+0.1577`
- tick `15145`, seconds `76.50`, LSTM `0.7072`, delta `+0.1516`
- tick `15561`, seconds `83.00`, LSTM `0.8657`, delta `+0.1480`
- tick `15465`, seconds `81.50`, LSTM `0.7128`, delta `-0.1392`
- tick `13449`, seconds `50.00`, LSTM `0.2482`, delta `-0.1240`
- tick `15177`, seconds `77.00`, LSTM `0.6095`, delta `-0.0977`
- tick `14793`, seconds `71.00`, LSTM `0.3621`, delta `+0.0814`
- tick `15625`, seconds `84.00`, LSTM `0.9499`, delta `+0.0760`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004875`, |coef| `0.004875`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002885`, |coef| `0.002885`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002521`, |coef| `0.002521`
- `lag_00__T_macro_B`: coefficient `-0.002521`, |coef| `0.002521`
- `lag_00__kill_diff_last_3s`: coefficient `0.002138`, |coef| `0.002138`
- `lag_00__CT_kills_last_3s`: coefficient `0.002070`, |coef| `0.002070`
- `lag_00__damage_diff_last_5s`: coefficient `0.001735`, |coef| `0.001735`
- `lag_00__CT_damage_last_5s`: coefficient `0.001662`, |coef| `0.001662`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001576`, |coef| `0.001576`
- `lag_00__CT_place_BANANA`: coefficient `-0.001511`, |coef| `0.001511`
- `lag_03__CT2__shots_fired`: coefficient `0.001501`, |coef| `0.001501`
- `lag_08__T1__is_walking`: coefficient `-0.001464`, |coef| `0.001464`
- `lag_11__T2__is_walking`: coefficient `0.001442`, |coef| `0.001442`
- `lag_05__CT_defusing_count`: coefficient `0.001419`, |coef| `0.001419`
- `lag_02__T_place_BOMBSITEB`: coefficient `-0.001418`, |coef| `0.001418`

## Top 10 utility ridge features

- `lag_00__CT_B_site_active_infernos`: coefficient `0.001369` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.001062` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000971` (lowers CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.000942` (lowers CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `-0.000930` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `-0.000891` (lowers CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000864` (raises CT win probability)
- `lag_10__CT4__molly`: coefficient `-0.000852` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000824` (lowers CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `0.000800` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.004875` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002885` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002521` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002521` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002138` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002070` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001735` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001662` (raises CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `0.001576` (raises CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `-0.001511` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14857`, seconds `72.00`, LSTM delta `+0.2347`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.016036`
- `lag_03__CT_shots_fired_sum`: contribution `+0.013142`
- `lag_01__CT2__shots_fired`: contribution `+0.012004`
- `lag_06__CT_flashed_players`: contribution `+0.008809`
- `lag_00__CT_kills_last_3s`: contribution `+0.005978`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `+0.004702`

### tick `15241`, seconds `78.00`, LSTM delta `+0.1618`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010023`
- `lag_13__CT2__shots_fired`: contribution `+0.008910`
- `lag_00__CT_kills_last_3s`: contribution `+0.005978`
- `lag_03__CT_shots_fired_sum`: contribution `+0.005476`
- `lag_12__CT_shots_fired_sum`: contribution `+0.005196`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.004702`
- `lag_13__CT5__flash_duration`: contribution `+0.002443`

### tick `15401`, seconds `80.50`, LSTM delta `+0.1577`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.047261`
- `lag_00__CT_shots_fired_sum`: contribution `+0.014032`
- `lag_01__CT1__shots_fired`: contribution `+0.003609`
- `lag_12__T_duck_amount_mean`: contribution `+0.003525`
- `lag_14__T_duck_amount_mean`: contribution `+0.003446`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15145`, seconds `76.50`, LSTM delta `+0.1516`

Top all feature movements:
- `lag_10__CT2__shots_fired`: contribution `+0.012997`
- `lag_10__CT_shots_fired_sum`: contribution `+0.010977`
- `lag_00__CT_shots_fired_sum`: contribution `+0.010023`
- `lag_12__CT_shots_fired_sum`: contribution `+0.007794`
- `lag_00__CT_kills_last_3s`: contribution `+0.005978`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `15561`, seconds `83.00`, LSTM delta `+0.1480`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.047261`
- `lag_05__CT_defusing_count`: contribution `+0.013755`
- `lag_03__CT_defusing_count`: contribution `+0.007533`
- `lag_04__CT1__shots_fired`: contribution `+0.005643`
- `lag_12__CT4__duck_amount`: contribution `+0.003815`

Top utility-only movements:
- `lag_10__CT_B_site_active_infernos`: contribution `+0.002172`
