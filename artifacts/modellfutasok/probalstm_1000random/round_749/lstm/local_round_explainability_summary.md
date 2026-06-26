# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `57952`, seconds `74.50`, LSTM `0.8473`, delta `+0.3177`
- tick `56160`, seconds `46.50`, LSTM `0.7275`, delta `-0.1763`
- tick `58528`, seconds `83.50`, LSTM `0.8774`, delta `+0.1653`
- tick `57984`, seconds `75.00`, LSTM `0.7235`, delta `-0.1238`
- tick `54336`, seconds `18.00`, LSTM `0.6995`, delta `+0.1199`
- tick `56096`, seconds `45.50`, LSTM `0.9053`, delta `+0.1114`
- tick `57152`, seconds `62.00`, LSTM `0.7559`, delta `+0.0808`
- tick `57536`, seconds `68.00`, LSTM `0.5785`, delta `-0.0736`
- tick `54368`, seconds `18.50`, LSTM `0.7654`, delta `+0.0660`
- tick `56064`, seconds `45.00`, LSTM `0.7939`, delta `+0.0656`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003128`, |coef| `0.003128`
- `lag_01__T_bomb_zone_count`: coefficient `0.002970`, |coef| `0.002970`
- `lag_00__CT_kills_last_3s`: coefficient `0.002948`, |coef| `0.002948`
- `lag_00__CT2__duck_amount`: coefficient `0.002799`, |coef| `0.002799`
- `lag_00__damage_diff_last_5s`: coefficient `0.002765`, |coef| `0.002765`
- `lag_00__CT2__flash_duration`: coefficient `0.002655`, |coef| `0.002655`
- `lag_13__T4__is_scoped`: coefficient `0.002433`, |coef| `0.002433`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002328`, |coef| `0.002328`
- `lag_00__CT_damage_last_5s`: coefficient `0.002297`, |coef| `0.002297`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002253`, |coef| `0.002253`
- `lag_10__T1__flash_duration`: coefficient `-0.002219`, |coef| `0.002219`
- `lag_03__CT_flashes_last_5s`: coefficient `-0.002059`, |coef| `0.002059`
- `lag_09__T5__duck_amount`: coefficient `0.001920`, |coef| `0.001920`
- `lag_15__T4__is_walking`: coefficient `0.001869`, |coef| `0.001869`
- `lag_00__T1__flash_duration`: coefficient `0.001867`, |coef| `0.001867`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `0.002655` (raises CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.002219` (lowers CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `-0.002059` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001867` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.001805` (lowers CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `0.001754` (raises CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.001541` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.001521` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.001480` (raises CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `0.001358` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003128` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `0.002970` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002948` (raises CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.002799` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002765` (raises CT win probability)
- `lag_13__T4__is_scoped`: coefficient `0.002433` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.002328` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002297` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002253` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.001920` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `57952`, seconds `74.50`, LSTM delta `+0.3177`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `+0.017291`
- `lag_13__T4__is_scoped`: contribution `+0.011299`
- `lag_00__CT2__duck_amount`: contribution `+0.010664`
- `lag_00__damage_diff_last_5s`: contribution `+0.009606`
- `lag_00__T_bomb_zone_count`: contribution `+0.008638`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `+0.008612`
- `lag_10__T1__flash_duration`: contribution `+0.007399`

### tick `56160`, seconds `46.50`, LSTM delta `-0.1763`

Top all feature movements:
- `lag_03__CT_flashes_last_5s`: contribution `-0.022644`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.017503`
- `lag_00__kill_diff_last_3s`: contribution `-0.007530`
- `lag_07__T_place_SIDEENTRANCE`: contribution `-0.006488`
- `lag_02__T_place_SIDEENTRANCE`: contribution `-0.004381`

Top utility-only movements:
- `lag_03__CT_flashes_last_5s`: contribution `-0.022644`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.002970`

### tick `58528`, seconds `83.50`, LSTM delta `+0.1653`

Top all feature movements:
- `lag_13__T4__is_scoped`: contribution `+0.011299`
- `lag_00__T_bomb_zone_count`: contribution `-0.008638`
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_00__kill_diff_last_3s`: contribution `+0.007530`
- `lag_08__T4__flash_duration`: contribution `+0.007156`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.007156`
- `lag_02__CT4__flash_duration`: contribution `+0.004277`
- `lag_08__CT4__flash_duration`: contribution `+0.003600`

### tick `57984`, seconds `75.00`, LSTM delta `-0.1238`

Top all feature movements:
- `lag_01__T_bomb_zone_count`: contribution `-0.017291`
- `lag_00__CT2__duck_amount`: contribution `-0.010664`
- `lag_00__CT2__flash_duration`: contribution `-0.008299`
- `lag_00__kill_diff_last_3s`: contribution `-0.007530`
- `lag_00__CT_shots_fired_sum`: contribution `-0.006260`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.008299`

### tick `54336`, seconds `18.00`, LSTM delta `+0.1199`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008512`
- `lag_00__kill_diff_last_3s`: contribution `+0.007530`
- `lag_06__CT4__flash_duration`: contribution `+0.004715`
- `lag_11__CT_place_SIDEHALL`: contribution `+0.003903`
- `lag_00__damage_diff_last_5s`: contribution `+0.003743`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.004715`
- `lag_15__CT4__flash_duration`: contribution `+0.003099`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.002568`
- `lag_04__CT_B_site_active_infernos`: contribution `+0.002464`
- `lag_12__T1__flash_duration`: contribution `+0.002318`
