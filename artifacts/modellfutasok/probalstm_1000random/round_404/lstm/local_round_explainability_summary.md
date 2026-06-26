# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `137994`, seconds `38.50`, LSTM `0.8081`, delta `+0.1259`
- tick `139018`, seconds `54.50`, LSTM `0.8771`, delta `+0.1169`
- tick `136842`, seconds `20.50`, LSTM `0.7289`, delta `+0.0906`
- tick `138954`, seconds `53.50`, LSTM `0.7990`, delta `-0.0452`
- tick `139146`, seconds `56.50`, LSTM `0.9356`, delta `+0.0446`
- tick `137674`, seconds `33.50`, LSTM `0.7523`, delta `+0.0413`
- tick `138986`, seconds `54.00`, LSTM `0.7602`, delta `-0.0389`
- tick `137066`, seconds `24.00`, LSTM `0.7237`, delta `-0.0385`
- tick `137162`, seconds `25.50`, LSTM `0.7174`, delta `-0.0382`
- tick `136874`, seconds `21.00`, LSTM `0.7640`, delta `+0.0351`

## Top 15 local ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001420`, |coef| `0.001420`
- `lag_00__CT_kills_last_3s`: coefficient `0.001413`, |coef| `0.001413`
- `lag_00__damage_diff_last_5s`: coefficient `0.001306`, |coef| `0.001306`
- `lag_00__T_bomb_zone_count`: coefficient `-0.001241`, |coef| `0.001241`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001196`, |coef| `0.001196`
- `lag_00__kill_diff_last_3s`: coefficient `0.001178`, |coef| `0.001178`
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001165`, |coef| `0.001165`
- `lag_00__CT_place_ARCH`: coefficient `-0.001048`, |coef| `0.001048`
- `lag_02__CT5__flash_duration`: coefficient `0.001037`, |coef| `0.001037`
- `lag_00__CT_damage_last_5s`: coefficient `0.001021`, |coef| `0.001021`
- `lag_02__CT3__is_walking`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_08__CT_shots_fired_sum`: coefficient `-0.000999`, |coef| `0.000999`
- `lag_00__T_flashed_players`: coefficient `-0.000988`, |coef| `0.000988`
- `lag_08__CT5__shots_fired`: coefficient `-0.000951`, |coef| `0.000951`
- `lag_08__CT_place_QUAD`: coefficient `0.000942`, |coef| `0.000942`

## Top 10 utility ridge features

- `lag_00__CT_utility_damage_last_5s`: coefficient `0.001420` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001165` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.001037` (raises CT win probability)
- `lag_13__T2__flash_duration`: coefficient `0.000912` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000855` (raises CT win probability)
- `lag_02__CT_flash_duration_sum`: coefficient `0.000723` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.000710` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000649` (lowers CT win probability)
- `lag_13__T_flash_duration_sum`: coefficient `0.000645` (raises CT win probability)
- `lag_11__T2__flash_duration`: coefficient `-0.000639` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001413` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001306` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.001241` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001196` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001178` (raises CT win probability)
- `lag_00__CT_place_ARCH`: coefficient `-0.001048` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001021` (raises CT win probability)
- `lag_02__CT3__is_walking`: coefficient `-0.001014` (lowers CT win probability)
- `lag_08__CT_shots_fired_sum`: coefficient `-0.000999` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.000988` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `137994`, seconds `38.50`, LSTM delta `+0.1259`

Top all feature movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.016722`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.011251`
- `lag_08__CT_shots_fired_sum`: contribution `+0.006942`
- `lag_08__CT5__shots_fired`: contribution `+0.005029`
- `lag_00__CT_place_ARCH`: contribution `+0.004275`

Top utility-only movements:
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.016722`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.011251`

### tick `139018`, seconds `54.50`, LSTM delta `+0.1169`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.007227`
- `lag_02__CT5__flash_duration`: contribution `+0.006996`
- `lag_13__T2__flash_duration`: contribution `+0.006542`
- `lag_01__T_bomb_zone_count`: contribution `+0.004490`
- `lag_00__CT_kills_last_3s`: contribution `+0.004080`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.006996`
- `lag_13__T2__flash_duration`: contribution `+0.006542`
- `lag_02__CT_flash_duration_sum`: contribution `+0.003556`
- `lag_02__T2__flash_duration`: contribution `+0.002957`
- `lag_02__CT4__flash_duration`: contribution `+0.002208`

### tick `136842`, seconds `20.50`, LSTM delta `+0.0906`

Top all feature movements:
- `lag_08__CT_place_QUAD`: contribution `+0.007423`
- `lag_13__T4__flash_duration`: contribution `+0.005475`
- `lag_01__T4__flash_duration`: contribution `+0.004548`
- `lag_00__CT_kills_last_3s`: contribution `+0.004080`
- `lag_00__kill_diff_last_3s`: contribution `+0.002836`

Top utility-only movements:
- `lag_13__T4__flash_duration`: contribution `+0.005475`
- `lag_01__T4__flash_duration`: contribution `+0.004548`
- `lag_13__T_flash_duration_sum`: contribution `+0.001738`
- `lag_05__CT_B_site_active_infernos`: contribution `+0.001271`

### tick `138954`, seconds `53.50`, LSTM delta `-0.0452`

Top all feature movements:
- `lag_11__T2__flash_duration`: contribution `-0.004589`
- `lag_00__CT5__flash_duration`: contribution `-0.004377`
- `lag_00__T2__flash_duration`: contribution `-0.003369`
- `lag_02__CT3__is_walking`: contribution `-0.002420`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002418`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `-0.004589`
- `lag_00__CT5__flash_duration`: contribution `-0.004377`
- `lag_00__T2__flash_duration`: contribution `-0.003369`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002418`
- `lag_00__CT4__flash_duration`: contribution `-0.001216`

### tick `139146`, seconds `56.50`, LSTM delta `+0.0446`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004080`
- `lag_00__kill_diff_last_3s`: contribution `+0.002836`
- `lag_01__CT5__duck_amount`: contribution `+0.002304`
- `lag_01__T4__is_walking`: contribution `+0.001666`
- `lag_06__CT5__flash_duration`: contribution `+0.001583`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.001583`
- `lag_06__CT_flash_duration_sum`: contribution `+0.000895`
