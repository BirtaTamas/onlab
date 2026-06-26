# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `14381`, seconds `97.50`, LSTM `0.6070`, delta `+0.4116`
- tick `14605`, seconds `101.00`, LSTM `0.8258`, delta `+0.2195`
- tick `11597`, seconds `54.00`, LSTM `0.5262`, delta `-0.2016`
- tick `11565`, seconds `53.50`, LSTM `0.7278`, delta `+0.1952`
- tick `13613`, seconds `85.50`, LSTM `0.2709`, delta `+0.1750`
- tick `11661`, seconds `55.00`, LSTM `0.2965`, delta `-0.1547`
- tick `12941`, seconds `75.00`, LSTM `0.3828`, delta `-0.1456`
- tick `13357`, seconds `81.50`, LSTM `0.1487`, delta `-0.1202`
- tick `11373`, seconds `50.50`, LSTM `0.5133`, delta `+0.0890`
- tick `11629`, seconds `54.50`, LSTM `0.4512`, delta `-0.0750`

## Top 15 local ridge features

- `lag_00__T_place_GRAVEYARD`: coefficient `-0.008892`, |coef| `0.008892`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.005181`, |coef| `0.005181`
- `lag_00__CT_place_QUAD`: coefficient `0.004915`, |coef| `0.004915`
- `lag_00__kill_diff_last_3s`: coefficient `0.003804`, |coef| `0.003804`
- `lag_00__CT_defusing_count`: coefficient `0.003553`, |coef| `0.003553`
- `lag_07__T_place_GRAVEYARD`: coefficient `-0.003172`, |coef| `0.003172`
- `lag_00__CT_duck_amount_mean`: coefficient `0.003131`, |coef| `0.003131`
- `lag_00__CT_velocity_mean`: coefficient `-0.002925`, |coef| `0.002925`
- `lag_00__T_place_BALCONY`: coefficient `-0.002821`, |coef| `0.002821`
- `lag_05__T_duck_amount_mean`: coefficient `0.002727`, |coef| `0.002727`
- `lag_07__T_bomb_zone_count`: coefficient `-0.002665`, |coef| `0.002665`
- `lag_00__CT4__is_walking`: coefficient `-0.002573`, |coef| `0.002573`
- `lag_00__CT_kills_last_3s`: coefficient `0.002551`, |coef| `0.002551`
- `lag_00__T_burning_players`: coefficient `-0.002417`, |coef| `0.002417`
- `lag_00__damage_diff_last_5s`: coefficient `0.002411`, |coef| `0.002411`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.005181` (lowers CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `-0.002220` (lowers CT win probability)
- `lag_00__T1__flash_duration`: coefficient `-0.002047` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.001914` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.001894` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `0.001500` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.001407` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001388` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001335` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `-0.001273` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_GRAVEYARD`: coefficient `-0.008892` (lowers CT win probability)
- `lag_00__CT_place_QUAD`: coefficient `0.004915` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003804` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003553` (raises CT win probability)
- `lag_07__T_place_GRAVEYARD`: coefficient `-0.003172` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.003131` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002925` (lowers CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.002821` (lowers CT win probability)
- `lag_05__T_duck_amount_mean`: coefficient `0.002727` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.002665` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14381`, seconds `97.50`, LSTM delta `+0.4116`

Top all feature movements:
- `lag_00__T_place_GRAVEYARD`: contribution `+0.174790`
- `lag_00__CT_place_QUAD`: contribution `+0.038739`
- `lag_00__T_flash_alpha_mean`: contribution `+0.031437`
- `lag_05__T_duck_amount_mean`: contribution `+0.015860`
- `lag_00__T_duck_amount_mean`: contribution `+0.010126`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.031437`
- `lag_00__T5__smoke`: contribution `+0.004104`

### tick `14605`, seconds `101.00`, LSTM delta `+0.2195`

Top all feature movements:
- `lag_07__T_place_GRAVEYARD`: contribution `+0.062354`
- `lag_00__CT_defusing_count`: contribution `+0.034446`
- `lag_07__CT_place_QUAD`: contribution `+0.017247`
- `lag_05__CT_place_QUAD`: contribution `+0.014096`
- `lag_07__T_flash_alpha_mean`: contribution `+0.013469`

Top utility-only movements:
- `lag_07__T_flash_alpha_mean`: contribution `+0.013469`

### tick `11597`, seconds `54.00`, LSTM delta `-0.2016`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.038790`
- `lag_00__kill_diff_last_3s`: contribution `-0.009155`
- `lag_00__CT_duck_amount_mean`: contribution `+0.008229`
- `lag_02__CT_place_BALCONY`: contribution `-0.007786`
- `lag_00__T_kills_last_3s`: contribution `-0.006994`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.006864`
- `lag_09__CT5__flash_duration`: contribution `-0.005773`

### tick `11565`, seconds `53.50`, LSTM delta `+0.1952`

Top all feature movements:
- `lag_01__CT_place_BALCONY`: contribution `+0.006291`
- `lag_06__CT2__flash_duration`: contribution `+0.005721`
- `lag_08__CT5__flash_duration`: contribution `+0.005690`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005321`
- `lag_09__T5__duck_amount`: contribution `+0.005275`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.005721`
- `lag_08__CT5__flash_duration`: contribution `+0.005690`
- `lag_12__CT_utility_damage_last_5s`: contribution `+0.003676`
- `lag_02__T1__flash_duration`: contribution `+0.003141`

### tick `13613`, seconds `85.50`, LSTM delta `+0.1750`

Top all feature movements:
- `lag_01__T_place_GRAVEYARD`: contribution `+0.015543`
- `lag_08__T_bomb_zone_count`: contribution `+0.012936`
- `lag_00__T1__flash_duration`: contribution `+0.012624`
- `lag_00__CT_duck_amount_mean`: contribution `+0.011732`
- `lag_10__T1__flash_duration`: contribution `+0.009252`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.012624`
- `lag_10__T1__flash_duration`: contribution `+0.009252`
- `lag_08__CT_A_site_active_infernos`: contribution `+0.003654`
