# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `94634`, seconds `72.00`, LSTM `0.7026`, delta `+0.4002`
- tick `94794`, seconds `74.50`, LSTM `0.4850`, delta `-0.3009`
- tick `94570`, seconds `71.00`, LSTM `0.3568`, delta `+0.2596`
- tick `94314`, seconds `67.00`, LSTM `0.3077`, delta `-0.2302`
- tick `94026`, seconds `62.50`, LSTM `0.5077`, delta `+0.2108`
- tick `93418`, seconds `53.00`, LSTM `0.4829`, delta `+0.1730`
- tick `94410`, seconds `68.50`, LSTM `0.1458`, delta `-0.1302`
- tick `93930`, seconds `61.00`, LSTM `0.4183`, delta `-0.1010`
- tick `91786`, seconds `27.50`, LSTM `0.4140`, delta `-0.0927`
- tick `94954`, seconds `77.00`, LSTM `0.3053`, delta `-0.0721`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003278`, |coef| `0.003278`
- `lag_00__damage_diff_last_5s`: coefficient `0.002859`, |coef| `0.002859`
- `lag_00__CT_kills_last_3s`: coefficient `0.002625`, |coef| `0.002625`
- `lag_07__T_place_MINI`: coefficient `0.002170`, |coef| `0.002170`
- `lag_07__T_utility_damage_last_5s`: coefficient `0.002163`, |coef| `0.002163`
- `lag_00__CT_place_MINI`: coefficient `0.002152`, |coef| `0.002152`
- `lag_12__CT_place_MINI`: coefficient `0.002142`, |coef| `0.002142`
- `lag_08__T_place_HUT`: coefficient `0.002080`, |coef| `0.002080`
- `lag_06__CT_place_CONTROL`: coefficient `0.002074`, |coef| `0.002074`
- `lag_04__T_place_HUT`: coefficient `-0.001997`, |coef| `0.001997`
- `lag_08__CT_place_MINI`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_12__T_place_HUT`: coefficient `0.001932`, |coef| `0.001932`
- `lag_15__CT_place_MINI`: coefficient `0.001911`, |coef| `0.001911`
- `lag_14__CT_place_MINI`: coefficient `0.001909`, |coef| `0.001909`
- `lag_10__T_place_HUT`: coefficient `0.001866`, |coef| `0.001866`

## Top 10 utility ridge features

- `lag_07__T_utility_damage_last_5s`: coefficient `0.002163` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `0.001687` (raises CT win probability)
- `lag_05__T_utility_damage_last_5s`: coefficient `0.001662` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.001472` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001455` (lowers CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001450` (lowers CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `-0.001344` (lowers CT win probability)
- `lag_13__CT4__flash_duration`: coefficient `0.001329` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.001290` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.001224` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003278` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002859` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002625` (raises CT win probability)
- `lag_07__T_place_MINI`: coefficient `0.002170` (raises CT win probability)
- `lag_00__CT_place_MINI`: coefficient `0.002152` (raises CT win probability)
- `lag_12__CT_place_MINI`: coefficient `0.002142` (raises CT win probability)
- `lag_08__T_place_HUT`: coefficient `0.002080` (raises CT win probability)
- `lag_06__CT_place_CONTROL`: coefficient `0.002074` (raises CT win probability)
- `lag_04__T_place_HUT`: coefficient `-0.001997` (lowers CT win probability)
- `lag_08__CT_place_MINI`: coefficient `-0.001976` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `94634`, seconds `72.00`, LSTM delta `+0.4002`

Top all feature movements:
- `lag_08__T_place_HUT`: contribution `+0.019386`
- `lag_12__T_place_HUT`: contribution `+0.018007`
- `lag_07__T_utility_damage_last_5s`: contribution `+0.017604`
- `lag_12__CT_place_MINI`: contribution `+0.013135`
- `lag_15__CT4__flash_duration`: contribution `+0.011748`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `+0.017604`
- `lag_15__CT4__flash_duration`: contribution `+0.011748`
- `lag_03__CT4__flash_duration`: contribution `+0.010099`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.006918`

### tick `94794`, seconds `74.50`, LSTM delta `-0.3009`

Top all feature movements:
- `lag_00__CT_place_MINI`: contribution `-0.013194`
- `lag_13__T_place_HUT`: contribution `-0.013024`
- `lag_08__CT_place_MINI`: contribution `-0.012113`
- `lag_12__T_utility_damage_last_5s`: contribution `-0.011982`
- `lag_15__CT_place_MINI`: contribution `-0.011718`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `-0.011982`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.007168`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.004323`
- `lag_08__CT4__flash_duration`: contribution `-0.004041`

### tick `94570`, seconds `71.00`, LSTM delta `+0.2596`

Top all feature movements:
- `lag_04__T_place_HUT`: contribution `+0.018617`
- `lag_10__T_place_HUT`: contribution `+0.017393`
- `lag_15__CT_place_SECRET`: contribution `+0.015412`
- `lag_05__T_utility_damage_last_5s`: contribution `+0.013527`
- `lag_08__CT_place_MINI`: contribution `+0.012113`

Top utility-only movements:
- `lag_05__T_utility_damage_last_5s`: contribution `+0.013527`
- `lag_13__CT4__flash_duration`: contribution `+0.009252`
- `lag_01__CT4__flash_duration`: contribution `+0.007548`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.005331`

### tick `94314`, seconds `67.00`, LSTM delta `-0.2302`

Top all feature movements:
- `lag_09__T_place_MINI`: contribution `-0.024594`
- `lag_00__CT_place_MINI`: contribution `-0.013194`
- `lag_12__CT_place_MINI`: contribution `-0.013135`
- `lag_02__T_place_HUT`: contribution `-0.012195`
- `lag_07__CT_place_SECRET`: contribution `-0.008908`

Top utility-only movements:
- `lag_05__CT4__flash_duration`: contribution `-0.007075`
- `lag_06__T_A_site_active_infernos`: contribution `-0.003840`
- `lag_06__T_B_site_active_infernos`: contribution `-0.003460`

### tick `94026`, seconds `62.50`, LSTM delta `+0.2108`

Top all feature movements:
- `lag_07__T_place_MINI`: contribution `+0.030197`
- `lag_00__T_place_MINI`: contribution `+0.024246`
- `lag_06__CT_place_ADMIN`: contribution `+0.011668`
- `lag_09__CT_place_ADMIN`: contribution `+0.009465`
- `lag_15__CT_place_CONTROL`: contribution `+0.007931`

Top utility-only movements:
- No utility movement among the top local contributors.
