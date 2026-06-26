# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `8`

## Largest probability jumps

- tick `63395`, seconds `55.50`, LSTM `0.7620`, delta `+0.1993`
- tick `63587`, seconds `58.50`, LSTM `0.8466`, delta `-0.0987`
- tick `63491`, seconds `57.00`, LSTM `0.9284`, delta `+0.0956`
- tick `63427`, seconds `56.00`, LSTM `0.8334`, delta `+0.0713`
- tick `63651`, seconds `59.50`, LSTM `0.9379`, delta `+0.0620`
- tick `63331`, seconds `54.50`, LSTM `0.5671`, delta `-0.0522`
- tick `62787`, seconds `46.00`, LSTM `0.7730`, delta `+0.0514`
- tick `62595`, seconds `43.00`, LSTM `0.7465`, delta `+0.0398`
- tick `60643`, seconds `12.50`, LSTM `0.7133`, delta `-0.0393`
- tick `63235`, seconds `53.00`, LSTM `0.6554`, delta `-0.0367`

## Top 15 local ridge features

- `lag_06__T4__flash_duration`: coefficient `0.001953`, |coef| `0.001953`
- `lag_00__CT_place_HOLE`: coefficient `0.001790`, |coef| `0.001790`
- `lag_06__T_flashed_players`: coefficient `0.001443`, |coef| `0.001443`
- `lag_02__CT_place_ARAMP`: coefficient `-0.001274`, |coef| `0.001274`
- `lag_00__T4__flash_duration`: coefficient `-0.001259`, |coef| `0.001259`
- `lag_06__CT3__flash_duration`: coefficient `0.001259`, |coef| `0.001259`
- `lag_00__T_flashed_players`: coefficient `-0.001233`, |coef| `0.001233`
- `lag_06__T_flash_duration_sum`: coefficient `0.001069`, |coef| `0.001069`
- `lag_02__CT3__flash_duration`: coefficient `-0.001042`, |coef| `0.001042`
- `lag_05__CT2__flash_duration`: coefficient `0.001031`, |coef| `0.001031`
- `lag_00__CT_kills_last_3s`: coefficient `0.001014`, |coef| `0.001014`
- `lag_02__T_place_UPPERTUNNEL`: coefficient `-0.000962`, |coef| `0.000962`
- `lag_06__T_place_UPPERTUNNEL`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_12__CT4__duck_amount`: coefficient `-0.000888`, |coef| `0.000888`
- `lag_05__T_place_UPPERTUNNEL`: coefficient `-0.000866`, |coef| `0.000866`

## Top 10 utility ridge features

- `lag_06__T4__flash_duration`: coefficient `0.001953` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001259` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `0.001259` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.001069` (raises CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.001042` (lowers CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `0.001031` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000743` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000742` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000713` (lowers CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000692` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_HOLE`: coefficient `0.001790` (raises CT win probability)
- `lag_06__T_flashed_players`: coefficient `0.001443` (raises CT win probability)
- `lag_02__CT_place_ARAMP`: coefficient `-0.001274` (lowers CT win probability)
- `lag_00__T_flashed_players`: coefficient `-0.001233` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001014` (raises CT win probability)
- `lag_02__T_place_UPPERTUNNEL`: coefficient `-0.000962` (lowers CT win probability)
- `lag_06__T_place_UPPERTUNNEL`: coefficient `-0.000894` (lowers CT win probability)
- `lag_12__CT4__duck_amount`: coefficient `-0.000888` (lowers CT win probability)
- `lag_05__T_place_UPPERTUNNEL`: coefficient `-0.000866` (lowers CT win probability)
- `lag_09__T_flashed_players`: coefficient `0.000865` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `63395`, seconds `55.50`, LSTM delta `+0.1993`

Top all feature movements:
- `lag_06__T4__flash_duration`: contribution `+0.015461`
- `lag_06__T_flashed_players`: contribution `+0.013922`
- `lag_06__CT3__flash_duration`: contribution `+0.009447`
- `lag_00__T4__flash_duration`: contribution `+0.009429`
- `lag_02__CT_place_ARAMP`: contribution `+0.007939`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.015461`
- `lag_06__CT3__flash_duration`: contribution `+0.009447`
- `lag_00__T4__flash_duration`: contribution `+0.009429`
- `lag_05__CT2__flash_duration`: contribution `+0.005952`
- `lag_06__T_flash_duration_sum`: contribution `+0.005563`

### tick `63587`, seconds `58.50`, LSTM delta `-0.0987`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.019978`
- `lag_06__T4__flash_duration`: contribution `-0.014626`
- `lag_06__T_flashed_players`: contribution `-0.008353`
- `lag_04__T_flashed_players`: contribution `-0.003981`
- `lag_00__kill_diff_last_3s`: contribution `-0.003877`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `-0.014626`
- `lag_06__T_flash_duration_sum`: contribution `-0.003847`
- `lag_04__T_flash_duration_sum`: contribution `-0.003671`
- `lag_04__T4__flash_duration`: contribution `-0.002769`
- `lag_04__T3__flash_duration`: contribution `-0.002137`

### tick `63491`, seconds `57.00`, LSTM delta `+0.0956`

Top all feature movements:
- `lag_09__T_flashed_players`: contribution `+0.008346`
- `lag_09__T4__flash_duration`: contribution `+0.005277`
- `lag_02__CT3__flash_duration`: contribution `+0.004883`
- `lag_01__T_flashed_players`: contribution `-0.004385`
- `lag_05__T_place_UPPERTUNNEL`: contribution `+0.003983`

Top utility-only movements:
- `lag_09__T4__flash_duration`: contribution `+0.005277`
- `lag_02__CT3__flash_duration`: contribution `+0.004883`
- `lag_03__T4__flash_duration`: contribution `+0.003826`
- `lag_01__T4__flash_duration`: contribution `-0.003211`
- `lag_09__CT3__flash_duration`: contribution `+0.002965`

### tick `63427`, seconds `56.00`, LSTM delta `+0.0713`

Top all feature movements:
- `lag_06__T_flashed_players`: contribution `-0.005569`
- `lag_01__T_flashed_players`: contribution `+0.004385`
- `lag_07__CT3__flash_duration`: contribution `+0.004308`
- `lag_05__CT2__is_scoped`: contribution `-0.004307`
- `lag_03__CT_place_ARAMP`: contribution `+0.003850`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.004308`
- `lag_01__T4__flash_duration`: contribution `+0.003760`
- `lag_07__T4__flash_duration`: contribution `+0.003257`
- `lag_06__CT2__flash_duration`: contribution `+0.002763`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001748`

### tick `63651`, seconds `59.50`, LSTM delta `+0.0620`

Top all feature movements:
- `lag_06__T4__flash_duration`: contribution `+0.012490`
- `lag_06__T_flashed_players`: contribution `+0.008353`
- `lag_06__T_flash_duration_sum`: contribution `+0.007045`
- `lag_02__CT_place_HOLE`: contribution `+0.004355`
- `lag_06__T2__flash_duration`: contribution `+0.003910`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.012490`
- `lag_06__T_flash_duration_sum`: contribution `+0.007045`
- `lag_06__T2__flash_duration`: contribution `+0.003910`
- `lag_06__CT3__flash_duration`: contribution `+0.002880`
- `lag_07__CT3__flash_duration`: contribution `-0.002690`
