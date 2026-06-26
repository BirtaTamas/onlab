# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `22`

## Largest probability jumps

- tick `199526`, seconds `99.00`, LSTM `0.5981`, delta `+0.3948`
- tick `199622`, seconds `100.50`, LSTM `0.8557`, delta `+0.2977`
- tick `199430`, seconds `97.50`, LSTM `0.3169`, delta `-0.2479`
- tick `199398`, seconds `97.00`, LSTM `0.5648`, delta `+0.1923`
- tick `198470`, seconds `82.50`, LSTM `0.3120`, delta `-0.0967`
- tick `199494`, seconds `98.50`, LSTM `0.2033`, delta `-0.0906`
- tick `199334`, seconds `96.00`, LSTM `0.3963`, delta `-0.0671`
- tick `198566`, seconds `84.00`, LSTM `0.3464`, delta `+0.0669`
- tick `199942`, seconds `105.50`, LSTM `0.9577`, delta `+0.0655`
- tick `198662`, seconds `85.50`, LSTM `0.3381`, delta `-0.0396`

## Top 15 local ridge features

- `lag_04__CT4__flash_duration`: coefficient `0.003078`, |coef| `0.003078`
- `lag_00__kill_diff_last_3s`: coefficient `0.002612`, |coef| `0.002612`
- `lag_00__CT4__flash_duration`: coefficient `0.002540`, |coef| `0.002540`
- `lag_04__CT5__flash_duration`: coefficient `0.002294`, |coef| `0.002294`
- `lag_00__T_place_ROOF`: coefficient `-0.002137`, |coef| `0.002137`
- `lag_10__T_place_SQUEAKY`: coefficient `-0.002085`, |coef| `0.002085`
- `lag_09__T_place_SQUEAKY`: coefficient `-0.002054`, |coef| `0.002054`
- `lag_00__T4__is_scoped`: coefficient `-0.002018`, |coef| `0.002018`
- `lag_00__CT_kills_last_3s`: coefficient `0.001977`, |coef| `0.001977`
- `lag_00__damage_diff_last_5s`: coefficient `0.001890`, |coef| `0.001890`
- `lag_04__CT_flash_duration_sum`: coefficient `0.001857`, |coef| `0.001857`
- `lag_03__CT4__flash_duration`: coefficient `-0.001785`, |coef| `0.001785`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001660`, |coef| `0.001660`
- `lag_15__T_place_SQUEAKY`: coefficient `0.001611`, |coef| `0.001611`
- `lag_01__CT5__flash_duration`: coefficient `-0.001550`, |coef| `0.001550`

## Top 10 utility ridge features

- `lag_04__CT4__flash_duration`: coefficient `0.003078` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.002540` (raises CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `0.002294` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001857` (raises CT win probability)
- `lag_03__CT4__flash_duration`: coefficient `-0.001785` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001660` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001550` (lowers CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001483` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.001379` (raises CT win probability)
- `lag_07__CT4__flash_duration`: coefficient `0.001329` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002612` (raises CT win probability)
- `lag_00__T_place_ROOF`: coefficient `-0.002137` (lowers CT win probability)
- `lag_10__T_place_SQUEAKY`: coefficient `-0.002085` (lowers CT win probability)
- `lag_09__T_place_SQUEAKY`: coefficient `-0.002054` (lowers CT win probability)
- `lag_00__T4__is_scoped`: coefficient `-0.002018` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001977` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001890` (raises CT win probability)
- `lag_15__T_place_SQUEAKY`: coefficient `0.001611` (raises CT win probability)
- `lag_04__CT_flashed_players`: coefficient `0.001549` (raises CT win probability)
- `lag_11__CT_place_HEAVEN`: coefficient `0.001549` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `199526`, seconds `99.00`, LSTM delta `+0.3948`

Top all feature movements:
- `lag_04__CT4__flash_duration`: contribution `+0.022068`
- `lag_04__CT5__flash_duration`: contribution `+0.016932`
- `lag_03__CT4__flash_duration`: contribution `+0.012799`
- `lag_09__T_place_SQUEAKY`: contribution `+0.012786`
- `lag_04__CT_flash_duration_sum`: contribution `+0.012277`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.022068`
- `lag_04__CT5__flash_duration`: contribution `+0.016932`
- `lag_03__CT4__flash_duration`: contribution `+0.012799`
- `lag_04__CT_flash_duration_sum`: contribution `+0.012277`
- `lag_01__CT4__flash_duration`: contribution `+0.009923`

### tick `199622`, seconds `100.50`, LSTM delta `+0.2977`

Top all feature movements:
- `lag_04__CT4__flash_duration`: contribution `+0.022152`
- `lag_10__T_place_SQUEAKY`: contribution `+0.012984`
- `lag_00__kill_diff_last_3s`: contribution `+0.012575`
- `lag_07__CT5__flash_duration`: contribution `+0.010946`
- `lag_15__T_place_SQUEAKY`: contribution `+0.010027`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.022152`
- `lag_07__CT5__flash_duration`: contribution `+0.010946`
- `lag_07__CT4__flash_duration`: contribution `+0.009526`
- `lag_07__CT_flash_duration_sum`: contribution `+0.007504`
- `lag_06__CT4__flash_duration`: contribution `+0.005004`

### tick `199430`, seconds `97.50`, LSTM delta `-0.2479`

Top all feature movements:
- `lag_00__CT4__flash_duration`: contribution `-0.018209`
- `lag_10__T_place_SQUEAKY`: contribution `-0.012984`
- `lag_09__T_place_SQUEAKY`: contribution `-0.012786`
- `lag_01__CT5__flash_duration`: contribution `-0.011443`
- `lag_14__CT_place_VENTS`: contribution `-0.010131`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.018209`
- `lag_01__CT5__flash_duration`: contribution `-0.011443`
- `lag_01__CT4__flash_duration`: contribution `+0.009885`
- `lag_01__CT_flash_duration_sum`: contribution `-0.006631`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005325`

### tick `199398`, seconds `97.00`, LSTM delta `+0.1923`

Top all feature movements:
- `lag_00__CT4__flash_duration`: contribution `+0.018209`
- `lag_09__T_place_SQUEAKY`: contribution `-0.012786`
- `lag_00__CT_flash_duration_sum`: contribution `+0.010977`
- `lag_00__T4__is_scoped`: contribution `+0.009373`
- `lag_00__CT5__flash_duration`: contribution `+0.008835`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.018209`
- `lag_00__CT_flash_duration_sum`: contribution `+0.010977`
- `lag_00__CT5__flash_duration`: contribution `+0.008835`

### tick `198470`, seconds `82.50`, LSTM delta `-0.0967`

Top all feature movements:
- `lag_00__T_place_ROOF`: contribution `-0.012105`
- `lag_15__T_place_SQUEAKY`: contribution `+0.010027`
- `lag_10__CT_place_VENTS`: contribution `-0.005443`
- `lag_14__T1__duck_amount`: contribution `-0.004421`
- `lag_03__CT_place_GARAGE`: contribution `-0.004315`

Top utility-only movements:
- `lag_13__CT_B_site_active_infernos`: contribution `-0.002169`
- `lag_13__CT_A_site_active_infernos`: contribution `-0.001525`
