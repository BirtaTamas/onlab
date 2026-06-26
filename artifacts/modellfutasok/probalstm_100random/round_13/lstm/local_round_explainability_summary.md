# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `72881`, seconds `65.00`, LSTM `0.6888`, delta `+0.3344`
- tick `73169`, seconds `69.50`, LSTM `0.1881`, delta `-0.2631`
- tick `72913`, seconds `65.50`, LSTM `0.4623`, delta `-0.2265`
- tick `73041`, seconds `67.50`, LSTM `0.4297`, delta `+0.2257`
- tick `72977`, seconds `66.50`, LSTM `0.2454`, delta `-0.2175`
- tick `72785`, seconds `63.50`, LSTM `0.3973`, delta `-0.1018`
- tick `72753`, seconds `63.00`, LSTM `0.4991`, delta `-0.0993`
- tick `73201`, seconds `70.00`, LSTM `0.1242`, delta `-0.0639`
- tick `73009`, seconds `67.00`, LSTM `0.2041`, delta `-0.0413`
- tick `73617`, seconds `76.50`, LSTM `0.0782`, delta `-0.0399`

## Top 15 local ridge features

- `lag_07__T_place_WALKWAY`: coefficient `-0.003096`, |coef| `0.003096`
- `lag_01__T_place_WALKWAY`: coefficient `-0.002871`, |coef| `0.002871`
- `lag_00__T_place_HEAVEN`: coefficient `-0.002781`, |coef| `0.002781`
- `lag_13__T_place_WALKWAY`: coefficient `-0.002215`, |coef| `0.002215`
- `lag_04__T_place_WALKWAY`: coefficient `0.002192`, |coef| `0.002192`
- `lag_06__CT_place_TUNNELSTAIRS`: coefficient `-0.001909`, |coef| `0.001909`
- `lag_08__CT_place_TUNNELSTAIRS`: coefficient `-0.001885`, |coef| `0.001885`
- `lag_05__T_place_HEAVEN`: coefficient `-0.001878`, |coef| `0.001878`
- `lag_15__T_place_HEAVEN`: coefficient `-0.001840`, |coef| `0.001840`
- `lag_06__CT5__flash_duration`: coefficient `0.001590`, |coef| `0.001590`
- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `0.001474`, |coef| `0.001474`
- `lag_08__T_place_HEAVEN`: coefficient `-0.001468`, |coef| `0.001468`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `-0.001437`, |coef| `0.001437`
- `lag_01__CT_place_BRICKS`: coefficient `0.001419`, |coef| `0.001419`
- `lag_10__T_place_WALKWAY`: coefficient `0.001371`, |coef| `0.001371`

## Top 10 utility ridge features

- `lag_06__CT5__flash_duration`: coefficient `0.001590` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `-0.000933` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.000814` (raises CT win probability)
- `lag_06__CT_flash_duration_sum`: coefficient `0.000656` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.000627` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.000578` (raises CT win probability)
- `lag_07__CT_flash_duration_sum`: coefficient `-0.000577` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000568` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000568` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000534` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_WALKWAY`: coefficient `-0.003096` (lowers CT win probability)
- `lag_01__T_place_WALKWAY`: coefficient `-0.002871` (lowers CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.002781` (lowers CT win probability)
- `lag_13__T_place_WALKWAY`: coefficient `-0.002215` (lowers CT win probability)
- `lag_04__T_place_WALKWAY`: coefficient `0.002192` (raises CT win probability)
- `lag_06__CT_place_TUNNELSTAIRS`: coefficient `-0.001909` (lowers CT win probability)
- `lag_08__CT_place_TUNNELSTAIRS`: coefficient `-0.001885` (lowers CT win probability)
- `lag_05__T_place_HEAVEN`: coefficient `-0.001878` (lowers CT win probability)
- `lag_15__T_place_HEAVEN`: coefficient `-0.001840` (lowers CT win probability)
- `lag_12__CT_place_TUNNELSTAIRS`: coefficient `0.001474` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `72881`, seconds `65.00`, LSTM delta `+0.3344`

Top all feature movements:
- `lag_04__T_place_WALKWAY`: contribution `+0.059621`
- `lag_01__T_place_WALKWAY`: contribution `+0.039038`
- `lag_00__T_place_HEAVEN`: contribution `+0.034128`
- `lag_01__CT_place_BRICKS`: contribution `+0.027254`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `+0.026883`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `+0.011248`
- `lag_06__CT_flash_duration_sum`: contribution `+0.002141`

### tick `73169`, seconds `69.50`, LSTM delta `-0.2631`

Top all feature movements:
- `lag_13__T_place_WALKWAY`: contribution `-0.060240`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `+0.026883`
- `lag_10__T_place_WALKWAY`: contribution `-0.018640`
- `lag_08__T_place_HEAVEN`: contribution `-0.018011`
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `-0.017445`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.011248`
- `lag_15__CT5__flash_duration`: contribution `-0.003777`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002141`

### tick `72913`, seconds `65.50`, LSTM delta `-0.2265`

Top all feature movements:
- `lag_00__T_place_HEAVEN`: contribution `-0.034128`
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `-0.026883`
- `lag_02__CT_place_BRICKS`: contribution `-0.019222`
- `lag_00__CT_place_BRICKS`: contribution `-0.017498`
- `lag_13__CT_place_TUNNELSTAIRS`: contribution `-0.016502`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `-0.006602`

### tick `73041`, seconds `67.50`, LSTM delta `+0.2257`

Top all feature movements:
- `lag_07__T_place_WALKWAY`: contribution `+0.042103`
- `lag_04__T_place_WALKWAY`: contribution `-0.029811`
- `lag_05__T_place_HEAVEN`: contribution `+0.023041`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `+0.020243`
- `lag_10__CT_place_TUNNELSTAIRS`: contribution `+0.018737`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.004434`
- `lag_11__CT5__flash_duration`: contribution `+0.004017`

### tick `72977`, seconds `66.50`, LSTM delta `-0.2175`

Top all feature movements:
- `lag_07__T_place_WALKWAY`: contribution `-0.084206`
- `lag_04__T_place_WALKWAY`: contribution `-0.029811`
- `lag_08__CT_place_TUNNELSTAIRS`: contribution `-0.026556`
- `lag_05__T_place_HEAVEN`: contribution `-0.023041`
- `lag_02__CT_place_BRICKS`: contribution `+0.019222`

Top utility-only movements:
- `lag_00__CT5__flash_duration`: contribution `-0.005759`
- `lag_09__CT5__flash_duration`: contribution `-0.003519`
