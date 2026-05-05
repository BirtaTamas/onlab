# Flash Ablation Conclusion

Ez az osszegzes a harom vegleges modell test eredmenyeit hasonlitja ossze:

- no-utility modell
- full utility modell
- utility modell flash feature-ok nelkul

## Globalis metrikak

| modell | feature count | accuracy | precision | recall | F1 | Brier | logloss | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no utility | 412 | 0.762628 | 0.733641 | 0.819891 | 0.774372 | 0.151830 | 0.450936 | 0.860441 |
| full utility | 531 | 0.762779 | 0.732923 | 0.822091 | 0.774951 | 0.151855 | 0.451270 | 0.860574 |
| utility flash nelkul | 498 | 0.763613 | 0.733686 | 0.822895 | 0.775734 | 0.151926 | 0.451494 | 0.860587 |

## Globalis ertelmezes

A flash nelkuli utility modell accuracy, recall, F1 es ROC AUC alapjan kicsit jobb lett, mint a no-utility es a full utility modell.

Viszont Brier es logloss alapjan rosszabb lett. Ez azt jelenti, hogy a `0.5` threshold melletti class dontesekben picit jobban teljesit, de probability kalibracio / valoszinusegi pontossag szempontbol nem javult.

Ezert nem mondhato, hogy globalisan egyertelmuen jobb modell lett. A korrekt allitas az, hogy a flash nelkuli utility feature-keszlet a class dontesekben tisztabb jelet adott, de probability szinten tovabbra sem eros globalis javulas.

## Class flip osszehasonlitas

Itt csak azok a snapshotok szamitanak, ahol a utility modell es a no-utility modell `0.5` threshold mellett mas class-t prediktalt.

| osszehasonlitas | flip sorok | utility jo | no-utility jo | kulonbseg | utility win rate |
|---|---:|---:|---:|---:|---:|
| full utility vs no utility | 13781 | 6951 | 6830 | +121 | 50.44% |
| utility flash nelkul vs no utility | 14855 | 7822 | 7033 | +789 | 52.66% |

## Aktiv utility helyzetek

| csoport | full utility win rate | flash nelkuli utility win rate |
|---|---:|---:|
| osszes flip | 50.44% | 52.66% |
| active/recent utility | 50.31% | 52.89% |
| strong utility action | 50.49% | 52.97% |
| utility damage | 51.35% | 51.92% |
| active smoke/inferno | 51.08% | 53.36% |
| recent utility last 5s | 50.20% | 51.95% |
| flash effect jelen van | 45.93% | 49.54% |

## Mit mutat a flash nelkuli modell?

A flash feature-ok kidobasa utan az atbilleneses elemzes sokkal kedvezobb lett:

- osszes flip: `50.44%` -> `52.66%`
- active/recent utility: `50.31%` -> `52.89%`
- strong utility action: `50.49%` -> `52.97%`
- active smoke/inferno: `51.08%` -> `53.36%`

Ez arra utal, hogy a flash feature-ok a class flip elemzesben zajosithattak a utility hatast.

Kozben a `flash effect` jelenlete mellett a win rate `45.93%`-rol `49.54%`-ra javult, pedig a flash feature-ok mar nincsenek a modellben. Ez is azt tamasztja ala, hogy a flash jellegu feature-ok onmagukban nem adtak stabil, jol hasznalhato jelet.

## Vegso konkluzio

A utility feature-ok haszna nem globalis metrika-javulaskent jelenik meg erosen. A full utility modell probability metrikakban nem javitott egyertelmuen a no-utility modellhez kepest.

Ugyanakkor a flash feature-ok eltavolitasa utan a megmarado utility feature-ok, foleg az active smoke/inferno, utility damage es recent utility jellegu feature-ok, jobban magyarazhato class flip hatast mutatnak.

Ez alapjan a legerosebb allitas:

> A utility feature-ok hatasa kontextusfuggo, es nem minden utility tipus egyforman hasznos. A flash jellegu feature-ok zajosabbnak bizonyultak, mig a flash nelkuli utility modell az atbilleneses vizsgalatban tisztabb pozitiv jelet adott, kulonosen aktiv smoke/inferno es utility damage helyzetekben.

Dolgozatban/prezentacioban erdemes ezt ugy megfogalmazni, hogy a utility nem altalanos, nagy modellezesi javulas, hanem bizonyos taktikai helyzetekben megjeleno plusz informacio.
