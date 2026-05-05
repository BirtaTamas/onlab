# Class Flip Only Summary

Ez az osszegzes csak azt mutatja, amikor a with-utility es no-utility modell `0.5` threshold mellett mas class-t prediktal.

Itt nem szamitjuk azt, amikor ugyanabban a classban maradnak, de a probability jobb vagy rosszabb lesz.

## Osszesites

- Eltéro class predikcioju snapshot: `13781`
- Utility modell jo, no-utility rossz: `6951`
- No-utility modell jo, utility rossz: `6830`
- Utility javara kulonbseg: `+121`
- Utility nyeresi arany az atbillenesek kozott: `50.44%`

## Mit jelent az atbillenes?

Pelda jo iranyra:

`p_no_utility = 0.439`, `p_with_utility = 0.532`, `ct_win = 1`

Itt a no-utility modell T wint prediktalt, a utility modell CT wint prediktalt, es a CT tenyleg nyert.

Pelda rossz iranyra:

`p_no_utility = 0.488`, `p_with_utility = 0.552`, `ct_win = 0`

Itt a no-utility modell T wint prediktalt, a utility modell CT wint prediktalt, de a T nyert.

## Legnagyobb jo iranyu atbillenesek

| meccs | round | ido | ct_win | no utility | with utility | delta |
|---|---:|---:|---:|---:|---:|---:|
| `heroic-vs-saw-m2-train.csv` | 8 | 26.0 | 1 | 0.439 | 0.532 | +0.093 |
| `vitality-vs-legacy-m2-dust2.csv` | 11 | 23.0 | 1 | 0.434 | 0.514 | +0.080 |
| `vitality-vs-legacy-m2-dust2.csv` | 11 | 22.5 | 1 | 0.434 | 0.514 | +0.080 |
| `heroic-vs-saw-m2-train.csv` | 8 | 25.5 | 1 | 0.457 | 0.536 | +0.080 |
| `vitality-vs-legacy-m2-dust2.csv` | 11 | 22.0 | 1 | 0.474 | 0.552 | +0.078 |

## Legnagyobb rossz iranyu atbillenesek

| meccs | round | ido | ct_win | no utility | with utility | delta |
|---|---:|---:|---:|---:|---:|---:|
| `g2-vs-fluxo-m3-mirage.csv` | 10 | 142.5 | 0 | 0.488 | 0.552 | +0.064 |
| `spirit-vs-heroic-m3-mirage.csv` | 3 | 31.0 | 0 | 0.442 | 0.502 | +0.060 |
| `legacy-vs-lynn-vision-m2-inferno.csv` | 15 | 87.5 | 0 | 0.472 | 0.531 | +0.059 |
| `lynn-vision-vs-3dmax-m3-train.csv` | 7 | 87.5 | 0 | 0.458 | 0.517 | +0.059 |
| `spirit-vs-heroic-m3-mirage.csv` | 3 | 32.0 | 0 | 0.460 | 0.517 | +0.057 |

## Dolgozatba javasolt mondat

A utility feature-ok hatasa a fix `0.5` threshold melletti atbillenesekben is vizsgalhato. A teszt adathalmazon `13781` olyan snapshot volt, ahol a ket modell masik class-t prediktalt. Ezek kozul `6951` esetben a utility modell predikcioja volt helyes, `6830` esetben pedig a no-utility modellé. Ez azt mutatja, hogy a utility feature-ok sok hatarhelyzetben kepesek megvaltoztatni a modell donteset, de az atbillenesek iranya kozel szimmetrikus, tehat a utility hatasa nem altalanos, hanem kontextusfuggo.
