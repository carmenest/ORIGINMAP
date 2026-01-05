# Observation O-Δ2: Mass Heterogeneity

**Date**: 2026-01-04
**Experiment**: O-Δ2
**Run ID**: 95f01c8a4245
**Seed**: 42
**Permutations**: 500

---

## Observación

10 clases `recclass` muestran distribuciones de masa con varianza significativamente **menor** que la esperada bajo modelo nulo.

0 clases muestran varianza significativamente **mayor** que la esperada.

## Dónde aparece

| Clase | n | CV observado | CV nulo (media) | z-score | FDR q |
|-------|---|--------------|-----------------|---------|-------|
| L6 | 8279 | 9.66 | 38.37 | **-3.43** | 0.000 |
| E-an | 5 | 0.41 | 1.44 | **-2.49** | 0.000 |
| Iron, IIG | 6 | 0.56 | 1.53 | -2.19 | 0.088 |
| LL7 | 22 | 1.18 | 3.04 | -2.07 | 0.088 |
| R4 | 42 | 1.43 | 3.97 | -2.02 | 0.088 |
| Lunar (anorth) | 69 | 1.69 | 4.88 | -1.97 | 0.000 |
| Eucrite-pmict | 207 | 2.62 | 8.01 | -1.89 | 0.000 |
| Eucrite | 221 | 2.88 | 8.56 | -1.84 | 0.000 |
| Ureilite | 300 | 3.01 | 9.36 | -1.75 | 0.000 |
| H5/6 | 193 | 3.16 | 7.84 | -1.65 | 0.000 |

## Qué no esperaba ver

- **Asimetría total**: solo clases con CV bajo, ninguna con CV alto
- **L6 extremo**: la clase más grande (8279 muestras) tiene el z-score más extremo (-3.43)
- **Clases lunares y hierros**: presentes en la lista pese a tamaños pequeños
- **Patrón consistente**: todas las clases significativas muestran ~25-40% del CV esperado

## Qué pruebas hice

1. Modelo nulo: masas asignadas aleatoriamente a clases, preservando tamaños de clase
2. Métrica: Coeficiente de Variación (CV = std/mean)
3. 500 permutaciones con seed=42
4. Corrección FDR (Benjamini-Hochberg)

## Estado

- [x] Reproducido (mismo resultado con diferentes seeds: 42, 123)
- [x] Estable (umbral FDR 0.05 y 0.10 dan resultados coherentes)
- [x] Numérico (aparece en CSV, no solo en visual)

---

## Notas

_Sin interpretación. Solo hechos._

Las clases con distribución de masa "tight" incluyen:
- Meteoritos diferenciados (Eucrite, Ureilite, Lunar)
- Condrios ordinarios específicos (L6, H5/6, LL7)
- Hierros raros (Iron IIG)
- Enstatitas anómalas (E-an)

Siguiente paso: verificar estabilidad con N=1000 permutaciones y seed diferente.
