# Observation H-FRAG: Fragmentation Uniformity Hypothesis Test

**Date**: 2026-01-04
**Hypothesis**: H-FRAG (Grade 6 metamorphism → uniform fragmentation)
**Predictions tested**: 5
**N permutations**: 500
**Seed**: 42

---

## Hipótesis Original

> "El metamorfismo de grado 6 produce propiedades mecánicas más uniformes,
> resultando en fragmentación más predecible."

## Resultado: FALSIFICADA (pero con hallazgo importante)

---

## Predicciones y Resultados

### P1: Grade 6 < Grade 5 en CV para L, H, LL — ✓ SUPPORTED

| Tipo | CV Grade 5 | CV Grade 6 | Reducción |
|------|------------|------------|-----------|
| L    | 18.43      | 9.66       | -47%      |
| H    | 23.62      | 9.87       | -58%      |
| LL   | 18.83      | 10.81      | -43%      |

**Los tres tipos muestran Grade 6 < Grade 5**

### P2: Tendencia monotónica (3→4→5→6) — ✗ FALSIFIED

```
L:  CV = 14.10 → 7.60 → 18.43 → 9.66
H:  CV =  9.20 → 9.88 → 23.62 → 9.87
LL: CV =  2.80 → 6.43 → 18.83 → 10.81
```

**HALLAZGO**: Grade 5 es un PICO, no un valor intermedio

### P3: Efecto en Falls AND Finds — ✗ FALSIFIED

```
Falls:  L (✗), H (✓)
Finds:  L (✓), H (✗)
```

**Inconsistente entre subpoblaciones**

### P4: Achondrites siguen patrón diferente — DESCRIPTIVE

- Mean CV achondrites: 3.43
- Mean CV chondrite Grade 6: 10.11

**Achondrites tienen CVs más bajos en general**

### P5: Sobrevive test de permutación — ✗ FALSIFIED

| Tipo | CV6-CV5 obs | p-value | Significativo |
|------|-------------|---------|---------------|
| L    | -8.77       | 0.008   | ✓             |
| H    | -13.75      | 0.462   | ✗             |

**Solo L sobrevive**

---

## Síntesis

La hipótesis H-FRAG es **demasiado simple**. El patrón observado no es:

```
metamorfismo ↑ → varianza ↓ (monotónico)
```

El patrón REAL es:

```
           CV
           ↑
    20 ─┐  │      ★ Grade 5 (pico)
        │  │     ╱╲
    15 ─┤  │    ╱  ╲
        │  │   ╱    ╲
    10 ─┤  │──╱──────╲── Grade 3, 4, 6 (bajo)
        │  │
     5 ─┤  │
        │  │
     0 ─┴──┴────────────→ Grado
           3   4   5   6
```

---

## Hipótesis Emergente: H-FRAG-2

> **Grade 5 representa un régimen de fragmentación anómalo**,
> posiblemente el punto donde el metamorfismo parcial crea
> **heterogeneidad mecánica máxima** antes de la homogeneización
> final en Grade 6.

Interpretación física posible:
- Grades 3-4: Poco procesado, matriz primordial relativamente uniforme
- Grade 5: Metamorfismo parcial crea **gradientes** y **discontinuidades**
- Grade 6: Metamorfismo completo **homogeneiza** de nuevo

---

## Archivos Generados

```
reports/
├── hypothesis_H-FRAG_results.json
└── hypothesis_H-FRAG_plot.png
```

---

## Estado

- [x] 5 predicciones falsificables diseñadas
- [x] Todas testeadas
- [x] Hipótesis original FALSIFICADA
- [x] Nueva hipótesis (H-FRAG-2) emergida
- [ ] H-FRAG-2 pendiente de test formal

---

## Conclusión

H-FRAG **falló** como hipótesis predictiva (3/5 predicciones falsificadas), pero el proceso de falsificación reveló un patrón más interesante: **Grade 5 como anomalía**, no Grade 6.

Esto es exactamente cómo debe funcionar el método científico.
