# Observation O-Δ3: Stratified Null Model — H-CONS Test

**Date**: 2026-01-04
**Experiment**: O-Δ3
**Hypothesis**: H-CONS (Constrained Formation)
**Seeds tested**: 42, 999
**Permutations**: 500 each
**Mass bins**: 10 quantiles

---

## Observación

El modelo nulo estratificado por bins de masa **separa estructura genuina de artefacto**.

De las 10 clases significativas en O-Δ2:
- **3 sobreviven** bajo el nulo más duro
- **7 caen** (eran artefacto de muestreo por rango de masa)

## Qué sobrevive (estructura genuina)

| Clase | n | z-score (seed 42) | z-score (seed 999) | Estable? |
|-------|---|-------------------|-------------------|----------|
| **L6** | 8279 | **-3.21** | **-3.15** | **SÍ** |
| **H6** | 4527 | -1.86 | -1.91 | **SÍ** |
| **Ureilite** | 300 | -1.71 | -1.67 | **SÍ** |

## Qué cae (era artefacto)

| Clase | n | z (O-Δ2) | z (O-Δ3) | q (O-Δ3) |
|-------|---|----------|----------|----------|
| LL7 | 22 | -2.07 | -0.42 | 0.874 |
| H/L6 | 11 | -1.96 | +0.01 | 0.980 |
| Lunar (anorth) | 69 | -1.97 | -0.93 | 0.778 |
| R4 | 42 | -2.02 | -1.38 | 0.440 |

## Casos borderline (sensibles al seed)

| Clase | Seed 42 | Seed 999 |
|-------|---------|----------|
| Eucrite | q=0.24 (cae) | q=0.088 (sobrevive) |
| H4 | q=0.176 (cae) | q=0.088 (sobrevive) |
| Eucrite-pmict | q=0.176 (cae) | q=0.176 (cae) |

## Qué NO esperaba ver

1. **L6 resiste con z=-3.2** — el efecto más extremo sobrevive el nulo más duro
2. **Ureilite confirma** — pequeña clase (n=300) con estructura genuina
3. **LL7 y H/L6 colapsan completamente** — z-scores cercanos a 0 bajo nulo estratificado
4. **Asimetría conservada** — solo clases "tight", ninguna "dispersed" genuina

## Qué pruebas hice

1. **Null-1 (O-Δ2)**: Permuta masas globalmente
2. **Null-2 (O-Δ3)**: Permuta masas solo dentro de cada bin de cuantil
3. Comparé qué sobrevive al pasar de Null-1 a Null-2
4. Verifiqué con dos seeds diferentes (42, 999)

## Modelo nulo estratificado

```
Preserva:
├── Tamaños de clase
├── Distribución global de masa
└── Distribución de masa POR BIN (nuevo)

Rompe:
└── Asociación específica masa↔clase DENTRO de cada bin
```

## Estado

- [x] Reproducido (L6, H6, Ureilite estables en ambos seeds)
- [x] Estable (efecto sobrevive nulo más duro)
- [x] Numérico (CSV con 220 clases, métricas exactas)

---

## Conclusión factual (sin interpretación física)

**L6, H6, y Ureilite** presentan distribuciones de masa con varianza significativamente menor que la esperada bajo un modelo nulo que preserva:
- Los tamaños de clase
- La distribución de masa por cuantiles

Este resultado **no** se explica por:
- Muestreo de rangos de masa restringidos
- Tamaño de clase
- Artefactos de catalogación simple

**Sí** es compatible con:
- Restricciones intrínsecas de escala de masa
- Procesos de formación/fragmentación regulados
- Estructura latente no trivial

---

## Archivos generados

```
reports/
├── hypothesis_O-Δ3_results.csv
├── hypothesis_O-Δ3_summary.json
└── hypothesis_O-Δ3_plot.png
```
