# Diccionario de datos — USROP

Fuente: [github.com/AndrzejTunkiel/USROP](https://github.com/AndrzejTunkiel/USROP)
(rama `master`, revisión de dataset "A"). Publicación de origen:

> Tunkiel, A.T., Sui, D., Wiktorski, T. (2021). *Reference dataset for rate of
> penetration benchmarking*. Journal of Petroleum Science and Engineering, 196,
> 108069.

Ver también [docs/adr/001-dataset-selection.md](adr/001-dataset-selection.md).
Valores de rango y missing values de esta página fueron calculados directamente
sobre los 7 CSV descargados vía [`ml/data/download_usrop.py`](../ml/data/download_usrop.py),
no tomados de la publicación.

## Convención de nombres de archivo

```
USROP_A{revisión} {índice_de_pozo} {código_de_formación}.csv
```

Ejemplo: `USROP_A 0 N-NA_F-9_Ad.csv` → revisión `A`, pozo índice `0`, código de
formación `N-NA_F-9_Ad`.

El repositorio solo tiene la revisión `A`. El índice de pozo (`0`-`6`) identifica
cada uno de los 7 pozos. El código de formación que sigue (ej. `N-SH_F-14d`,
`N-S_F-7d`) es una etiqueta de litología/escenario definida por los autores; su
codificación completa está en la publicación de origen (con acceso restringido,
no se pudo verificar el detalle exacto de cada sufijo) — no se reinterpreta acá
para evitar documentar algo no verificado.

## Atributos (12 + índice)

Los CSV traen una columna de índice sin nombre (0..N-1 por pozo, no es un
atributo del dataset) y 12 columnas de datos, con el nombre y unidad ya
embebidos en el header original:

| Código | Columna en el CSV | Unidad | Descripción | Rango observado (min–max, los 7 pozos) |
|---|---|---|---|---|
| MD | `Measured Depth m` | m | Profundidad medida a lo largo del pozo | 225.17 – 4090.00 |
| WOB | `Weight on Bit kkgf` | kkgf (kilo-kgf) | Peso sobre el trépano | 0.0018 – 31.41 |
| SPP | `Average Standpipe Pressure kPa` | kPa | Presión promedio en standpipe | 1432.66 – 24998.46 |
| T | `Average Surface Torque kN.m` | kN·m | Torque promedio en superficie | 0.0081 – 36.49 |
| RPM | `Average Rotary Speed rpm` | rpm | Velocidad rotatoria promedio | 0.00 – 311.23 |
| FR | `Mud Flow In L/min` | L/min | Caudal de lodo de entrada | 185.42 – 4538.45 |
| DS | `Mud Density In g/cm3` | g/cm³ | Densidad del lodo de entrada | 1.02 – 12.02 |
| HD | `Diameter mm` | mm | Diámetro del hoyo/broca | 215.90 – 444.50 |
| HL | `Average Hookload kkgf` | kkgf | Hookload promedio | 84.05 – 152.93 |
| VD | `Hole Depth (TVD) m` | m | Profundidad vertical verdadera | 225.16 – 3248.39 |
| GR | `USROP Gamma gAPI` | gAPI | Rayos gamma | 0.00 – 260.90 |
| **ROP** (target) | `Rate of Penetration m/h` | m/h | Tasa de penetración — variable a predecir | 0.33 – 99.21 |

**Nota:** el mínimo de RPM y de GR es exactamente `0`. Investigado en M2 con
criterio físico/operacional — ver
[docs/eda_findings.md](eda_findings.md#rpm--0-y-gr--0).

## Filas y missing values por pozo

Calculado sobre los 7 archivos descargados; **no se encontraron valores
faltantes (NaN) en ninguna columna de ningún pozo**.

| Archivo | Pozo (índice) | Filas | Missing values |
|---|---|---|---|
| `USROP_A 0 N-NA_F-9_Ad.csv` | 0 | 13,746 | ninguno |
| `USROP_A 1 N-S_F-7d.csv` | 1 | 6,389 | ninguno |
| `USROP_A 2 N-SH_F-14d.csv` | 2 | 47,645 | ninguno |
| `USROP_A 3 N-SH-F-15d.csv` | 3 | 53,041 | ninguno |
| `USROP_A 4 N-SH_F-15Sd.csv` | 4 | 51,708 | ninguno |
| `USROP_A 5 N-SH-F-5d.csv` | 5 | 18,548 | ninguno |
| `USROP_A 6 N-SH_F-9d.csv` | 6 | 7,851 | ninguno |
| **Total** | — | **198,928** | — |

Consistente con el "~200,000 muestras" declarado en ADR-001.

## Limitación conocida: relleno forward/backward aplicado por los autores

Tunkiel, Sui & Wiktorski (2021) — la misma publicación citada en
[ADR-001](adr/001-dataset-selection.md) — aplicaron *forward/backward filling*
para completar los huecos generados por frecuencias de logueo desiguales entre
sensores antes de publicar USROP. Esto explica por qué el dataset no tiene
ningún `NaN` (sección anterior): los gaps ya fueron rellenados por los autores
antes de la publicación, no es que las mediciones originales no tuvieran huecos.

**Consecuencia para este proyecto:** no hay forma de distinguir, dentro de
USROP, un valor efectivamente medido de un valor interpolado (repetido del
último/próximo dato válido) por los autores. Cualquier análisis de
"continuidad" o "eventos" hecho sobre este dataset (ver
[docs/eda_findings.md](eda_findings.md)) opera sobre la versión ya rellenada,
no sobre la señal cruda de los sensores. Se documenta como limitación
conocida del dataset, no como algo a corregir — no hay acceso a los datos
crudos pre-relleno.

## Licencia de los datos

CC BY-NC-SA 4.0 (no comercial), declarada explícitamente en el README del
repositorio de origen. Ver la sección "Licenciamiento" de
[README.md](../README.md) para el detalle completo, separado de la licencia
MIT del código de este proyecto.
