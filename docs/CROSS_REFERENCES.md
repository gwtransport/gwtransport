# Documentation Cross-References

Enrich function docstrings with references to concepts and assumptions when they meaningfully aid understanding. Use Sphinx cross-references that render as clickable links.

Only add references that genuinely help users understand when/how to use a function: when a function assumes something non-obvious, when users need context to choose between functions, or when physical limitations affect interpretation.

## Syntax by Context

| Context                | Syntax                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| Python docstrings      | `:ref:\`concept-dispersion-scales\``                                                          |
| Jupyter notebooks / md | `[Dispersion scales](https://gwtransport.github.io/gwtransport/user_guide/concepts.html#...)` |
| API links (notebooks)  | `[module](https://gwtransport.github.io/gwtransport/api/modules.html#module-gwtransport.xxx)` |

Base URL: `https://gwtransport.github.io/gwtransport/`

## Available Labels

### Concepts (`docs/source/user_guide/concepts.rst`)

| Label                              | Topic                                           |
| ---------------------------------- | ----------------------------------------------- |
| `concept-pore-volume-distribution` | Central concept: aquifer heterogeneity          |
| `concept-residence-time`           | Time in aquifer (V·R/Q)                         |
| `concept-retardation-factor`       | Slower movement due to sorption                 |
| `concept-transport-physics`        | Transport physics overview                      |
| `concept-transport-equation`       | Flow-weighted averaging                         |
| `concept-dispersion`               | Macrodispersion and microdispersion             |
| `concept-dispersion-scales`        | Scale-dependent: macro vs microdispersion       |
| `concept-variance-components`      | Variance decomposition of transport             |
| `concept-gamma-distribution`       | Two-parameter pore volume model                 |
| `concept-gamma-loc`                | Shifted gamma with minimum pore volume (loc)    |
| `concept-nonlinear-sorption`       | Freundlich & Langmuir isotherms, front-tracking |
| `concept-kinematic-wave`           | KW percolation through thick unsaturated zones  |

### Module overview (`docs/source/user_guide/modules.rst`)

| Label                    | Topic                                             |
| ------------------------ | ------------------------------------------------- |
| `concept-drift-envelope` | Regional-drift feasibility table for `radial_asr` |

### Assumptions (`docs/source/user_guide/assumptions.rst`)

| Label                                 | Topic                           |
| ------------------------------------- | ------------------------------- |
| `assumptions`                         | Full assumptions page           |
| `assumption-advection-dominated`      | When diffusion is negligible    |
| `assumption-steady-streamlines`       | Fixed flow path geometry        |
| `assumption-saturated-flow`           | Saturated flow conditions       |
| `assumption-single-porosity`          | Single porosity model           |
| `assumption-no-reactions`             | Conservative transport          |
| `assumption-no-transverse-mixing`     | Independent streamtubes         |
| `assumption-incompressible-flow`      | Incompressible flow             |
| `assumption-gamma-distribution`       | Gamma distribution adequacy     |
| `assumption-linear-retardation`       | Constant retardation factor     |
| `assumption-thermal-retardation`      | Thermal retardation             |
| `assumption-representative-input`     | Representative input data       |
| `assumption-well-mixed-extraction`    | Well-mixed extraction           |
| `assumption-adequate-time-resolution` | Adequate time resolution        |
| `assumption-adequate-discretization`  | Adequate spatial discretization |

### Examples (`examples/`)

| Path                                               | Topic                          |
| -------------------------------------------------- | ------------------------------ |
| `examples/01_Aquifer_Characterization_Temperature` | Temperature tracer calibration |
| `examples/02_Residence_Time_Analysis`              | Residence time calculations    |
| `examples/03_Pathogen_Removal_Bank_Filtration`     | Log removal efficiency         |
| `examples/04_Deposition_Analysis_Bank_Filtration`  | Deposition analysis            |
| `examples/10_Advection_with_non_linear_sorption`   | Freundlich sorption            |
