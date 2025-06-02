# ScienceFair2025-Microplastics

**Mapping Microplastics: Computer Modelling to Predict and Address Microplastic Accumulation in Lake Winnipeg**

This repository contains supporting files for my 2025 science fair project, which explores how computer simulations based on environmental data can predict microplastic accumulation patterns in large freshwater bodies.

### References
- [`SF 2025 References.pdf`](./SF%202025%20References.pdf) - Complete list of academic sources, datasets, and supporting material.

---

### Wind-Based Model

- [`red dot test 2.py`](./red%20dot%20test%202.py)  
  A complete wind-based model that:
  - Loads real weather station wind speed and direction data,
  - Interpolates a spatial wind field,
  - Simulates red particle (microplastic) drift accordingly.

---

### Turbulence-Based Model

This multi-step model simulates microplastic transport by combining **fetch (wave exposure)** and **water level gradient** data into a turbulence map.

#### 1. **Fetch Calculation**
- [`fetch1.py`](./fetch1.py)  
  Uses Bresenham’s algorithm to calculate directional fetch distances (wave exposure) across a water body image.
  
- [`fetch2.py`](./fetch2.py)  
  Converts raw fetch data into a structured `(x, y, total_value)` format usable for turbulence modeling.

#### 2. **Water Level Gradient**
- [`waterLevel1.py`](./waterLevel1.py)  
  Uses radial basis function (RBF) interpolation to generate a smooth water level gradient image from a small set of control points.

- [`waterLevel2.py`](./waterLevel2.py)  
  Extracts grayscale values from the water level image into a coordinate-based CSV format.

#### 3. **Turbulence Map Creation**
- [`Turbulence1.py`](./Turbulence1.py)  
  Merges the fetch and water level gradient data using a weighted equation:
