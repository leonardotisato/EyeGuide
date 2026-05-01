# Hardware result summaries

This folder tracks lightweight hardware-result artifacts organized by:

- board
- quantization level

Convention:

- successful hardware build:
  - `build.json`
  - optional `benchmark.json` when on-board runtime measurements exist
- failed hardware build:
  - `failure.md`

Tracked artifacts in this tree must contain only real numbers extracted from:

- Vivado reports
- FINN-generated build artifacts
- on-board benchmark runs

Do not track analytical or estimated performance numbers here.

