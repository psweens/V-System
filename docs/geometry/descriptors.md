Means ± standard deviations over seeds 1, 2, 3, 4, 5; every row reproducible with `python main.py --count 1 --seed S <options>`. Definitions: describe.py.

| configuration | points | length mm | d P50 | d P90 | d P99 | arc/chord mean | arc/chord P90 | curvature 1/um | tips | tips/mm | junctions | b0 | b1 | FA | min clearance um | violations | length density mm/mm3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default CLI (4-12 generations) | 70547 ± 105959 | 18.52 ± 24.71 | 7.84 ± 5.26 | 14.49 ± 7.48 | 21.61 ± 6.40 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0541 ± 0.0363 | 569 ± 853 | 20.51 ± 13.17 | 567 ± 853 | 1.0 | 0.0 | 0.16 ± 0.18 | -0.80 ± 1.10 | 172.4 ± 162.4 | 47.94 ± 28.68 |
| default, 8 generations | 15936 ± 133 | 8.58 ± 1.80 | 6.52 ± 1.29 | 14.10 ± 3.23 | 23.49 ± 4.86 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0416 ± 0.0094 | 129 | 15.63 ± 3.60 | 127 | 1.0 | 0.0 | 0.08 ± 0.01 | 0.04 ± 0.03 | 0.0 | 32.52 ± 9.47 |
| LSM calibration (d0 25, d_min 1), default depth | 58129 ± 81932 | 20.91 ± 25.92 | 9.46 ± 6.33 | 17.27 ± 8.76 | 26.63 ± 6.34 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0410 ± 0.0268 | 441 ± 608 | 14.86 ± 8.89 | 439 ± 608 | 1.0 | 0.0 | 0.16 ± 0.18 | -0.18 ± 0.77 | 81.8 ± 112.3 | 32.00 ± 17.36 |
| LSM calibration, 8 generations | 15764 ± 285 | 10.51 ± 2.37 | 7.81 ± 1.18 | 17.27 ± 3.74 | 28.49 ± 4.86 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0336 ± 0.0073 | 127 ± 2 | 12.55 ± 2.71 | 125 ± 2 | 1.0 | 0.0 | 0.08 ± 0.01 | 0.07 ± 0.03 | 0.0 | 22.39 ± 6.52 |
| walk, persistence 2 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.34 ± 0.12 | 1.68 ± 0.22 | 0.2775 ± 0.0402 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.07 ± 0.02 | -10.99 ± 4.56 | 9394.0 ± 5378.0 | 134.97 ± 69.62 |
| walk, persistence 3 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.21 ± 0.07 | 1.41 ± 0.12 | 0.2266 ± 0.0328 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.08 ± 0.02 | -8.52 ± 4.24 | 4582.4 ± 2904.6 | 112.59 ± 63.32 |
| walk, persistence 5 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.12 ± 0.04 | 1.23 ± 0.07 | 0.1755 ± 0.0254 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.10 ± 0.03 | -9.73 ± 5.25 | 2454.4 ± 1271.7 | 80.94 ± 44.83 |
| walk, persistence 8 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.07 ± 0.02 | 1.14 ± 0.04 | 0.1387 ± 0.0201 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.10 ± 0.02 | -6.88 ± 3.91 | 998.6 ± 573.8 | 52.34 ± 22.37 |
| walk, persistence 12 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.05 ± 0.01 | 1.09 ± 0.02 | 0.1133 ± 0.0164 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.11 ± 0.03 | -5.34 ± 3.94 | 413.2 ± 146.9 | 42.63 ± 14.70 |
| walk, persistence 20 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.03 ± 0.01 | 1.05 ± 0.01 | 0.0877 ± 0.0127 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.11 ± 0.04 | -3.71 ± 2.52 | 279.8 ± 122.9 | 39.34 ± 10.59 |
| walk, persistence 40 | 11856 ± 133 | 8.72 ± 1.83 | 6.32 ± 1.22 | 14.00 ± 3.39 | 23.49 ± 4.86 | 1.01 ± 0.00 | 1.03 ± 0.01 | 0.0620 ± 0.0090 | 129 | 15.38 ± 3.54 | 127 | 1.0 | 0.0 | 0.12 ± 0.04 | -3.25 ± 3.81 | 126.0 ± 141.3 | 34.18 ± 6.68 |
| stems + collision avoidance (margin 1) | 15870 ± 202 | 8.57 ± 1.81 | 6.53 ± 1.30 | 14.11 ± 3.23 | 23.49 ± 4.86 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0414 ± 0.0092 | 129 | 15.66 ± 3.62 | 127 | 1.0 | 0.0 | 0.08 ± 0.01 | 1.05 ± 0.04 | 0.0 | 32.46 ± 9.42 |
| walk p=8 + collision avoidance (margin 1) | 8022 ± 2841 | 6.42 ± 2.61 | 7.04 ± 1.71 | 15.24 ± 3.97 | 23.49 ± 4.86 | 1.08 ± 0.03 | 1.14 ± 0.05 | 0.1307 ± 0.0252 | 93 ± 31 | 14.90 ± 3.55 | 91 ± 31 | 1.0 | 0.0 | 0.12 ± 0.05 | 1.00 ± 0.00 | 0.0 | 45.95 ± 16.12 |
| anastomosis any, fraction 0.25 | 9354 ± 3216 | 7.48 ± 2.90 | 6.38 ± 1.49 | 14.49 ± 4.20 | 23.49 ± 4.86 | 1.20 ± 0.03 | 1.36 ± 0.11 | 0.1326 ± 0.0280 | 62 ± 22 | 8.47 ± 1.83 | 93 ± 32 | 1.0 | 16.4 ± 5.4 | 0.10 ± 0.05 | 1.00 ± 0.00 | 0.0 | 53.06 ± 18.40 |
| anastomosis any, fraction 0.5 | 10441 ± 3547 | 8.36 ± 3.10 | 5.72 ± 1.16 | 13.86 ± 3.97 | 23.49 ± 4.86 | 1.29 ± 0.04 | 1.99 ± 0.23 | 0.1352 ± 0.0298 | 41 ± 15 | 4.94 ± 1.65 | 98 ± 36 | 1.0 | 29.8 ± 10.6 | 0.09 ± 0.04 | 1.00 ± 0.00 | 0.0 | 58.22 ± 18.51 |
| anastomosis any, fraction 0.75 | 11489 ± 4093 | 9.28 ± 3.39 | 5.62 ± 1.49 | 13.44 ± 3.91 | 23.49 ± 4.86 | 1.36 ± 0.07 | 2.24 ± 0.37 | 0.1331 ± 0.0274 | 25 ± 9 | 2.82 ± 0.58 | 101 ± 37 | 1.0 | 39.0 ± 14.6 | 0.10 ± 0.05 | 1.00 ± 0.00 | 0.0 | 63.70 ± 20.68 |
| anastomosis arteriovenous, fraction 0.25 | 10260 ± 2163 | 7.91 ± 0.69 | 6.35 ± 1.44 | 16.66 ± 4.02 | 23.49 ± 4.86 | 1.24 ± 0.03 | 1.63 ± 0.13 | 0.1451 ± 0.0263 | 77 ± 22 | 9.80 ± 2.92 | 105 ± 25 | 1.0 | 15.2 ± 2.6 | 0.16 ± 0.03 | 1.00 ± 0.00 | 0.0 | 300.66 ± 301.34 |
| anastomosis arteriovenous, fraction 0.5 | 11403 ± 2118 | 8.84 ± 0.83 | 5.64 ± 1.13 | 16.26 ± 3.75 | 23.49 ± 4.86 | 1.32 ± 0.06 | 1.90 ± 0.25 | 0.1464 ± 0.0276 | 54 ± 25 | 6.29 ± 3.22 | 110 ± 23 | 1.0 | 29.0 ± 2.6 | 0.15 ± 0.03 | 1.00 ± 0.00 | 0.0 | 328.40 ± 315.13 |
| anastomosis arteriovenous, fraction 0.75 | 12000 ± 2394 | 9.29 ± 0.75 | 5.54 ± 1.23 | 16.13 ± 3.64 | 23.49 ± 4.86 | 1.34 ± 0.06 | 2.00 ± 0.21 | 0.1458 ± 0.0274 | 42 ± 21 | 4.55 ± 2.46 | 117 ± 25 | 1.0 | 38.8 ± 4.8 | 0.14 ± 0.03 | 1.00 ± 0.00 | 0.0 | 351.42 ± 348.66 |
| anastomosis arteriovenous, fraction 1.0, radius 10 | 10362 ± 2493 | 7.95 ± 0.86 | 6.38 ± 1.52 | 16.63 ± 4.05 | 23.49 ± 4.86 | 1.27 ± 0.02 | 1.74 ± 0.12 | 0.1468 ± 0.0264 | 55 ± 8 | 7.01 ± 1.36 | 122 ± 27 | 1.0 | 34.6 ± 11.5 | 0.14 ± 0.03 | 1.00 ± 0.00 | 0.0 | 309.43 ± 322.82 |
| anastomosis arteriovenous, fraction 1.0, radius 20 | 12011 ± 2674 | 9.18 ± 0.75 | 5.54 ± 1.18 | 16.23 ± 3.78 | 23.49 ± 4.86 | 1.32 ± 0.04 | 1.93 ± 0.17 | 0.1471 ± 0.0276 | 36 ± 14 | 3.99 ± 1.62 | 125 ± 28 | 1.0 | 45.2 ± 8.6 | 0.15 ± 0.03 | 1.00 ± 0.00 | 0.0 | 350.66 ± 354.88 |
| family tree | 15936 ± 133 | 8.58 ± 1.80 | 6.52 ± 1.29 | 14.10 ± 3.23 | 23.49 ± 4.86 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0416 ± 0.0094 | 129 | 15.63 ± 3.60 | 127 | 1.0 | 0.0 | 0.08 ± 0.01 | 0.04 ± 0.03 | 0.0 | 32.52 ± 9.47 |
| family mesh | 14239 ± 3563 | 11.32 ± 3.95 | 5.72 ± 1.15 | 15.10 ± 2.95 | 23.49 ± 4.86 | 1.25 ± 0.03 | 1.72 ± 0.10 | 0.1339 ± 0.0232 | 65 ± 12 | 6.24 ± 2.27 | 140 ± 39 | 1.0 | 38.4 ± 14.7 | 0.17 ± 0.05 | 1.00 ± 0.00 | 0.0 | 277.36 ± 108.04 |
| family tumour | 7390 ± 2600 | 5.40 ± 1.77 | 6.17 ± 2.26 | 18.58 ± 8.92 | 29.67 ± 12.72 | 1.46 ± 0.14 | 2.12 ± 0.27 | 0.2121 ± 0.0453 | 19 ± 3 | 3.76 ± 1.09 | 58 ± 21 | 1.0 | 20.4 ± 9.5 | 0.13 ± 0.04 | 1.00 ± 0.00 | 0.0 | 123.64 ± 76.35 |

Counted events (mean over seeds):

| configuration | bound_terminations | walk_bound_terminations | collision_truncated_stems | collision_terminations | anastomosis_bridges | anastomosis_bridges_arteriovenous | anastomosis_no_partner | anastomosis_collision_failed |
|---|---|---|---|---|---|---|---|---|
| default CLI (4-12 generations) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| default, 8 generations | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| LSM calibration (d0 25, d_min 1), default depth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| LSM calibration, 8 generations | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 12 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk, persistence 40 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| stems + collision avoidance (margin 1) | 0.0 | 0.0 | 1.2 ± 1.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| walk p=8 + collision avoidance (margin 1) | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| anastomosis any, fraction 0.25 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 16.4 ± 5.4 | 0.0 | 1.0 | 0.4 ± 0.9 |
| anastomosis any, fraction 0.5 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 29.8 ± 10.6 | 0.0 | 2.0 ± 1.2 | 1.2 ± 1.6 |
| anastomosis any, fraction 0.75 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 39.0 ± 14.6 | 0.0 | 2.4 ± 1.9 | 1.6 ± 2.1 |
| anastomosis arteriovenous, fraction 0.25 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 16.2 ± 2.6 | 5.6 ± 3.4 | 0.0 | 1.6 ± 1.1 |
| anastomosis arteriovenous, fraction 0.5 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 30.0 ± 2.6 | 8.8 ± 4.2 | 0.0 | 4.0 ± 2.9 |
| anastomosis arteriovenous, fraction 0.75 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 39.8 ± 4.8 | 13.8 ± 7.9 | 0.6 ± 0.9 | 6.0 ± 5.4 |
| anastomosis arteriovenous, fraction 1.0, radius 10 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 35.6 ± 11.5 | 7.0 ± 6.0 | 23.4 ± 8.8 | 8.2 ± 3.8 |
| anastomosis arteriovenous, fraction 1.0, radius 20 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 46.2 ± 8.6 | 14.0 ± 9.9 | 3.4 ± 2.4 | 8.8 ± 7.2 |
| family tree | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| family mesh | 0.0 | 0.4 ± 0.5 | 0.0 | 30.4 ± 3.8 | 39.4 ± 14.7 | 17.0 ± 8.6 | 1.0 ± 1.2 | 4.6 ± 1.8 |
| family tumour | 0.0 | 0.0 | 0.0 | 16.2 ± 5.1 | 20.4 ± 9.5 | 0.0 | 0.6 ± 0.5 | 0.8 ± 0.8 |

Options per row:

- default CLI (4-12 generations): `(defaults)`
- default, 8 generations: `--iterations 8 8`
- LSM calibration (d0 25, d_min 1), default depth: `--d0 25 5 --d-min 1`
- LSM calibration, 8 generations: `--d0 25 5 --d-min 1 --iterations 8 8`
- walk, persistence 2: `--iterations 8 8 --tortuosity walk --persistence 2`
- walk, persistence 3: `--iterations 8 8 --tortuosity walk --persistence 3`
- walk, persistence 5: `--iterations 8 8 --tortuosity walk --persistence 5`
- walk, persistence 8: `--iterations 8 8 --tortuosity walk --persistence 8`
- walk, persistence 12: `--iterations 8 8 --tortuosity walk --persistence 12`
- walk, persistence 20: `--iterations 8 8 --tortuosity walk --persistence 20`
- walk, persistence 40: `--iterations 8 8 --tortuosity walk --persistence 40`
- stems + collision avoidance (margin 1): `--iterations 8 8 --avoid-collisions`
- walk p=8 + collision avoidance (margin 1): `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions`
- anastomosis any, fraction 0.25: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode any --anastomosis-fraction 0.25`
- anastomosis any, fraction 0.5: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode any --anastomosis-fraction 0.5`
- anastomosis any, fraction 0.75: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode any --anastomosis-fraction 0.75`
- anastomosis arteriovenous, fraction 0.25: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode arteriovenous --anastomosis-fraction 0.25 --grow-in-volume`
- anastomosis arteriovenous, fraction 0.5: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode arteriovenous --anastomosis-fraction 0.5 --grow-in-volume`
- anastomosis arteriovenous, fraction 0.75: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --anastomose --anastomose-mode arteriovenous --anastomosis-fraction 0.75 --grow-in-volume`
- anastomosis arteriovenous, fraction 1.0, radius 10: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --grow-in-volume --anastomose --anastomose-mode arteriovenous --anastomosis-fraction 1.0 --anastomosis-radius 10`
- anastomosis arteriovenous, fraction 1.0, radius 20: `--iterations 8 8 --tortuosity walk --persistence 8 --avoid-collisions --grow-in-volume --anastomose --anastomose-mode arteriovenous --anastomosis-fraction 1.0 --anastomosis-radius 20`
- family tree: `--iterations 8 8 --family tree`
- family mesh: `--iterations 8 8 --family mesh`
- family tumour: `--iterations 8 8 --family tumour`
