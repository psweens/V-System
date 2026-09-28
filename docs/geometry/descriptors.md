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
| anastomosis any, fraction 0.25 | 8638 ± 3036 | 6.97 ± 2.78 | 6.71 ± 1.58 | 14.84 ± 3.95 | 23.49 ± 4.86 | 1.22 ± 0.10 | 1.21 ± 0.05 | 0.1353 ± 0.0271 | 69 ± 23 | 10.07 ± 1.98 | 95 ± 34 | 1.0 | 14.4 ± 6.1 | 0.11 ± 0.03 | 1.00 ± 0.00 | 0.0 | 49.50 ± 16.55 |
| anastomosis any, fraction 0.5 | 9165 ± 3225 | 7.40 ± 2.94 | 6.54 ± 1.68 | 14.48 ± 4.21 | 23.49 ± 4.86 | 1.39 ± 0.20 | 1.90 ± 0.90 | 0.1387 ± 0.0274 | 49 ± 15 | 6.94 ± 1.87 | 104 ± 38 | 1.0 | 28.6 ± 11.6 | 0.09 ± 0.03 | 1.00 ± 0.00 | 0.0 | 52.48 ± 17.67 |
| anastomosis any, fraction 0.75 | 9586 ± 3360 | 7.79 ± 3.13 | 6.36 ± 1.68 | 14.28 ± 4.42 | 23.49 ± 4.86 | 1.54 ± 0.24 | 2.80 ± 1.00 | 0.1412 ± 0.0304 | 34 ± 10 | 4.63 ± 1.04 | 110 ± 39 | 1.0 | 39.0 ± 15.6 | 0.09 ± 0.04 | 1.00 ± 0.00 | 0.0 | 54.98 ± 19.21 |
| anastomosis arteriovenous, fraction 0.25 | 9419 ± 2051 | 7.28 ± 0.77 | 6.69 ± 1.55 | 17.09 ± 3.98 | 23.49 ± 4.86 | 1.25 ± 0.05 | 1.50 ± 0.12 | 0.1479 ± 0.0274 | 81 ± 19 | 11.20 ± 2.73 | 107 ± 24 | 1.2 ± 0.4 | 14.4 ± 3.0 | 0.15 ± 0.03 | 1.00 ± 0.00 | 0.0 | 278.36 ± 281.27 |
| anastomosis arteriovenous, fraction 0.5 | 9918 ± 2193 | 7.68 ± 0.82 | 6.45 ± 1.53 | 16.67 ± 4.03 | 23.49 ± 4.86 | 1.36 ± 0.08 | 1.76 ± 0.13 | 0.1505 ± 0.0277 | 60 ± 16 | 7.92 ± 2.33 | 113 ± 25 | 1.2 ± 0.4 | 27.4 ± 5.2 | 0.14 ± 0.03 | 1.00 ± 0.00 | 0.0 | 293.38 ± 295.77 |
| anastomosis arteriovenous, fraction 0.75 | 10220 ± 2174 | 7.94 ± 0.75 | 6.31 ± 1.58 | 16.66 ± 4.02 | 23.49 ± 4.86 | 1.45 ± 0.09 | 2.30 ± 0.38 | 0.1521 ± 0.0281 | 49 ± 15 | 6.22 ± 2.02 | 120 ± 25 | 1.0 | 36.6 ± 6.9 | 0.14 ± 0.03 | 1.00 ± 0.00 | 0.0 | 303.88 ± 307.79 |
| anastomosis arteriovenous, fraction 1.0, radius 10 | 10485 ± 2244 | 8.16 ± 0.87 | 6.24 ± 1.50 | 16.63 ± 4.05 | 23.49 ± 4.86 | 1.48 ± 0.12 | 2.41 ± 0.38 | 0.1523 ± 0.0282 | 40 ± 13 | 5.05 ± 1.87 | 125 ± 26 | 1.0 | 43.4 ± 9.6 | 0.13 ± 0.03 | 1.00 ± 0.00 | 0.0 | 311.70 ± 314.58 |
| anastomosis arteriovenous, fraction 1.0, radius 20 | 11373 ± 2590 | 8.86 ± 0.76 | 5.76 ± 1.47 | 16.26 ± 3.77 | 23.49 ± 4.86 | 1.43 ± 0.13 | 2.34 ± 0.36 | 0.1497 ± 0.0273 | 37 ± 16 | 4.21 ± 1.86 | 117 ± 27 | 1.0 | 41.0 ± 8.4 | 0.14 ± 0.03 | 1.00 ± 0.00 | 0.0 | 339.52 ± 345.23 |
| family tree | 15936 ± 133 | 8.58 ± 1.80 | 6.52 ± 1.29 | 14.10 ± 3.23 | 23.49 ± 4.86 | 1.02 ± 0.00 | 1.03 ± 0.00 | 0.0416 ± 0.0094 | 129 | 15.63 ± 3.60 | 127 | 1.0 | 0.0 | 0.08 ± 0.01 | 0.04 ± 0.03 | 0.0 | 32.52 ± 9.47 |
| family mesh | 13424 ± 3273 | 10.74 ± 3.79 | 6.01 ± 1.04 | 15.53 ± 3.06 | 23.49 ± 4.86 | 1.27 ± 0.06 | 1.73 ± 0.15 | 0.1373 ± 0.0221 | 65 ± 14 | 6.73 ± 3.07 | 137 ± 35 | 1.0 | 37.0 ± 13.6 | 0.15 ± 0.06 | 1.00 ± 0.00 | 0.0 | 262.36 ± 100.75 |
| family tumour | 6388 ± 2441 | 4.66 ± 1.52 | 7.04 ± 2.43 | 19.79 ± 8.84 | 29.67 ± 12.72 | 1.55 ± 0.17 | 2.20 ± 0.45 | 0.2149 ± 0.0476 | 25 ± 8 | 5.41 ± 0.72 | 63 ± 26 | 1.0 | 20.0 ± 9.0 | 0.14 ± 0.04 | 1.00 ± 0.00 | 0.0 | 108.28 ± 65.24 |

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
| anastomosis any, fraction 0.25 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 14.4 ± 6.1 | 0.0 | 2.0 ± 3.4 | 2.4 ± 1.5 |
| anastomosis any, fraction 0.5 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 28.6 ± 11.6 | 0.0 | 2.2 ± 3.3 | 4.6 ± 2.6 |
| anastomosis any, fraction 0.75 | 0.0 | 0.0 | 0.0 | 12.8 ± 3.2 | 39.0 ± 15.6 | 0.0 | 3.0 ± 3.7 | 7.8 ± 4.8 |
| anastomosis arteriovenous, fraction 0.25 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 15.2 ± 3.3 | 2.6 ± 2.1 | 0.8 ± 0.8 | 3.0 ± 1.2 |
| anastomosis arteriovenous, fraction 0.5 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 28.2 ± 5.3 | 4.0 ± 3.8 | 1.4 ± 1.3 | 6.4 ± 2.2 |
| anastomosis arteriovenous, fraction 0.75 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 37.6 ± 6.9 | 5.0 ± 4.0 | 3.0 ± 3.2 | 8.6 ± 3.1 |
| anastomosis arteriovenous, fraction 1.0, radius 10 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 44.4 ± 9.6 | 6.8 ± 5.7 | 4.0 ± 3.2 | 12.0 ± 4.6 |
| anastomosis arteriovenous, fraction 1.0, radius 20 | 0.0 | 0.2 ± 0.4 | 0.0 | 28.4 ± 11.5 | 42.0 ± 8.4 | 12.8 ± 8.6 | 0.2 ± 0.4 | 12.0 ± 5.8 |
| family tree | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| family mesh | 0.0 | 0.4 ± 0.5 | 0.0 | 30.4 ± 3.8 | 38.0 ± 13.6 | 15.6 ± 8.6 | 0.0 | 6.6 ± 4.8 |
| family tumour | 0.0 | 0.0 | 0.0 | 16.2 ± 5.1 | 20.0 ± 9.0 | 0.0 | 1.8 ± 2.5 | 3.4 ± 1.7 |

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
