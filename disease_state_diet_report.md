# Diet Optimization Report — Full MILP Pipeline
Generated: 2026-04-20 08:43

**Profile:** 29yo male, 70kg, 178cm, weight_lifting
**Supplements:** B12, Vitamin D, Iodine
**Baseline:** $4.15/day, cap=5, 21 unique foods

## Disease States

| Disease | Cost/day | Cap | Foods | Energy | Protein | Eff. ω-3 | ω-3:ω-6 | Polyphenols |
|---|---|---|---|---|---|---|---|---|
| cardiovascular | $9.15 | 7 | 9 | 2669 kcal | 145.8g | 4.25g | 1:4 | 1664 mg |
| celiac | $5.35 | 4 | 26 | 2795 kcal | 136.2g | 0.32g | 1:73 | 1821 mg |
| diabetes | $3.68 | 7 | 15 | 2603 kcal | 126.0g | 0.21g | 1:119 | 811 mg |
| gout | $3.73 | 7 | 13 | 2732 kcal | 80.0g | 0.36g | 1:94 | 546 mg |
| hypertension | $3.9 | 6 | 21 | 2791 kcal | 119.3g | 0.34g | 1:97 | 832 mg |
| hypothyroidism | $4.06 | 5 | 22 | 2795 kcal | 130.3g | 0.32g | 1:89 | 978 mg |
| ibd crohns | INFEASIBLE | — | — | — | — | — | — | — |
| iron deficiency anemia | $4.29 | 5 | 21 | 2831 kcal | 130.0g | 0.32g | 1:83 | 1126 mg |
| kidney disease | INFEASIBLE | — | — | — | — | — | — | — |
| migraine | $5.14 | 6 | 18 | 2830 kcal | 133.4g | 0.34g | 1:80 | 1223 mg |
| nafld | INFEASIBLE | — | — | — | — | — | — | — |
| osteoporosis | $4.11 | 5 | 19 | 2765 kcal | 129.7g | 0.33g | 1:89 | 1032 mg |
| preeclampsia risk | $3.88 | 7 | 18 | 2791 kcal | 107.1g | 0.36g | 1:99 | 673 mg |
| pregnancy | $4.19 | 6 | 16 | 2800 kcal | 141.8g | 0.34g | 1:90 | 986 mg |
| sarcopenia | $4.41 | 5 | 21 | 2821 kcal | 142.6g | 0.3g | 1:87 | 1053 mg |

## Dietary Presets

| Preset | Cost/day | Cap | Foods | Energy | Protein | Eff. ω-3 | ω-3:ω-6 | Polyphenols |
|---|---|---|---|---|---|---|---|---|
| dairy free | $3.66 | 6 | 18 | 2798 kcal | 124.7g | 0.37g | 1:94 | 976 mg |
| egg free | $4.18 | 7 | 12 | 2840 kcal | 143.7g | 0.35g | 1:71 | 939 mg |
| gluten free | $3.67 | 7 | 14 | 2744 kcal | 105.0g | 0.37g | 1:99 | 501 mg |
| lacto vegetarian | $4.46 | 7 | 11 | 2851 kcal | 122.2g | 0.42g | 1:57 | 1513 mg |
| legume free | $3.84 | 7 | 12 | 2748 kcal | 108.0g | 0.37g | 1:84 | 565 mg |
| nightshade free | $4.34 | 5 | 16 | 2782 kcal | 134.5g | 0.34g | 1:90 | 1157 mg |
| nut free | $3.83 | 7 | 14 | 2851 kcal | 107.1g | 0.37g | 1:95 | 438 mg |
| ovo vegetarian | $4.03 | 5 | 16 | 2851 kcal | 125.1g | 0.49g | 1:55 | 888 mg |
| paleo | $4.2 | 7 | 9 | 2851 kcal | 105.0g | 0.36g | 1:63 | 922 mg |
| pescatarian | $3.88 | 6 | 21 | 2790 kcal | 110.7g | 0.3g | 1:99 | 837 mg |
| soy free | $4.33 | 5 | 21 | 2798 kcal | 126.5g | 0.31g | 1:94 | 1528 mg |
| vegan | $5.97 | 6 | 17 | 2832 kcal | 120.2g | 0.37g | 1:68 | 1431 mg |
| vegan gluten free | $5.46 | 6 | 16 | 2824 kcal | 116.4g | 0.38g | 1:66 | 1392 mg |
| vegan nut free | $5.69 | 7 | 10 | 2851 kcal | 114.5g | 0.43g | 1:77 | 1415 mg |
| vegetarian | $3.88 | 6 | 21 | 2790 kcal | 110.7g | 0.3g | 1:99 | 837 mg |
| whole30 | $4.2 | 7 | 9 | 2851 kcal | 105.0g | 0.36g | 1:63 | 922 mg |
| whole foods | $4.15 | 5 | 21 | 2804 kcal | 124.0g | 0.32g | 1:91 | 1549 mg |

**29/32 configurations solved** (12/15 diseases, 17/17 dietary presets)

Infeasible cases (IBD/Crohn's, kidney disease, NAFLD) have constraint combinations too tight for the 150-food pool — the single-day LP is feasible but the weekly MILP times out. These can be solved with a larger pool or longer time limit.