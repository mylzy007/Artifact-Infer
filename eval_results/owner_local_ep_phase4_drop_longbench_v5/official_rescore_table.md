# Official re-score (LongBench-style F1 + rouge_score package)

Metrics:
- `lb_f1`: LongBench-official QA F1 (article/punct removal + token F1)
- `lb_recall`: same normalization, recall-only (forgiving to long CoT)
- `lb_substring`: 1 if every ref token appears in gen (normalized)
- `rougeL`: Google rouge_score package with Porter stemmer

| plan | label | n | lb_f1 | lb_recall | lb_substring | rouge1 | rouge2 | rougeL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lbg_greedybal | baseline | 32 | 0.1145 | 0.7392 | 0.5000 | 0.1119 | 0.0774 | 0.1059 |
| lbg_greedybal | cross_numa_first_r0.1 | 32 | 0.1081 | 0.7217 | 0.5000 | 0.1083 | 0.0746 | 0.1020 |
| lbg_greedybal | cross_numa_first_r0.3 | 32 | 0.1000 | 0.6306 | 0.4062 | 0.1013 | 0.0677 | 0.0972 |
| lbg_greedybal | cross_numa_first_r0.5 | 32 | 0.0992 | 0.6124 | 0.3438 | 0.1054 | 0.0676 | 0.0998 |
| lbg_greedybal | cross_numa_uniform_r0.1 | 32 | 0.1055 | 0.6603 | 0.5000 | 0.1065 | 0.0742 | 0.1029 |
| lbg_greedybal | cross_numa_uniform_r0.3 | 32 | 0.1143 | 0.7062 | 0.4688 | 0.1120 | 0.0786 | 0.1025 |
| lbg_greedybal | cross_numa_uniform_r0.5 | 32 | 0.0950 | 0.5902 | 0.2812 | 0.0966 | 0.0610 | 0.0879 |
| lbg_greedybal | random_r0.1 | 32 | 0.1151 | 0.7566 | 0.5312 | 0.1152 | 0.0765 | 0.1084 |
| lbg_greedybal | random_r0.3 | 32 | 0.0900 | 0.5445 | 0.3750 | 0.0895 | 0.0570 | 0.0847 |
| lbg_greedybal | random_r0.5 | 32 | 0.0897 | 0.5760 | 0.3438 | 0.0985 | 0.0617 | 0.0908 |
| lbg_greedybal | tail_weight_r0.1 | 32 | 0.1123 | 0.7298 | 0.5312 | 0.1153 | 0.0791 | 0.1086 |
| lbg_greedybal | tail_weight_r0.3 | 32 | 0.1065 | 0.7732 | 0.5625 | 0.1083 | 0.0713 | 0.1038 |
| lbg_greedybal | tail_weight_r0.5 | 32 | 0.1284 | 0.7832 | 0.5625 | 0.1226 | 0.0876 | 0.1174 |
| lbg_greedybal | weighted_tail_r0.1 | 32 | 0.1246 | 0.8063 | 0.6562 | 0.1221 | 0.0893 | 0.1195 |
| lbg_greedybal | weighted_tail_r0.3 | 32 | 0.1264 | 0.7514 | 0.5312 | 0.1227 | 0.0829 | 0.1160 |
| lbg_greedybal | weighted_tail_r0.5 | 32 | 0.1093 | 0.7122 | 0.5000 | 0.1056 | 0.0699 | 0.0969 |
| rr_mincomm | baseline | 32 | 0.1145 | 0.7392 | 0.5000 | 0.1119 | 0.0774 | 0.1059 |
| rr_mincomm | cross_numa_first_r0.1 | 32 | 0.1168 | 0.7398 | 0.5625 | 0.1141 | 0.0815 | 0.1112 |
| rr_mincomm | cross_numa_first_r0.3 | 32 | 0.0962 | 0.6370 | 0.3750 | 0.0984 | 0.0625 | 0.0929 |
| rr_mincomm | cross_numa_first_r0.5 | 32 | 0.1085 | 0.7405 | 0.5000 | 0.1096 | 0.0680 | 0.1018 |
| rr_mincomm | cross_numa_uniform_r0.1 | 32 | 0.1124 | 0.7167 | 0.5312 | 0.1135 | 0.0768 | 0.1088 |
| rr_mincomm | cross_numa_uniform_r0.3 | 32 | 0.0986 | 0.6207 | 0.3125 | 0.1030 | 0.0589 | 0.0913 |
| rr_mincomm | cross_numa_uniform_r0.5 | 32 | 0.1054 | 0.6321 | 0.2188 | 0.1125 | 0.0595 | 0.1000 |
| rr_mincomm | random_r0.1 | 32 | 0.0984 | 0.6847 | 0.3750 | 0.1000 | 0.0580 | 0.0916 |
| rr_mincomm | random_r0.3 | 32 | 0.0946 | 0.5986 | 0.3750 | 0.0975 | 0.0664 | 0.0865 |
| rr_mincomm | random_r0.5 | 32 | 0.0981 | 0.5581 | 0.3125 | 0.1097 | 0.0669 | 0.1007 |
| rr_mincomm | tail_weight_r0.1 | 32 | 0.1014 | 0.6806 | 0.5000 | 0.1047 | 0.0712 | 0.0990 |
| rr_mincomm | tail_weight_r0.3 | 32 | 0.1144 | 0.7568 | 0.5625 | 0.1150 | 0.0830 | 0.1107 |
| rr_mincomm | tail_weight_r0.5 | 32 | 0.1124 | 0.7745 | 0.6250 | 0.1120 | 0.0774 | 0.1050 |
| rr_mincomm | weighted_tail_r0.1 | 32 | 0.1135 | 0.6852 | 0.5000 | 0.1156 | 0.0815 | 0.1114 |
| rr_mincomm | weighted_tail_r0.3 | 32 | 0.1063 | 0.6604 | 0.4375 | 0.1085 | 0.0728 | 0.1032 |
| rr_mincomm | weighted_tail_r0.5 | 32 | 0.0905 | 0.6479 | 0.4688 | 0.0918 | 0.0585 | 0.0850 |