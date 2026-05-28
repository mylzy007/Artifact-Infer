# Official re-score (LongBench-style F1 + rouge_score package)

Metrics:
- `lb_f1`: LongBench-official QA F1 (article/punct removal + token F1)
- `lb_recall`: same normalization, recall-only (forgiving to long CoT)
- `lb_substring`: 1 if every ref token appears in gen (normalized)
- `rougeL`: Google rouge_score package with Porter stemmer

| plan | label | n | lb_f1 | lb_recall | lb_substring | rouge1 | rouge2 | rougeL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lbg_greedybal | baseline | 64 | 0.1050 | 0.5454 | 0.0625 | 0.1135 | 0.0524 | 0.0908 |
| lbg_greedybal | cross_numa_first_r0.1 | 64 | 0.1045 | 0.5616 | 0.0625 | 0.1111 | 0.0517 | 0.0914 |
| lbg_greedybal | cross_numa_first_r0.3 | 64 | 0.1038 | 0.5723 | 0.0781 | 0.1133 | 0.0519 | 0.0934 |
| lbg_greedybal | cross_numa_first_r0.5 | 64 | 0.0881 | 0.4865 | 0.0156 | 0.0971 | 0.0369 | 0.0791 |
| lbg_greedybal | cross_numa_uniform_r0.1 | 64 | 0.1041 | 0.5637 | 0.0781 | 0.1118 | 0.0533 | 0.0919 |
| lbg_greedybal | cross_numa_uniform_r0.3 | 64 | 0.0921 | 0.5124 | 0.0938 | 0.0995 | 0.0496 | 0.0858 |
| lbg_greedybal | cross_numa_uniform_r0.5 | 64 | 0.0994 | 0.5188 | 0.0469 | 0.1078 | 0.0440 | 0.0853 |
| lbg_greedybal | random_r0.1 | 64 | 0.1004 | 0.5430 | 0.0625 | 0.1084 | 0.0511 | 0.0896 |
| lbg_greedybal | random_r0.3 | 64 | 0.1017 | 0.5494 | 0.0625 | 0.1109 | 0.0513 | 0.0875 |
| lbg_greedybal | random_r0.5 | 64 | 0.0907 | 0.4782 | 0.0312 | 0.0984 | 0.0393 | 0.0787 |
| lbg_greedybal | tail_weight_r0.1 | 64 | 0.1046 | 0.5552 | 0.0469 | 0.1114 | 0.0541 | 0.0914 |
| lbg_greedybal | tail_weight_r0.3 | 64 | 0.1062 | 0.5712 | 0.0625 | 0.1139 | 0.0570 | 0.0965 |
| lbg_greedybal | tail_weight_r0.5 | 64 | 0.1051 | 0.5761 | 0.0625 | 0.1121 | 0.0504 | 0.0905 |
| lbg_greedybal | weighted_tail_r0.1 | 64 | 0.1004 | 0.5470 | 0.0781 | 0.1082 | 0.0491 | 0.0900 |
| lbg_greedybal | weighted_tail_r0.3 | 64 | 0.1072 | 0.5704 | 0.0781 | 0.1139 | 0.0537 | 0.0928 |
| lbg_greedybal | weighted_tail_r0.5 | 64 | 0.1040 | 0.5433 | 0.0469 | 0.1115 | 0.0509 | 0.0932 |
| rr_mincomm | baseline | 64 | 0.1050 | 0.5454 | 0.0625 | 0.1135 | 0.0524 | 0.0908 |
| rr_mincomm | cross_numa_first_r0.1 | 64 | 0.1064 | 0.5637 | 0.0781 | 0.1136 | 0.0562 | 0.0931 |
| rr_mincomm | cross_numa_first_r0.3 | 64 | 0.1043 | 0.5583 | 0.0469 | 0.1117 | 0.0559 | 0.0942 |
| rr_mincomm | cross_numa_first_r0.5 | 64 | 0.0955 | 0.5203 | 0.0469 | 0.1032 | 0.0447 | 0.0846 |
| rr_mincomm | cross_numa_uniform_r0.1 | 64 | 0.1063 | 0.5634 | 0.0781 | 0.1160 | 0.0528 | 0.0924 |
| rr_mincomm | cross_numa_uniform_r0.3 | 64 | 0.1016 | 0.5552 | 0.0938 | 0.1096 | 0.0502 | 0.0891 |
| rr_mincomm | cross_numa_uniform_r0.5 | 64 | 0.0924 | 0.4908 | 0.0312 | 0.1016 | 0.0423 | 0.0819 |
| rr_mincomm | random_r0.1 | 64 | 0.1029 | 0.5463 | 0.0625 | 0.1106 | 0.0551 | 0.0932 |
| rr_mincomm | random_r0.3 | 64 | 0.1067 | 0.5795 | 0.1094 | 0.1138 | 0.0587 | 0.0949 |
| rr_mincomm | random_r0.5 | 64 | 0.0938 | 0.5201 | 0.0469 | 0.1027 | 0.0473 | 0.0863 |
| rr_mincomm | tail_weight_r0.1 | 64 | 0.1028 | 0.5602 | 0.0781 | 0.1086 | 0.0502 | 0.0888 |
| rr_mincomm | tail_weight_r0.3 | 64 | 0.1018 | 0.5427 | 0.0469 | 0.1100 | 0.0476 | 0.0902 |
| rr_mincomm | tail_weight_r0.5 | 64 | 0.1054 | 0.5824 | 0.1094 | 0.1121 | 0.0561 | 0.0939 |
| rr_mincomm | weighted_tail_r0.1 | 64 | 0.1061 | 0.5599 | 0.0312 | 0.1135 | 0.0523 | 0.0923 |
| rr_mincomm | weighted_tail_r0.3 | 64 | 0.1052 | 0.5666 | 0.0781 | 0.1132 | 0.0572 | 0.0963 |
| rr_mincomm | weighted_tail_r0.5 | 64 | 0.0916 | 0.5039 | 0.0469 | 0.1013 | 0.0402 | 0.0814 |