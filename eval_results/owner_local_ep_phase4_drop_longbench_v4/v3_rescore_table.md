# v3 — multi-metric re-score

Generation truncated to 200 chars (`gen_text_head[:200]`). Reference is full LEval `reference_answer`. n=80 per cell.

| plan | policy | rate | F1 | recall | substring | exact | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|
| RR/min_comm | baseline | — | 0.108 | 0.577 | 0.062 | 0.000 | 0.087 |
| RR/min_comm | cross_numa_first | 0.1 | 0.110 | 0.587 | 0.078 | 0.000 | 0.090 |
| RR/min_comm | cross_numa_first | 0.3 | 0.108 | 0.587 | 0.062 | 0.000 | 0.092 |
| RR/min_comm | cross_numa_first | 0.5 | 0.098 | 0.539 | 0.078 | 0.000 | 0.081 |
| RR/min_comm | cross_numa_uniform | 0.1 | 0.110 | 0.598 | 0.078 | 0.000 | 0.089 |
| RR/min_comm | cross_numa_uniform | 0.3 | 0.104 | 0.579 | 0.094 | 0.000 | 0.085 |
| RR/min_comm | cross_numa_uniform | 0.5 | 0.097 | 0.522 | 0.031 | 0.000 | 0.080 |
| RR/min_comm | random | 0.1 | 0.106 | 0.571 | 0.062 | 0.000 | 0.090 |
| RR/min_comm | random | 0.3 | 0.110 | 0.609 | 0.125 | 0.000 | 0.093 |
| RR/min_comm | random | 0.5 | 0.098 | 0.551 | 0.047 | 0.000 | 0.083 |
| RR/min_comm | tail_weight | 0.1 | 0.105 | 0.585 | 0.094 | 0.000 | 0.087 |
| RR/min_comm | tail_weight | 0.3 | 0.106 | 0.578 | 0.047 | 0.000 | 0.088 |
| RR/min_comm | tail_weight | 0.5 | 0.108 | 0.605 | 0.125 | 0.000 | 0.091 |
| RR/min_comm | weighted_tail | 0.1 | 0.109 | 0.583 | 0.031 | 0.000 | 0.089 |
| RR/min_comm | weighted_tail | 0.3 | 0.109 | 0.591 | 0.078 | 0.000 | 0.094 |
| RR/min_comm | weighted_tail | 0.5 | 0.095 | 0.523 | 0.047 | 0.000 | 0.078 |
| LBG/greedy_balance | baseline | — | 0.108 | 0.577 | 0.062 | 0.000 | 0.087 |
| LBG/greedy_balance | cross_numa_first | 0.1 | 0.107 | 0.585 | 0.078 | 0.000 | 0.089 |
| LBG/greedy_balance | cross_numa_first | 0.3 | 0.108 | 0.604 | 0.094 | 0.000 | 0.090 |
| LBG/greedy_balance | cross_numa_first | 0.5 | 0.094 | 0.516 | 0.016 | 0.000 | 0.076 |
| LBG/greedy_balance | cross_numa_uniform | 0.1 | 0.108 | 0.597 | 0.078 | 0.000 | 0.090 |
| LBG/greedy_balance | cross_numa_uniform | 0.3 | 0.096 | 0.541 | 0.094 | 0.000 | 0.083 |
| LBG/greedy_balance | cross_numa_uniform | 0.5 | 0.104 | 0.537 | 0.047 | 0.000 | 0.083 |
| LBG/greedy_balance | random | 0.1 | 0.104 | 0.575 | 0.062 | 0.000 | 0.087 |
| LBG/greedy_balance | random | 0.3 | 0.105 | 0.585 | 0.078 | 0.000 | 0.084 |
| LBG/greedy_balance | random | 0.5 | 0.094 | 0.503 | 0.031 | 0.000 | 0.076 |
| LBG/greedy_balance | tail_weight | 0.1 | 0.107 | 0.585 | 0.047 | 0.000 | 0.089 |
| LBG/greedy_balance | tail_weight | 0.3 | 0.109 | 0.596 | 0.078 | 0.000 | 0.093 |
| LBG/greedy_balance | tail_weight | 0.5 | 0.108 | 0.611 | 0.062 | 0.000 | 0.089 |
| LBG/greedy_balance | weighted_tail | 0.1 | 0.104 | 0.575 | 0.078 | 0.000 | 0.087 |
| LBG/greedy_balance | weighted_tail | 0.3 | 0.110 | 0.595 | 0.078 | 0.000 | 0.091 |
| LBG/greedy_balance | weighted_tail | 0.5 | 0.108 | 0.574 | 0.078 | 0.000 | 0.090 |