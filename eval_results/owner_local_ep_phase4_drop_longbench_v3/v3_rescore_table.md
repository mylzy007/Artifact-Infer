# v3 — multi-metric re-score

Generation truncated to 200 chars (`gen_text_head[:200]`). Reference is full LEval `reference_answer`. n=80 per cell.

| plan | policy | rate | F1 | recall | substring | exact | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|
| RR/min_comm | baseline | — | 0.216 | 0.325 | 0.025 | 0.000 | 0.174 |
| RR/min_comm | cross_numa_first | 0.1 | 0.220 | 0.329 | 0.025 | 0.000 | 0.178 |
| RR/min_comm | cross_numa_first | 0.3 | 0.227 | 0.346 | 0.013 | 0.000 | 0.187 |
| RR/min_comm | cross_numa_first | 0.5 | 0.197 | 0.298 | 0.013 | 0.000 | 0.166 |
| RR/min_comm | cross_numa_uniform | 0.1 | 0.221 | 0.334 | 0.037 | 0.000 | 0.175 |
| RR/min_comm | cross_numa_uniform | 0.3 | 0.218 | 0.334 | 0.000 | 0.000 | 0.174 |
| RR/min_comm | cross_numa_uniform | 0.5 | 0.219 | 0.326 | 0.000 | 0.000 | 0.180 |
| RR/min_comm | random | 0.1 | 0.230 | 0.339 | 0.025 | 0.000 | 0.189 |
| RR/min_comm | random | 0.3 | 0.223 | 0.347 | 0.025 | 0.000 | 0.179 |
| RR/min_comm | random | 0.5 | 0.225 | 0.334 | 0.013 | 0.000 | 0.180 |
| RR/min_comm | tail_weight | 0.1 | 0.220 | 0.331 | 0.025 | 0.000 | 0.175 |
| RR/min_comm | tail_weight | 0.3 | 0.214 | 0.325 | 0.013 | 0.000 | 0.175 |
| RR/min_comm | tail_weight | 0.5 | 0.219 | 0.329 | 0.025 | 0.000 | 0.181 |
| RR/min_comm | weighted_tail | 0.1 | 0.221 | 0.342 | 0.025 | 0.000 | 0.171 |
| RR/min_comm | weighted_tail | 0.3 | 0.209 | 0.318 | 0.025 | 0.000 | 0.166 |
| RR/min_comm | weighted_tail | 0.5 | 0.209 | 0.317 | 0.013 | 0.000 | 0.174 |
| LBG/greedy_balance | baseline | — | 0.216 | 0.325 | 0.025 | 0.000 | 0.174 |
| LBG/greedy_balance | cross_numa_first | 0.1 | 0.211 | 0.309 | 0.013 | 0.000 | 0.167 |
| LBG/greedy_balance | cross_numa_first | 0.3 | 0.218 | 0.336 | 0.025 | 0.000 | 0.172 |
| LBG/greedy_balance | cross_numa_first | 0.5 | 0.207 | 0.303 | 0.000 | 0.000 | 0.168 |
| LBG/greedy_balance | cross_numa_uniform | 0.1 | 0.216 | 0.324 | 0.000 | 0.000 | 0.170 |
| LBG/greedy_balance | cross_numa_uniform | 0.3 | 0.229 | 0.332 | 0.037 | 0.000 | 0.190 |
| LBG/greedy_balance | cross_numa_uniform | 0.5 | 0.213 | 0.325 | 0.013 | 0.000 | 0.173 |
| LBG/greedy_balance | random | 0.1 | 0.234 | 0.345 | 0.025 | 0.000 | 0.183 |
| LBG/greedy_balance | random | 0.3 | 0.227 | 0.345 | 0.013 | 0.000 | 0.177 |
| LBG/greedy_balance | random | 0.5 | 0.221 | 0.341 | 0.025 | 0.000 | 0.180 |
| LBG/greedy_balance | tail_weight | 0.1 | 0.218 | 0.327 | 0.025 | 0.000 | 0.173 |
| LBG/greedy_balance | tail_weight | 0.3 | 0.221 | 0.337 | 0.025 | 0.000 | 0.184 |
| LBG/greedy_balance | tail_weight | 0.5 | 0.234 | 0.361 | 0.013 | 0.000 | 0.191 |
| LBG/greedy_balance | weighted_tail | 0.1 | 0.223 | 0.342 | 0.025 | 0.000 | 0.182 |
| LBG/greedy_balance | weighted_tail | 0.3 | 0.217 | 0.335 | 0.013 | 0.000 | 0.177 |
| LBG/greedy_balance | weighted_tail | 0.5 | 0.226 | 0.338 | 0.000 | 0.000 | 0.185 |