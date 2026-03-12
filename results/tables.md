# Main Results

Accuracy (%) / Mean Latency (s) / SLO Compliance at 100s budget (%)

MAS uses cost_rate=10.0. Baseline accuracy is mean across all arrival rates (routing is identical). InfraMind accuracy is per-rate (it adapts).


## Low (10) req/min

| Dataset | INFRAMIND Acc | INFRAMIND Lat | INFRAMIND SLO | MASRouter Acc | MASRouter Lat | MASRouter SLO | MoA Acc | MoA Lat | MoA SLO | GPTSwarm Acc | GPTSwarm Lat | GPTSwarm SLO |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| MATH | 70.8 | **27.2** | **98.7** | 70.0 | 49.6 | 89.6 | **74.6** | 122.0 | 34.4 | 70.7 | 84.1 | 67.8 |
| MBPP | 78.3 | **43.2** | **90.2** | **81.7** | 253.2 | 32.6 | 76.1 | 67.8 | 86.0 | 77.3 | 72.7 | 79.2 |
| GSM-Hard | **59.1** | **25.6** | **99.2** | 55.5 | 37.7 | 98.0 | 49.0 | 38.2 | **99.2** | 54.4 | 52.1 | 95.0 |
| HumanEval | **98.5** | **18.1** | **99.2** | 91.8 | 69.6 | 74.8 | 88.4 | 305.1 | 11.4 | 88.1 | 55.4 | **99.2** |
| MMLU-Pro | 53.5 | **30.2** | 96.4 | **57.7** | 82.6 | 72.0 | 44.0 | 123.9 | 64.0 | 56.9 | 35.6 | **99.8** |

## Mid (100) req/min

| Dataset | INFRAMIND Acc | INFRAMIND Lat | INFRAMIND SLO | MASRouter Acc | MASRouter Lat | MASRouter SLO | MoA Acc | MoA Lat | MoA SLO | GPTSwarm Acc | GPTSwarm Lat | GPTSwarm SLO |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| MATH | 66.3 | **142.9** | **59.0** | 70.0 | 398.7 | 22.0 | **74.6** | 1159.7 | 0.0 | 70.7 | 1325.1 | 0.2 |
| MBPP | 73.4 | **103.3** | **67.5** | **81.7** | 984.8 | 22.3 | 76.1 | 992.8 | 0.2 | 77.3 | 1369.4 | 0.2 |
| GSM-Hard | 51.8 | **98.2** | **63.4** | **55.5** | 652.3 | 8.6 | 49.0 | 524.5 | 4.0 | 54.4 | 806.2 | 2.2 |
| HumanEval | **97.7** | **32.7** | **96.8** | 91.8 | 205.1 | 28.2 | 88.4 | 952.3 | 0.0 | 88.1 | 351.6 | 3.0 |
| MMLU-Pro | 52.1 | **217.7** | **53.4** | **57.7** | 907.8 | 7.5 | 44.0 | 1170.9 | 0.0 | 56.9 | 753.3 | 5.2 |

## High (200) req/min

| Dataset | INFRAMIND Acc | INFRAMIND Lat | INFRAMIND SLO | MASRouter Acc | MASRouter Lat | MASRouter SLO | MoA Acc | MoA Lat | MoA SLO | GPTSwarm Acc | GPTSwarm Lat | GPTSwarm SLO |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| MATH | 66.2 | **191.8** | **44.8** | 70.0 | 429.8 | 9.0 | **74.6** | 1162.1 | 0.0 | 70.7 | 1263.3 | 0.0 |
| MBPP | 73.2 | **172.9** | **55.6** | **81.7** | 1164.4 | 17.3 | 76.1 | 1002.1 | 0.0 | 77.3 | 1333.5 | 0.0 |
| GSM-Hard | 51.1 | **92.6** | **66.0** | **55.5** | 680.0 | 8.6 | 49.0 | 560.5 | 2.6 | 54.4 | 877.1 | 0.6 |
| HumanEval | **98.2** | **35.6** | **95.1** | 91.8 | 205.4 | 32.1 | 88.4 | 877.1 | 0.0 | 88.1 | 399.9 | 0.0 |
| MMLU-Pro | 51.2 | **184.8** | **59.5** | **57.7** | 949.1 | 6.0 | 44.0 | 1221.3 | 0.0 | 56.9 | 844.2 | 2.4 |

# Paper Table: Main Results


## Low Load (10 req/min)

| Method | MATH Acc | MATH Lat | MBPP Acc | MBPP Lat | GSM-Hard Acc | GSM-Hard Lat | HumanEval Acc | HumanEval Lat | MMLU-Pro Acc | MMLU-Pro Lat |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| InfraMind | 70.8 | 27 | 78.3 | 43 | 59.1 | 26 | 98.5 | 18 | 53.5 | 30 |
| MASRouter | 70.0 | 50 | 81.7 | 253 | 55.5 | 38 | 91.8 | 70 | 57.7 | 83 |
| GPTSwarm | 70.7 | 84 | 77.3 | 73 | 54.4 | 52 | 88.1 | 55 | 56.9 | 36 |
| MoA | 74.6 | 122 | 76.1 | 68 | 49.0 | 38 | 88.4 | 305 | 44.0 | 124 |

## High Load (200 req/min)

| Method | MATH Acc | MATH Lat | MBPP Acc | MBPP Lat | GSM-Hard Acc | GSM-Hard Lat | HumanEval Acc | HumanEval Lat | MMLU-Pro Acc | MMLU-Pro Lat |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| InfraMind | 66.2 | 192 | 73.2 | 173 | 51.1 | 93 | 98.2 | 36 | 51.2 | 185 |
| MASRouter | 70.0 | 430 | 81.7 | 1164 | 55.5 | 680 | 91.8 | 205 | 57.7 | 949 |
| GPTSwarm | 70.7 | 1263 | 77.3 | 1333 | 54.4 | 877 | 88.1 | 400 | 56.9 | 844 |
| MoA | 74.6 | 1162 | 76.1 | 1002 | 49.0 | 561 | 88.4 | 877 | 44.0 | 1221 |

# InfraMind: Performance by Budget Tier (averaged across arrival rates)

| Dataset | Budget | Acc (%) | Lat (s) | SLO@100 (%) | SLO@300 (%) |
|---------|--------|--------:|--------:|------------:|------------:|
| MATH | 30.0 | 66.2 | 88.0 | 67.9 | 93.0 |
| MATH | 50.0 | 67.3 | 107.1 | 62.6 | 91.2 |
| MATH | 100.0 | 67.7 | 123.0 | 60.2 | 87.2 |
| MATH | 200.0 | 69.6 | 139.0 | 58.3 | 81.7 |
| MATH | 300.0 | 68.9 | 150.0 | 56.9 | 80.0 |
| MBPP | 30.0 | 74.2 | 91.6 | 71.0 | 93.7 |
| MBPP | 50.0 | 75.3 | 72.0 | 77.5 | 96.2 |
| MBPP | 100.0 | 75.0 | 63.1 | 80.5 | 97.8 |
| MBPP | 200.0 | 74.8 | 77.1 | 74.7 | 95.1 |
| MBPP | 300.0 | 74.0 | 126.1 | 67.8 | 90.9 |
| GSM-Hard | 30.0 | 53.4 | 75.2 | 67.3 | 98.2 |
| GSM-Hard | 50.0 | 53.6 | 81.0 | 67.1 | 97.7 |
| GSM-Hard | 100.0 | 54.7 | 83.0 | 66.0 | 97.6 |
| GSM-Hard | 200.0 | 54.3 | 88.2 | 66.9 | 96.2 |
| GSM-Hard | 300.0 | 53.1 | 88.8 | 65.6 | 96.4 |
| HumanEval | 30.0 | 97.9 | 30.3 | 95.4 | 99.8 |
| HumanEval | 50.0 | 97.4 | 26.6 | 97.2 | 100.0 |
| HumanEval | 100.0 | 98.2 | 27.7 | 97.9 | 100.0 |
| HumanEval | 200.0 | 98.5 | 31.0 | 95.6 | 100.0 |
| HumanEval | 300.0 | 97.9 | 28.0 | 97.1 | 100.0 |
| MMLU-Pro | 30.0 | 51.6 | 135.5 | 64.2 | 81.5 |
| MMLU-Pro | 50.0 | 51.6 | 132.7 | 65.2 | 82.1 |
| MMLU-Pro | 100.0 | 52.3 | 152.7 | 61.3 | 79.0 |
| MMLU-Pro | 200.0 | 52.2 | 150.2 | 62.9 | 78.5 |
| MMLU-Pro | 300.0 | 53.7 | 163.1 | 61.6 | 74.4 |

# Load Degradation: Low (10) → Mid (100) → High (200) req/min

Latency increase factor from low to high load. Baseline accuracy is constant (mean across rates).

| Method | Dataset | Accuracy (%) | Lat Low (s) | Lat Mid (s) | Lat High (s) | Lat ×increase |
|--------|---------|-------------:|------------:|------------:|--------------:|--------------:|
| INFRAMIND | MATH | 70.8 | 27.2 | 142.9 | 191.8 | 7.0× |
| INFRAMIND | MBPP | 78.3 | 43.2 | 103.3 | 172.9 | 4.0× |
| INFRAMIND | GSM-Hard | 59.1 | 25.6 | 98.2 | 92.6 | 3.6× |
| INFRAMIND | HumanEval | 98.5 | 18.1 | 32.7 | 35.6 | 2.0× |
| INFRAMIND | MMLU-Pro | 53.5 | 30.2 | 217.7 | 184.8 | 6.1× |
| MASRouter | MATH | 70.0 | 49.6 | 398.7 | 429.8 | 8.7× |
| MASRouter | MBPP | 81.7 | 253.2 | 984.8 | 1164.4 | 4.6× |
| MASRouter | GSM-Hard | 55.5 | 37.7 | 652.3 | 680.0 | 18.0× |
| MASRouter | HumanEval | 91.8 | 69.6 | 205.1 | 205.4 | 3.0× |
| MASRouter | MMLU-Pro | 57.7 | 82.6 | 907.8 | 949.1 | 11.5× |
| MoA | MATH | 74.6 | 122.0 | 1159.7 | 1162.1 | 9.5× |
| MoA | MBPP | 76.1 | 67.8 | 992.8 | 1002.1 | 14.8× |
| MoA | GSM-Hard | 49.0 | 38.2 | 524.5 | 560.5 | 14.7× |
| MoA | HumanEval | 88.4 | 305.1 | 952.3 | 877.1 | 2.9× |
| MoA | MMLU-Pro | 44.0 | 123.9 | 1170.9 | 1221.3 | 9.9× |
| GPTSwarm | MATH | 70.7 | 84.1 | 1325.1 | 1263.3 | 15.0× |
| GPTSwarm | MBPP | 77.3 | 72.7 | 1369.4 | 1333.5 | 18.3× |
| GPTSwarm | GSM-Hard | 54.4 | 52.1 | 806.2 | 877.1 | 16.8× |
| GPTSwarm | HumanEval | 88.1 | 55.4 | 351.6 | 399.9 | 7.2× |
| GPTSwarm | MMLU-Pro | 56.9 | 35.6 | 753.3 | 844.2 | 23.7× |

# InfraMind: Model & Strategy Adaptation Patterns

Shows how InfraMind shifts model/strategy selection under different loads.


## MATH

### Strategy Distribution (% of steps)

| Load | Flash | Concise | DeepThink |
|------|------:|--------:|----------:|
| Low (10) | 25.2 | 37.6 | 37.2 |
| Mid (100) | 26.1 | 36.4 | 37.5 |
| High (200) | 26.1 | 35.5 | 38.4 |

### Model Distribution (% of steps)

| Load |Qwen2.5-Coder-14B-In | DeepSeek-R1-Distill- | Llama-3.1-8B-Instruc | Llama-3.2-3B-Instruc | Mistral-Small-24B-In |
|------|----: | ----: | ----: | ----: | ----: |
| Low (10) | 31.1 | 30.6 | 11.0 | 19.7 | 7.5 |
| Mid (100) | 37.6 | 16.2 | 15.5 | 20.8 | 9.9 |
| High (200) | 35.9 | 16.7 | 16.0 | 21.3 | 10.0 |

## MBPP

### Strategy Distribution (% of steps)

| Load | Flash | Concise | DeepThink |
|------|------:|--------:|----------:|
| Low (10) | 22.9 | 44.8 | 32.3 |
| Mid (100) | 23.9 | 48.8 | 27.3 |
| High (200) | 24.5 | 47.3 | 28.2 |

### Model Distribution (% of steps)

| Load |Qwen2.5-Coder-14B-In | DeepSeek-R1-Distill- | Llama-3.1-8B-Instruc | Llama-3.2-3B-Instruc | Mistral-Small-24B-In |
|------|----: | ----: | ----: | ----: | ----: |
| Low (10) | 45.9 | 17.5 | 14.7 | 10.5 | 11.4 |
| Mid (100) | 54.3 | 10.0 | 15.4 | 9.3 | 10.9 |
| High (200) | 52.5 | 11.3 | 15.8 | 9.1 | 11.3 |

## GSM-Hard

### Strategy Distribution (% of steps)

| Load | Flash | Concise | DeepThink |
|------|------:|--------:|----------:|
| Low (10) | 37.3 | 41.0 | 21.8 |
| Mid (100) | 33.3 | 40.9 | 25.7 |
| High (200) | 35.7 | 37.7 | 26.6 |

### Model Distribution (% of steps)

| Load |Qwen2.5-Coder-14B-In | DeepSeek-R1-Distill- | Llama-3.1-8B-Instruc | Llama-3.2-3B-Instruc | Mistral-Small-24B-In |
|------|----: | ----: | ----: | ----: | ----: |
| Low (10) | 22.4 | 53.5 | 8.2 | 8.0 | 7.9 |
| Mid (100) | 38.3 | 17.6 | 15.5 | 11.7 | 16.9 |
| High (200) | 36.5 | 14.8 | 17.8 | 12.8 | 18.1 |

## HumanEval

### Strategy Distribution (% of steps)

| Load | Flash | Concise | DeepThink |
|------|------:|--------:|----------:|
| Low (10) | 37.9 | 42.2 | 19.9 |
| Mid (100) | 34.0 | 46.4 | 19.5 |
| High (200) | 35.0 | 44.7 | 20.3 |

### Model Distribution (% of steps)

| Load |Qwen2.5-Coder-14B-In | DeepSeek-R1-Distill- | Llama-3.1-8B-Instruc | Llama-3.2-3B-Instruc | Mistral-Small-24B-In |
|------|----: | ----: | ----: | ----: | ----: |
| Low (10) | 41.4 | 4.3 | 22.1 | 22.2 | 10.0 |
| Mid (100) | 43.6 | 4.9 | 21.8 | 19.9 | 9.9 |
| High (200) | 41.0 | 5.0 | 22.7 | 20.0 | 11.4 |

## MMLU-Pro

### Strategy Distribution (% of steps)

| Load | Flash | Concise | DeepThink |
|------|------:|--------:|----------:|
| Low (10) | 42.7 | 32.9 | 24.4 |
| Mid (100) | 40.3 | 33.9 | 25.8 |
| High (200) | 41.8 | 32.3 | 26.0 |

### Model Distribution (% of steps)

| Load |Qwen2.5-Coder-14B-In | DeepSeek-R1-Distill- | Llama-3.1-8B-Instruc | Llama-3.2-3B-Instruc | Mistral-Small-24B-In |
|------|----: | ----: | ----: | ----: | ----: |
| Low (10) | 35.6 | 33.0 | 9.0 | 14.5 | 7.8 |
| Mid (100) | 38.0 | 21.4 | 11.3 | 19.1 | 10.2 |
| High (200) | 37.1 | 18.2 | 12.7 | 20.5 | 11.4 |