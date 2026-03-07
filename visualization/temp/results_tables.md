# Comprehensive Results Tables: MBPP and MATH

Generated from test sweep logs across InfraMind, MAS Router, MoA, and GPTSwarm.

## 1. Overall Accuracy and Latency

### MBPP

| Method | Accuracy (%) | Mean Latency (s) | Median Latency (s) | P90 Latency (s) | N |
|--------|-------------|-------------------|---------------------|------------------|---|
| InfraMind | 74.7 | 86.0 | 38.2 | 210.0 | 12484 |
| MAS Router | 81.0 | 620.2 | 457.5 | 1546.3 | 8977 |
| GPTSwarm | 77.3 | 1018.2 | 1018.2 | 1933.8 | 2500 |
| MoA | 76.1 | 776.1 | 927.3 | 1266.5 | 2500 |

**Interpretation:** On MBPP, MAS Router achieves the highest overall accuracy, while InfraMind has the lowest mean latency. Note that InfraMind results are averaged across all budget tiers and arrival rates, so its latency reflects budget-constrained operation.

### MATH

| Method | Accuracy (%) | Mean Latency (s) | Median Latency (s) | P90 Latency (s) | N |
|--------|-------------|-------------------|---------------------|------------------|---|
| InfraMind | 67.4 | 120.3 | 46.7 | 359.1 | 23999 |
| MAS Router | 66.5 | 140.3 | 79.0 | 357.2 | 3000 |
| GPTSwarm | 69.6 | 1020.1 | 1011.1 | 1900.5 | 3000 |
| MoA | 72.5 | 948.9 | 1012.0 | 1565.1 | 3000 |

**Interpretation:** On MATH, MoA achieves the highest overall accuracy, while InfraMind has the lowest mean latency. Note that InfraMind results are averaged across all budget tiers and arrival rates, so its latency reflects budget-constrained operation.

## 2. Accuracy by Arrival Rate

### MBPP

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 |
|--------|--------|--------|--------|--------|--------|--------|
| InfraMind | 78.3% | 75.4% | 73.1% | 73.4% | -- | 73.2% |
| MAS Router | 80.9% | 82.8% | 81.0% | 80.5% | 82.1% | 78.7% |
| GPTSwarm | 76.4% | 79.2% | 76.6% | 78.6% | -- | 75.8% |
| MoA | 77.2% | 74.6% | 75.2% | 75.8% | -- | 77.8% |

**Interpretation:** This table shows how accuracy degrades as arrival rate increases for MBPP. Higher arrival rates create more queuing contention on the vLLM servers, potentially causing timeouts and degraded quality. InfraMind values are averaged across budget tiers at each arrival rate.

### MATH

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 |
|--------|--------|--------|--------|--------|--------|--------|
| InfraMind | 70.8% | 68.4% | 67.0% | 66.3% | 65.8% | 66.2% |
| MAS Router | 63.8% | 65.8% | 67.2% | 66.6% | 69.0% | 66.6% |
| GPTSwarm | 70.2% | 70.8% | 70.0% | 70.8% | 70.0% | 66.0% |
| MoA | 73.0% | 73.6% | 71.8% | 73.0% | 71.4% | 72.4% |

**Interpretation:** This table shows how accuracy degrades as arrival rate increases for MATH. Higher arrival rates create more queuing contention on the vLLM servers, potentially causing timeouts and degraded quality. InfraMind values are averaged across budget tiers at each arrival rate.

## 3. Mean Latency by Arrival Rate

### MBPP

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 |
|--------|--------|--------|--------|--------|--------|--------|
| InfraMind | 43.2s | 51.2s | 59.4s | 103.3s | -- | 172.9s |
| MAS Router | 136.9s | 551.0s | 657.5s | 766.4s | 764.1s | 846.9s |
| GPTSwarm | 72.7s | 1008.5s | 1306.9s | 1369.4s | -- | 1333.5s |
| MoA | 67.8s | 849.1s | 968.8s | 992.8s | -- | 1002.1s |

**Interpretation:** Latency increases with arrival rate due to queuing effects. Methods with more agents (MoA uses 3 proposers + 1 aggregator) tend to generate more LLM calls per item, amplifying queuing delays under high load.

### MATH

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 |
|--------|--------|--------|--------|--------|--------|--------|
| InfraMind | 27.2s | 84.9s | 102.3s | 142.9s | 173.0s | 191.8s |
| MAS Router | 31.8s | 47.4s | 104.5s | 194.6s | 234.8s | 228.8s |
| GPTSwarm | 102.6s | 937.7s | 1192.3s | 1309.6s | 1291.9s | 1286.2s |
| MoA | 162.9s | 1021.0s | 1100.9s | 1116.7s | 1152.2s | 1139.6s |

**Interpretation:** Latency increases with arrival rate due to queuing effects. Methods with more agents (MoA uses 3 proposers + 1 aggregator) tend to generate more LLM calls per item, amplifying queuing delays under high load.

## 4. InfraMind Budget Scaling

### MBPP (Arrival Rate = 10)

| Budget (s) | Accuracy (%) | Mean Latency (s) | Median Latency (s) | SLO Compliance (%) | N |
|-----------|-------------|-------------------|---------------------|---------------------|---|
| 30 | 80.0 | 37.1 | 22.2 | 58.3 | 499 |
| 50 | 80.4 | 41.2 | 26.9 | 68.0 | 500 |
| 100 | 76.6 | 42.7 | 25.7 | 91.4 | 499 |
| 200 | 78.2 | 50.7 | 28.2 | 98.0 | 500 |
| 300 | 76.4 | 44.3 | 27.4 | 99.6 | 500 |

### MBPP (Arrival Rate = 200)

| Budget (s) | Accuracy (%) | Mean Latency (s) | Median Latency (s) | SLO Compliance (%) | N |
|-----------|-------------|-------------------|---------------------|---------------------|---|
| 30 | 72.7 | 239.2 | 212.4 | 12.0 | 499 |
| 50 | 73.5 | 74.7 | 49.8 | 50.3 | 499 |
| 100 | 72.7 | 88.3 | 51.1 | 74.3 | 499 |
| 200 | 76.0 | 80.1 | 40.8 | 89.0 | 500 |
| 300 | 71.1 | 382.5 | 190.3 | 61.7 | 499 |

**Interpretation:** For MBPP, tighter budgets force InfraMind to select faster models and lighter strategies (Flash), trading accuracy for speed. SLO compliance measures the percentage of requests completing within their assigned budget. At low arrival rates, compliance is generally high; at high arrival rates, queuing delays cause more SLO violations, especially for tight budgets.

### MATH (Arrival Rate = 10)

| Budget (s) | Accuracy (%) | Mean Latency (s) | Median Latency (s) | SLO Compliance (%) | N |
|-----------|-------------|-------------------|---------------------|---------------------|---|
| 10 | 60.4 | 11.7 | 7.6 | 61.0 | 500 |
| 30 | 66.8 | 23.2 | 18.3 | 72.0 | 500 |
| 50 | 70.8 | 27.3 | 21.0 | 81.6 | 500 |
| 100 | 72.4 | 28.9 | 20.9 | 98.8 | 500 |
| 200 | 73.8 | 32.2 | 23.6 | 99.8 | 500 |
| 300 | 74.2 | 33.1 | 23.8 | 100.0 | 500 |
| 600 | 74.4 | 30.3 | 23.6 | 100.0 | 500 |
| 1000 | 73.8 | 31.0 | 24.1 | 100.0 | 500 |

### MATH (Arrival Rate = 200)

| Budget (s) | Accuracy (%) | Mean Latency (s) | Median Latency (s) | SLO Compliance (%) | N |
|-----------|-------------|-------------------|---------------------|---------------------|---|
| 10 | 58.2 | 77.4 | 60.8 | 15.4 | 500 |
| 30 | 65.6 | 148.9 | 117.2 | 27.6 | 500 |
| 50 | 66.8 | 183.8 | 135.2 | 30.8 | 500 |
| 100 | 66.4 | 188.9 | 138.0 | 40.8 | 500 |
| 200 | 68.2 | 207.9 | 131.7 | 64.6 | 500 |
| 300 | 66.8 | 227.0 | 158.3 | 68.0 | 500 |
| 600 | 68.2 | 250.9 | 157.6 | 86.0 | 500 |
| 1000 | 69.4 | 249.3 | 163.2 | 100.0 | 500 |

**Interpretation:** For MATH, tighter budgets force InfraMind to select faster models and lighter strategies (Flash), trading accuracy for speed. SLO compliance measures the percentage of requests completing within their assigned budget. At low arrival rates, compliance is generally high; at high arrival rates, queuing delays cause more SLO violations, especially for tight budgets.

## 5. SLO Compliance Summary

SLO compliance = percentage of requests finishing within a time budget. For baselines (MAS Router, GPTSwarm, MoA), compliance is computed against fixed budget thresholds [30, 50, 100, 200, 300]s and averaged. For InfraMind, compliance uses the per-item budget assignment.

### MBPP

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 | Overall |
|--------|--------|--------|--------|--------|--------|--------|---------|
| InfraMind | 83.1% | 81.6% | 79.4% | 60.6% | -- | 57.5% | 72.4% |
| MAS Router | 58.0% | 33.1% | 30.7% | 27.2% | 24.7% | 21.7% | 32.6% |
| GPTSwarm | 61.3% | 4.2% | 2.4% | 0.9% | -- | 0.3% | 13.8% |
| MoA | 64.0% | 3.3% | 1.8% | 0.5% | -- | 0.2% | 14.0% |

**Interpretation:** SLO compliance reveals how well each method respects time constraints. InfraMind is explicitly trained to meet per-item budgets via its Lagrangian cost constraint, giving it an advantage in compliance. Baseline methods have no budget awareness and rely on fixed topologies, so their compliance depends entirely on whether the workload happens to finish within the threshold. High arrival rates degrade compliance across all methods due to increased queuing contention.

### MATH

| Method | AR=10 | AR=30 | AR=50 | AR=100 | AR=150 | AR=200 | Overall |
|--------|--------|--------|--------|--------|--------|--------|---------|
| InfraMind | 89.1% | 72.2% | 71.6% | 61.6% | 56.5% | 54.1% | 67.5% |
| MAS Router | 86.5% | 79.5% | 65.3% | 41.3% | 32.4% | 31.6% | 56.1% |
| GPTSwarm | 54.8% | 7.9% | 4.5% | 1.9% | 1.2% | 0.8% | 11.8% |
| MoA | 40.4% | 2.8% | 1.8% | 0.5% | 0.2% | 0.0% | 7.6% |

**Interpretation:** SLO compliance reveals how well each method respects time constraints. InfraMind is explicitly trained to meet per-item budgets via its Lagrangian cost constraint, giving it an advantage in compliance. Baseline methods have no budget awareness and rely on fixed topologies, so their compliance depends entirely on whether the workload happens to finish within the threshold. High arrival rates degrade compliance across all methods due to increased queuing contention.
