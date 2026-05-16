# Baseline results

Test set: `data/processed/test.jsonl` (n=20)

| model | exec_acc | exact | valid_sql | mean_latency_s | mean_cost_usd | n |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 0.850 | 0.500 | 1.000 | 0.874 | 0.000602 | 20 |
| llama-3.2-3b-base | 0.650 | 0.350 | 0.900 | 2.714 | 0.000000 | 20 |
