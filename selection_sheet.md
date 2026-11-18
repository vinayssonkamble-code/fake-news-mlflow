# Champion–Challenger Model Selection Sheet

## 1. Project Details
- Project: Fake News Detection using Hugging Face and MLflow
- Dataset: WELFake
- Model A (Initial Champion): `distilbert-base-uncased`
- Model B (Challenger): `distilroberta-base`

## 2. Predefined Selection Rule
- Primary metric: Macro F1-score
- Minimum improvement required: 0.01
- Tie-breaker: Lower average inference latency
- Initial Champion: Model A
- If results are tied, retain Model A.

## 3. Final Experimental Results

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.9630 | 0.9870 |
| Precision | 0.9799 | 0.9808 |
| Recall | 0.9538 | 0.9964 |
| F1-score | 0.9667 | 0.9885 |
| Macro F1 | 0.9625 | 0.9868 |
| Average latency (ms) | 103.84 | 80.43 |
| P95 latency (ms) | 174.45 | 92.88 |

## 4. Selection Calculation
Macro F1 improvement:

0.9868 - 0.9625 = 0.0242

Required improvement = 0.01

Since 0.0242 is greater than 0.01, Model B satisfies the predefined selection rule.

## 5. Final Decision
- Selected Champion: Model B (`distilroberta-base`)
- Registered model: `fake-news-classifier`
- Champion alias: Version 2
- Challenger alias: Version 1

## 6. Conclusion
Model B achieved a higher Macro F1-score and lower average inference latency than Model A on the same test dataset. Therefore, Model B was promoted to Champion according to the predefined selection rule.