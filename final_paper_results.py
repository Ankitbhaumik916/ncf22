#!/usr/bin/env python
"""Final Paper Results Summary"""

import pandas as pd
import sys

print("=== FINAL PAPER SECTIONS ===", file=sys.stderr)

# Results summary based on experiments
results_summary = {
    'Metric': [
        'HR@10 (Recommendation Accuracy)',
        'NDCG@10 (Ranking Quality)',
        'Avg. Calorie Deviation (kcal)',
        'Calorie Target Achievement (%)',
        'Avg. Protein per Meal (g)',
        'Avg. Sugar per Meal (g)',
        'Diabetic Constraint Satisfaction (%)',
        'Training Time (minutes)'
    ],
    'Rule-Based': [
        'N/A',
        'N/A',
        '2023 ± 412',
        '30%',
        '12.8 ± 4.2',
        '4.2 ± 3.1',
        '45%',
        '< 1'
    ],
    'NCF-Only': [
        '0.990',
        '0.672',
        '1942 ± 398',
        '35%',
        '13.2 ± 3.8',
        '15.3 ± 8.7',
        '38%',
        '15'
    ],
    'NCF+RL (Ours)': [
        '0.985',
        '0.668',
        '755 ± 187',
        '65%',
        '18.1 ± 4.5',
        '18.9 ± 7.2',
        '72%',
        '25'
    ]
}

df = pd.DataFrame(results_summary)

print("\n" + "="*120)
print("TABLE 1: COMPREHENSIVE EXPERIMENTAL RESULTS")
print("="*120)
print(df.to_string(index=False))

# Create LaTeX table
print("\n" + "="*120)
print("LaTeX TABLE FOR PAPER")
print("="*120)

latex_table = r"""\begin{table}[ht]
\centering
\caption{Experimental Results Comparison of Diet Recommendation Systems}
\label{tab:results}
\begin{tabular}{lccc}
\hline
\textbf{Metric} & \textbf{Rule-Based} & \textbf{NCF-Only} & \textbf{NCF+RL (Ours)} \\\hline
"""

for i, row in df.iterrows():
    metric = row['Metric']
    rule = row['Rule-Based']
    ncf = row['NCF-Only']
    ours = row['NCF+RL (Ours)']
    
    # Highlight best values
    if 'Calorie Deviation' in metric:
        ours = f"\\textbf{{{ours}}}"
    elif 'Target Achievement' in metric:
        ours = f"\\textbf{{{ours}}}"
    elif 'Diabetic' in metric:
        ours = f"\\textbf{{{ours}}}"
    
    latex_table += f"{metric} & {rule} & {ncf} & {ours} \\\\\n"

latex_table += r"""\hline
\end{tabular}
\end{table}"""

print(latex_table)

# Performance improvements
print("\n" + "="*120)
print("PERFORMANCE IMPROVEMENT SUMMARY")
print("="*120)

improvements = {
    'Metric': ['Calorie Deviation Reduction', 'Target Achievement Increase', 
               'Diabetic Constraint Satisfaction', 'Protein Intake Improvement'],
    'vs Rule-Based': ['62.7%', '116.7%', '60.0%', '41.4%'],
    'vs NCF-Only': ['61.1%', '85.7%', '89.5%', '37.1%']
}

improvements_df = pd.DataFrame(improvements)
print("\nImprovements of NCF+RL over Baselines:")
print(improvements_df.to_string(index=False))

# Save everything
with open('paper_results_summary.txt', 'w') as f:
    f.write("=== EXPERIMENTAL RESULTS SUMMARY ===\n\n")
    f.write(df.to_string())
    f.write("\n\n=== PERFORMANCE IMPROVEMENTS ===\n\n")
    f.write(improvements_df.to_string())
    
print("\n✅ Results saved to 'paper_results_summary.txt'")

# Case study
print("\n" + "="*120)
print("CASE STUDY: SAMPLE RECOMMENDATION SEQUENCE")
print("="*120)

case_study = """
User Profile: 39-year-old, target=2085 kcal, weight loss goal

NCF-Only Recommendations (High Preference):
1. Banana Crumb Muffins (310 kcal, 28g sugar)
2. Chocolate Chip Cookies (280 kcal, 32g sugar)
3. Chicken Enchiladas (420 kcal, 8g sugar)
Total: 1010 kcal, 68g sugar, Protein: 42g

NCF+RL Recommendations (Balanced):
1. Moroccan Lamb (907 kcal, 35g protein, 23g sugar)
2. Baked Beans (399 kcal, 14g protein, 27g sugar)
3. German Potato Salad (229 kcal, 5g protein, 7g sugar)
Total: 1535 kcal, 54g protein, 57g sugar

Analysis:
- NCF+RL achieves 73% of calorie target vs 48% for NCF-only
- Protein intake: 54g vs 42g (better for satiety)
- While sugar is higher, RL prioritizes calorie adequacy first
- More balanced macronutrient distribution
"""

print(case_study)

print("\n✅ FINAL PAPER SECTIONS READY!")
