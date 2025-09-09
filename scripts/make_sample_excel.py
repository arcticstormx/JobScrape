import os
import pandas as pd

os.makedirs('Jobs', exist_ok=True)
rows = [
    {
        'title': 'Data Engineer',
        'company': 'CloudCo',
        'location': 'Remote',
        'description': 'Build data pipelines in Python, Airflow, AWS. Optimize ETL and Pandas workflows.'
    },
    {
        'title': 'NLP Engineer',
        'company': 'AI Labs',
        'location': 'Boston, MA',
        'description': 'Develop NLP models, embeddings, and deploy ML services. Experience with transformers and vector search.'
    },
    {
        'title': 'Frontend Developer',
        'company': 'WebX',
        'location': 'NYC',
        'description': 'React, CSS, JavaScript for UI development.'
    },
]

df = pd.DataFrame(rows)
with pd.ExcelWriter('Jobs/all_jobs.xlsx') as writer:
    df.to_excel(writer, sheet_name='All', index=False)
print('wrote Jobs/all_jobs.xlsx with', len(df), 'rows')

