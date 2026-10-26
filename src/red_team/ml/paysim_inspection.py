import pandas as pd
import numpy as np
import json
import os

def analyze_paysim(csv_path: str, output_json: str):
    print("Loading PaySim...")
    df = pd.read_csv(csv_path)
    
    stats = {}
    
    # 2. Schema
    stats["rows"] = len(df)
    stats["columns"] = len(df.columns)
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    stats["missing"] = df.isnull().sum().to_dict()
    stats["unique"] = df.nunique().to_dict()
    
    # 3. Legitimate vs Fraud
    fraud_counts = df['isFraud'].value_counts()
    legit_rows = int(fraud_counts.get(0, 0))
    fraud_rows = int(fraud_counts.get(1, 0))
    stats["total_rows"] = len(df)
    stats["legit_rows"] = legit_rows
    stats["fraud_rows"] = fraud_rows
    stats["legit_pct"] = legit_rows / len(df)
    stats["fraud_pct"] = fraud_rows / len(df)
    
    # Legit only df
    df_legit = df[df['isFraud'] == 0]
    
    # 4. Time / Step Analysis
    stats["step_min"] = int(df['step'].min())
    stats["step_max"] = int(df['step'].max())
    stats["step_unique"] = df['step'].nunique()
    stats["step_repeated"] = len(df) - stats["step_unique"]
    stats["rows_per_step_mean"] = df.groupby('step').size().mean()
    stats["approx_sim_duration"] = "{} hours".format(stats["step_max"])
    
    def calc_recurrence(series):
        vc = series.value_counts()
        total_events = int(vc.sum())
        unique_entities = len(vc)
        min_events = int(vc.min())
        max_events = int(vc.max())
        mean_events = float(vc.mean())
        median_events = float(vc.median())
        return {
            "total_events": total_events,
            "unique_entities": unique_entities,
            "min_events": min_events,
            "max_events": max_events,
            "mean_events": mean_events,
            "median_events": median_events,
            "pct_ge_2": float((vc >= 2).mean()),
            "pct_ge_3": float((vc >= 3).mean()),
            "pct_ge_5": float((vc >= 5).mean()),
            "pct_ge_10": float((vc >= 10).mean()),
            "pct_ge_20": float((vc >= 20).mean()),
            "buckets": {
                "1": int((vc == 1).sum()),
                "2": int((vc == 2).sum()),
                "3-4": int(((vc >= 3) & (vc <= 4)).sum()),
                "5-9": int(((vc >= 5) & (vc <= 9)).sum()),
                "10-19": int(((vc >= 10) & (vc <= 19)).sum()),
                "20+": int((vc >= 20).sum())
            }
        }
    
    # 5. Origin / Customer-like Recurrence (Legit Only)
    stats["origin"] = calc_recurrence(df_legit['nameOrig'])
    
    # 6. Destination Recurrence (Legit Only)
    stats["destination_all"] = calc_recurrence(df_legit['nameDest'])
    
    df_m = df_legit[df_legit['nameDest'].str.startswith('M')]
    df_c = df_legit[df_legit['nameDest'].str.startswith('C')]
    
    stats["destination_M"] = calc_recurrence(df_m['nameDest'])
    stats["destination_C"] = calc_recurrence(df_c['nameDest'])
    
    # 9. Transaction type dynamics
    stats["type_frequencies"] = df['type'].value_counts(normalize=True).to_dict()
    stats["type_mean_amount"] = df.groupby('type')['amount'].mean().to_dict()
    
    # 10. Entity persistence across steps (Legit Only)
    # Count distinct steps per nameOrig
    orig_steps = df_legit.groupby('nameOrig')['step'].nunique()
    stats["origin_distinct_steps"] = {
        "mean": float(orig_steps.mean()),
        "max": int(orig_steps.max()),
        "pct_ge_2": float((orig_steps >= 2).mean())
    }
    
    # Count distinct steps per C-destination
    c_dest_steps = df_c.groupby('nameDest')['step'].nunique()
    stats["c_dest_distinct_steps"] = {
        "mean": float(c_dest_steps.mean()),
        "max": int(c_dest_steps.max()),
        "pct_ge_2": float((c_dest_steps >= 2).mean())
    }

    # Count distinct steps per M-destination
    m_dest_steps = df_m.groupby('nameDest')['step'].nunique()
    stats["m_dest_distinct_steps"] = {
        "mean": float(m_dest_steps.mean()),
        "max": int(m_dest_steps.max()),
        "pct_ge_2": float((m_dest_steps >= 2).mean())
    }
    
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print("Done")

if __name__ == "__main__":
    analyze_paysim(
        csv_path="scratch/paysim/paysim dataset.csv",
        output_json="scratch/paysim_stats.json"
    )
