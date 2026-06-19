import pandas as pd
import json

def analyze_trends():
    data = [json.loads(line) for line in open("research_log.jsonl")]
    df = pd.DataFrame(data)
    
    print("--- RESEARCH TRENDS ---")
    print(df['decision'].value_counts())