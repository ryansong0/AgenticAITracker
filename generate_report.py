import json
import os

def generate_report():
    report_path = "research_summary.md"
    with open("research_log.jsonl", "r") as f:
        entries = [json.loads(line) for line in f]

    with open(report_path, "w") as f:
        f.write("# Daily Research Digest\n\n")
        for entry in entries:
            f.write(f"### {entry.get('title', 'No Title')}\n")
            f.write(f"- **Score:** {entry.get('eval_score')}\n")
            f.write(f"- **Summary:** {entry.get('summary')}\n\n")
    print(f"--- REPORT GENERATED: {report_path} ---")

if __name__ == "__main__":
    generate_report()