from ingest_arxiv import app

def run_test(content, description):
    print(f"\n--- Testing: {description} ---")
    result = app.invoke({"paper_content": content})
    print(f"Decision: {result.get('route', 'No route found')}")

run_test("Agentic AI Tracker Project", "Relevant AI Paper")
run_test("Chairs and tables are good", "Irrelevant Information to the World Cup")