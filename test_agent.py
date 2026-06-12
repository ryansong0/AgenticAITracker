from ingest_arxiv import app

state_relevant = {"paper_content": "This paper proposes a new architecture for Large Language Models..."}
state_irrelevant = {"paper_content": "World Cup."}

result = app.invoke(state_relevant)
print(f"Result for relevant: {result['route']}")