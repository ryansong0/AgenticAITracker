import arxiv
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class TrackerState(TypedDict):
    raw_data: List[dict] # What you fetch
    summaries: List[str] # What your LLM adds
    novelty_scores: List[float] # What your analyzer adds

from langgraph.graph import StateGraph, START, END

# defining the node
def ingestion_node(state: TrackerState):
    print("--- FETCHING DATA ---")
    # call existing function here
    papers = fetch_arxiv_papers() 
    return {"raw_data": papers}

# building the graph
workflow = StateGraph(TrackerState)

# adding the node
workflow.add_node("ingestion", ingestion_node)

# setting the path
workflow.add_edge(START, "ingestion")
workflow.add_edge("ingestion", END)

# compiling
app = workflow.compile()

def fetch_arxiv_papers(query, max_results = 3):
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()
    papers_list = []

    print(f"--- Fetching the latest papers for: '{query}' ---\n")

    for result in client.results(search):
        paper_data = {
            "title": result.title,
            "published": str(result.published.date()),
            "summary": result.summary,
            "url": result.entry_id
        }
        papers_list.append(paper_data)

        print(f"Fetched: {result.title}")

    with open('papers.json', 'w', encoding='utf-8') as f:
        json.dump(papers_list, f, indent=4)

    print(f"\nSuccessfully saved {len(papers_list)} papers to papers.json!")


if __name__ == "__main__":
    fetch_arxiv_papers("Agentic AI")