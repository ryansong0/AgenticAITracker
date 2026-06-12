import arxiv
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")


class TrackerState(TypedDict):
    raw_data: List[dict] # What you fetch
    summaries: List[str] # What your LLM adds
    novelty_scores: List[float] # What your analyzer adds

def fetch_arxiv_papers(query, max_results = 3):
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    return[{"title": r.title,
            "published": str(r.published.date()),
            "summary": r.summary,
            "url": r.entry_id}
            for r in client.results(search)]

# defining the node
def ingestion_node(state: TrackerState):
    print("--- FETCHING DATA ---")
    # call existing function here
    papers = fetch_arxiv_papers("Agentic AI") 
    return {"raw_data": papers}

def summarizer_node(state: TrackerState):
    print("--- SUMMARIZING PAPERS ---")
    summaries = [f"Summary of {paper['title']}" for paper in state['raw_data']]
    return {"summaries": summaries}

# if "agentic" is in the summary, analyze it further
def route_relevance(state: TrackerState):
    last_summary = state['summaries'][-1].lower()
    if "agentic" in last_summary:
        return "analyze"
    return "end"

def analysis_node(state: TrackerState):
    print("--- PERFORMING DEEP ANALYSIS ---")
    state['novelty_scores'].append(0.95) # fake score
    return {"novelty_scores": state['novelty_scores']}

# building the graph
workflow = StateGraph(TrackerState)

# adding the node
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("analyze", analysis_node)

# setting the path
workflow.add_edge(START, "ingestion")
workflow.add_edge("ingestion", "summarizer")


workflow.add_node("router", route_relevance) # Optional: only if you want a dedicated router node, otherwise use the function directly in edges

# adding the conditional edges
workflow.add_conditional_edges(
    "summarizer",
    route_relevance,
    {
        "analyze": "analyze",
        "end": END
    }
)

# adding the analysis end point
workflow.add_edge("analyze", END)

# compiling
app = workflow.compile()

if __name__ == "__main__":
    initial_state = {"raw_data": [], "summaries": [], "novelty_scores": []}
    final_state = app.invoke(initial_state)
    
    print("\n--- FINAL STATE ---")
    print(f"Fetched {len(final_state['raw_data'])} papers.")
    print(f"Generated {len(final_state['summaries'])} summaries.")