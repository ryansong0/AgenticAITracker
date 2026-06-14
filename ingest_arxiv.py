import arxiv
import json
import os
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    num_predict = 50, 
    temperature = 0  
) 

class RouteDecision(BaseModel):
    reasoning: str = Field(description = "Brief explanation of why the paper is relevant or not.")
    decision: Literal["relevant", "irrelevant"] = Field(description = "Must be 'relevant' or 'irrelevant'")

structured_llm = llm.with_structured_output(RouteDecision)

class TrackerState(TypedDict):
    raw_data: List[dict] # What you fetch
    summaries: List[str] # What your LLM adds
    novelty_scores: List[float] # What your analyzer adds
    paper_content: Optional[str]

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

def get_processed_urls():
    if not os.path.exists("processed_papers.txt"):
        return set()
    with open("processed_papers.txt", "r") as f:
        return set(line.strip() for line in f)

def mark_as_processed(url):
    with open("processed_papers.txt", "a") as f:
        f.write(f"{url}\n")

# defining the node
def ingestion_node(state: TrackerState):
    print("--- FETCHING DATA ---")
    # call existing function here
    papers = fetch_arxiv_papers("Agentic AI") 

    # memory of what we've seen
    processed_urls = get_processed_urls()
    
    # keep papers not in processed_papers.txt
    new_papers = [p for p in papers if p['url'] not in processed_urls]
    
    print(f"Found {len(new_papers)} new papers to process.")

    return {"raw_data": new_papers}

def summarizer_node(state: TrackerState):
    print("--- SUMMARIZING PAPERS ---")
    summaries = [f"Summary of {paper['title']}" for paper in state['raw_data']]
    return {"summaries": summaries}

# if "agentic" is in the summary, analyze it further
def route_relevance(state: TrackerState):
    content = state.get('paper_content') or ""
    prompt = f"Analyze the following paper content and decide if it is relevant to 'Agentic AI': \n\n{content}"
    decision_obj = structured_llm.invoke(prompt)

    print(f"--- LLM Reasoning: {decision_obj.reasoning} ---")
    print(f"--- Decision: {decision_obj.decision} ---")  

    if decision_obj.decision == "relevant":      
        return "analyze"
    return "end"

def analysis_node(state: TrackerState):
    print("--- PERFORMING DEEP ANALYSIS ---")
    # get URL of latest paper being analyzed
    if state['raw_data']:
        latest_url = state['raw_data'][-1]['url']
        mark_as_processed(latest_url)
        print(f"--- Paper marked as processed: {latest_url} ---")

    scores = state.get('novelty_scores', [])
    scores.append(0.95) # fake score
    return {"novelty_scores": scores}

# building the graph
workflow = StateGraph(TrackerState)

# adding the node
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("router", route_relevance)
workflow.add_node("analyze", analysis_node)

# setting the path
workflow.add_edge(START, "ingestion")
workflow.add_edge("ingestion", "summarizer")


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
    initial_state = {"raw_data": [], "summaries": [], "novelty_scores": [], "paper_content": None}
    current_state = app.invoke(initial_state)
    
    for paper in current_state['raw_data']:
        print(f"\n--- Processing: {paper['title']} ---")
        app.invoke({"paper_content": paper['summary']})