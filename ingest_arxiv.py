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
    last_decision: Optional[RouteDecision]
    eval_score: Optional[float]
    eval_critique: Optional[str]

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

    state['last_decision'] = decision_obj

    if decision_obj.decision == "relevant":      
        return "analyze"
    return "end"

def analysis_node(state: TrackerState):
    print("--- PERFORMING DEEP ANALYSIS ---")
    state['revision_count'] = state.get('revision_count', 0) + 1

    current_paper = state['raw_data'][-1]
    decision_obj = state.get('last_decision')

    if current_paper and decision_obj:
        save_to_journal(current_paper, decision_obj)
        mark_as_processed(current_paper['url'])
        print(f"--- Journal Updated: {current_paper['title']} ---")

    scores = state.get('novelty_scores', [])
    scores.append(0.95) # fake score
    return {"novelty_scores": scores}

def evaluator_node(state: TrackerState):
    # retrieve the last decision and raw content
    decision = state['last_decision']
    content = state['paper_content']

    # get current score
    score = state.get('eval_score', 1.0)

    if state['eval_score'] < 0.7:
        return "needs_revision"
    return "done"
    
    # prompt the LLM to act as a judge
    prompt = f"Evaluate the reasoning: '{decision.reasoning}'. Content: {content}. Return a float score (0-1) and a critique."
    
    return {"eval_score": 0.95, "eval_critique": "Analysis is well-grounded."}

def save_to_journal(paper, decision_obj, eval_score, eval_critique):
    # appending a paper summary to the research journal
    with open("agentic_ai_journal.md", "a", encoding = "utf-8") as f:
        f.write(f"## {paper['title']}\n")
        f.write(f"- **URL:** {paper['url']}\n")
        f.write(f"- **Decision:** {decision_obj.decision.upper()}\n")
        f.write(f"- **Reasoning:** {decision_obj.reasoning}\n")
        f.write(f"- **Date Processed:** 2026-06-13\n\n")
        f.write("---\n")

        log_entry = {
        "title": paper['title'],
        "decision": decision_obj.decision,
        "reasoning": decision_obj.reasoning,
        "date": "2026-06-13"
    }
    with open("research_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# building the graph
workflow = StateGraph(TrackerState)

# adding the node
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("router", route_relevance)
workflow.add_node("analyze", analysis_node)
workflow.add_node("evaluator", evaluator_node)

# setting the path
workflow.add_edge(START, "ingestion")
workflow.add_edge("ingestion", "summarizer")
workflow.add_edge("analysis_node", "evaluator")


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

workflow.add_edge("evaluator", END)

# compiling
app = workflow.compile()

if __name__ == "__main__":
    initial_state = {"raw_data": [], "summaries": [], "novelty_scores": [], "paper_content": None}
    current_state = app.invoke(initial_state)
    
    app.invoke({
        "raw_data": [], 
        "summaries": [], 
        "novelty_scores": [], 
        "paper_content": None,
        "eval_score": 1.0
    })
    print("--- Research Session Complete ---")