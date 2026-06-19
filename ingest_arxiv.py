import arxiv
import json
import time
import os
import logging
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from datetime import datetime
from config import SEARCH_QUERY, MAX_RESULTS, LOG_FILE, URL_REGISTRY

llm = ChatOllama(
    model = "llama3.2",
    num_predict = 50, 
    temperature = 0  
) 

logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    reasoning: str = Field(description = "Brief explanation of why the paper is relevant or not.")
    decision: Literal["relevant", "irrelevant"] = Field(description = "Must be 'relevant' or 'irrelevant'. Do not include extra text.")

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
    client = arxiv.Client(
        page_size = 100,
        delay_seconds = 3.0,
        num_retries = 3
    )
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
    logger.info("--- FETCHING DATA ---")
    # call existing function here
    papers = fetch_arxiv_papers("Agentic AI", max_results = 3) 

    # memory of what we've seen
    processed_urls = get_processed_urls()
    
    # keep papers not in processed_papers.txt
    new_papers = [p for p in papers if p['url'] not in processed_urls]
    
    return {"raw_data": new_papers}

def summarizer_node(state: TrackerState):
    logger.info("--- SUMMARIZING PAPERS ---")

    raw_data = state.get('raw_data', [])

    if not raw_data:
        logger.info("--- NO PAPERS TO SUMMARIZE ---")
        return {"summaries": [], "paper_content": ""}

    paper = raw_data[0]
    summaries = f"Summary of {paper['title']}: {paper['summary']}"
    return {"summaries": summaries, "paper_content": summaries}

# if "agentic" is in the summary, analyze it further
def route_relevance(state: TrackerState):
    content = state.get('paper_content') or ""

    if not content:
        return "end"

    prompt = (
        f"Analyze the following paper content and decide if it is relevant to 'Agentic AI'. "
        f"Output must be a valid JSON object with 'reasoning' and 'decision' fields. "
        f"The 'decision' field must be either 'relevant' or 'irrelevant'.\n\n"
        f"Content: {content[:1000]}"
    )
    try:
        decision_obj = structured_llm.invoke(prompt)
        state['last_decision'] = decision_obj

        logger.info(f"--- LLM Reasoning: {decision_obj.reasoning} ---")
        logger.info(f"--- Decision: {decision_obj.decision} ---")  

        if decision_obj.decision == "relevant":      
            return "analyze"
    except Exception as e:
        logger.info(f"--- LLM Parsing Error: {e} ---")
    return "end"

def route_after_ingestion(state: TrackerState):
    if not state.get("raw_data"):
        return END
    return "summarizer"

def analysis_node(state: TrackerState):
    logger.info("--- PERFORMING DEEP ANALYSIS ---")
    state['revision_count'] = state.get('revision_count', 0) + 1

    current_paper = state['raw_data'][-1]
    decision_obj = state.get('last_decision')

    if current_paper and decision_obj:
        save_to_journal(current_paper, decision_obj)
        mark_as_processed(current_paper['url'])
        logger.info(f"--- Journal Updated: {current_paper['title']} ---")

    scores = state.get('novelty_scores', [])
    scores.append(0.95) # fake score
    return {"novelty_scores": scores}

def evaluator_node(state: TrackerState):
    # retrieve the last decision and raw content
    decision = state.get('last_decision')
    content = state.get('paper_content', ' ')
    paper = state['raw_data'][-1] if state.get('raw_data') else None

    # get current score
    score = state.get('eval_score', 1.0)

    eval_score = 0.95
    eval_critique = "Analysis is well-grounded."

    if paper:
        save_to_journal(paper, decision, eval_score, eval_critique)

        mark_as_processed(paper['url'])
        logger.info(f"DEBUG: Marked {paper['url']} as processed.")
        
    return {"eval_score": eval_score, "eval_critique": eval_critique}

def filter_node(state: TrackerState):
    # getting the raw papers
    papers = state.get('raw_data', [])
    
    # defining keywords with a "High Quality" paper for research
    signal_keywords = ["agentic", "multi-agent", "scalability", "production"]
    
    # keeping only papers with one or more keywords in the summary
    filtered_papers = [
        p for p in papers 
        if any(word in p['summary'].lower() for word in signal_keywords)
    ]
    
    logger.info(f"--- FILTERED: {len(papers)} -> {len(filtered_papers)} relevant papers ---")
    return {"raw_data": filtered_papers}

def save_to_journal(paper, decision_obj, eval_score, eval_critique):
    # appending a paper summary to the research journal
    with open("agentic_ai_journal.md", "a", encoding = "utf-8") as f:
        f.write(f"## {paper['title']}\n")
        f.write(f"- **URL:** {paper['url']}\n")
        f.write(f"- **Decision:** {decision_obj.decision.upper()}\n")
        f.write(f"- **Reasoning:** {decision_obj.reasoning}\n")
        f.write(f"- **Score:** {eval_score}\n")
        f.write(f"- **Critique:** {eval_critique}\n")
        f.write(f"- **Date Processed:** 2026-06-13\n\n")
        f.write("---\n")

        log_entry = {
        "title": paper['title'],
        "url": paper['url'],
        "decision": decision_obj.decision,
        "reasoning": decision_obj.reasoning,
        "eval_score": eval_score,
        "eval_critique": eval_critique,
        "date": "2026-06-13"
    }
    with open("research_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def mark_as_processed(url):
    """Appends a URL to the processed_papers.txt file."""
    with open("processed_papers.txt", "a") as f:
        f.write(f"{url}\n")

# building the graph
workflow = StateGraph(TrackerState)

# adding the node
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("filter", filter_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("router", route_relevance)
workflow.add_node("analyze", analysis_node)
workflow.add_node("evaluator", evaluator_node)

# setting the path
workflow.add_edge(START, "ingestion")

workflow.add_conditional_edges("ingestion", route_after_ingestion, {
    "summarizer": "summarizer",
    END: END
})

workflow.add_edge("filter", "summarizer")

# connect summarizer to router
workflow.add_edge("summarizer", "router")

workflow.add_conditional_edges(
    "router",
    route_relevance,
    {
        "analyze": "analyze",
        "end": END
    }
)

workflow.add_edge("analyze", "evaluator")
workflow.add_edge("evaluator", END)

# compiling
app = workflow.compile()