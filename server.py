import logging
import json
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from typing import Optional
# import the compiled LangGraph app from your script
from ingest_arxiv import app as graph_app
from apscheduler.schedulers.background import BackgroundScheduler

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrackerServer")

# initialize FastAPI
app = FastAPI(
    title = "Agentic AI Research Tracker API",
    description = "Production-grade API layer to trigger and monitor the LangGraph research ingestion pipeline.",
    version = "1.0.0"
)

# define the request schema for incoming API calls
class TriggerPipelineRequest(BaseModel):
    query: Optional[str] = "Agentic AI"
    max_results: Optional[int] = 3

def run_pipeline_in_background(query: str, max_results: int):
    """
    Executes the compiled LangGraph workflow synchronously in a background thread.
    """
    logger.info(f"Background task started for query: '{query}'")
    initial_state = {
        "raw_data": [],
        "summaries": [],
        "novelty_scores": [],
        "paper_content": "",
        "last_decision": None,
        "eval_score": None,
        "eval_critique": ""
    }
    try:
        # fire the graph execution
        graph_app.invoke(initial_state)
        logger.info("Background pipeline execution completed successfully.")
    except Exception as e:
        logger.error(f"Background pipeline failed: {str(e)}")

scheduler = BackgroundScheduler()

scheduler.add_job(
    run_pipeline_in_background, 
    trigger = "interval", 
    hours = 12, 
    id = "arxiv_tracker_cron",
    replace_existing = True
)


@app.post("/trigger", status_code=202)
async def trigger_pipeline(payload: TriggerPipelineRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to kick off the research pipeline asynchronously.
    Returns HTTP 202 Accepted immediately so the client isn't left hanging.
    """
    logger.info("Received request to trigger pipeline.")
    
    # offload the heavy graph execution to background tasks
    background_tasks.add_task(
        run_pipeline_in_background, 
        query=payload.query, 
        max_results=payload.max_results
    )
    
    return {
        "status": "accepted",
        "message": f"Pipeline processing for query '{payload.query}' has been initiated in the background."
    }

@app.get("/health")
async def health_check():
    """Simple status check for monitoring tools."""
    return {"status": "healthy", "service": "agentic-tracker-api"}

@app.get("/journal", status_code = 200)
async def get_research_journal():
    """
    retrieves all historically processed papers and their LLM evaluation scores 
    directly from the local database log.
    """
    log_file_path = "research_log.jsonl"
    
    # if no papers have been successfully processed yet, return an empty list
    if not os.path.exists(log_file_path):
        return {"total_papers": 0, "papers": []}
        
    papers = []
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    papers.append(json.loads(line.strip()))
                    
        return {
            "total_papers": len(papers),
            "papers": papers[::-1]  # reverse the list so the newest papers show up at the top
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Failed to read research log: {str(e)}")