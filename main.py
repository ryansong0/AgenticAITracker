from ingest_arxiv import app
from generate_report import generate_report
import logging

# configure logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')

def run_research():
    logging.info("Starting automated research session...")
    
    # run the graph
    initial_state = {"raw_data": [], "summaries": [], "novelty_scores": [], "paper_content": None}
    app.invoke(initial_state)
    
    # generate the final report
    logging.info("Research session finished. Generating report...")
    generate_report()

if __name__ == "__main__":
    run_research()