import arxiv
import json

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