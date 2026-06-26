import arxiv
from datetime import datetime
from core.models import Document


def fetch_arxiv(arxiv_id: str) -> Document:
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[arxiv_id])))

    if not results:
        raise ValueError(f"No paper found for arxiv id: {arxiv_id}")

    paper = results[0]

    return Document(
        source="arxiv",
        title=paper.title,
        url=paper.entry_id,
        content=paper.summary,
        published_at=paper.published,
        tier=1,
        authors=[a.name for a in paper.authors],
        categories=paper.categories,
        knowledge_type="research_paper",
    )