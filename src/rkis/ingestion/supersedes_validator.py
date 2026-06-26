from storage.sqlite_repo import SQLiteDocumentRepository

SIMILARITY_THRESHOLD = 0.3

_doc_repo = SQLiteDocumentRepository()


def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def validate_supersedes(
    current_abstract: str,
    prior_doc_id: str | None = None,
    prior_abstract: str | None = None,
) -> bool:
    if prior_abstract is None and prior_doc_id is not None:
        prior_doc = _doc_repo.get_by_id(prior_doc_id)
        if prior_doc is None:
            return False
        prior_abstract = prior_doc.content

    if not prior_abstract or not current_abstract:
        return False

    try:
        model = _get_cross_encoder()
    except Exception:
        return False

    score = float(model.predict([(current_abstract, prior_abstract)]))
    return score >= SIMILARITY_THRESHOLD
