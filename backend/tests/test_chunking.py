from app.services.chunking import split_pages, split_text


def test_split_pages_overlap_and_pages():
    pages = [(1, "A" * 500 + ". " + "B" * 500), (2, "C" * 200)]
    chunks = split_pages(pages, chunk_size=300, chunk_overlap=60)
    assert len(chunks) >= 3
    assert chunks[0].page == 1
    assert all(c.text for c in chunks)


def test_split_markdown_prefers_paragraphs():
    md = "# Titulo\n\n" + ("parrafo uno. " * 40) + "\n\n## Sec\n\n" + ("mas texto. " * 40)
    chunks = split_text(md, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2


def test_overlap_validation():
    try:
        split_pages([(1, "hola")], chunk_size=10, chunk_overlap=10)
        assert False, "debía fallar"
    except ValueError:
        pass
