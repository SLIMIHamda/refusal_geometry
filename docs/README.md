# Documentation

**[`TECHNICAL_REFERENCE.pdf`](TECHNICAL_REFERENCE.pdf)** — the primary technical and scientific
reference for the `asw` harness (26 pages). It is a hybrid research paper, software-architecture
guide, and developer/reproduction manual: each module is motivated scientifically before its
design and implementation are examined, with 9 diagrams, 20 tables, code excerpts, a complete
end-to-end experimental-workflow section, and a standalone Kaggle execution manual.

## Regenerating the PDF

The document is generated programmatically (no LaTeX/browser needed) from
[`reportlab`](https://pypi.org/project/reportlab/) + [`matplotlib`](https://matplotlib.org/):

```bash
pip install reportlab matplotlib pillow pymupdf
python docs/_build/make_figs.py     # -> docs/_build/figs/*.png   (the diagrams)
python docs/_build/build_pdf.py     # -> docs/TECHNICAL_REFERENCE.pdf
```

`make_figs.py` draws the ten architecture/pipeline/sequence diagrams; `build_pdf.py` lays out
the document (cover, hyperlinked table of contents, PDF bookmarks, tables, code blocks, and the
embedded figures). Both read nothing from the network and run on a CPU-only machine.

When the codebase changes materially, update the prose in `build_pdf.py`, re-run both scripts,
and commit the refreshed `TECHNICAL_REFERENCE.pdf`.
