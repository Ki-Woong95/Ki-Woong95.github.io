# Academic website refresh — October 2026

This ZIP updates the existing Jekyll / Academic Pages site; no theme migration was made.

## Modified files

- `_pages/about.md`: updated introduction and research summary.
- `_pages/research.md`: new consolidated research, publications, presentations and data page.
- `_pages/teaching.md`: teaching page populated from academic CV.
- `_pages/cv.md`: short CV page with working PDF link.
- `_pages/publications.md`: pointer from previous `/publications/` address to the new page.
- `_pages/portfolio.html`: older class / internship projects are labeled as an archive.
- `_data/navigation.yml`: four-link academic navigation.
- `_config.yml`: updated academic description, sidebar biography and contact email.
- `files/MKW_CV.pdf`: replaced with user's 3-page October 2026 academic CV.
- `_posts/*.md`: removed the five stock/demo blog posts.

## Before publishing

1. Review About and Research text, particularly the status of manuscripts and preliminary speech-reasoning work.
2. Confirm the two Research Data Repository DOIs and the arXiv link resolve correctly.
3. Check the avatar photo and public contact information.
4. Update `files/MKW_CV.pdf` whenever the CV changes.
5. To publish, copy/commit the updated repository contents to the GitHub Pages repo's publishing branch (do not upload the enclosing ZIP itself as a file).
6. The project retains its existing Jekyll setup. If Jekyll dependencies are installed, test with `bundle exec jekyll build` before pushing.

## Research status terminology

- POMA article: published.
- ICASSP 2027 / ICPhS 2027: submitted (not accepted).
- Two manuscript entries: in preparation.
- Speech reasoning: ongoing; preliminary claims and performance numbers are deliberately omitted from the website.

## Legacy materials

The three older portfolio articles are retained in their original paths but are no longer in the top navigation. The generic theme demo pages are untouched; remove them later only after you are confident they are not needed.

- Replaced the template-focused README with maintenance instructions; removed three template demo pages, three dummy paper PDFs, and an unrelated internship rubric PDF.
- The legacy `/talks/` page now points to the conference presentation list; removed the unused talk-map page and the template privacy-policy text.
