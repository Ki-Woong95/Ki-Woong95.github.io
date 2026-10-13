# Ki Woong Moon — Academic Website

Source for [ki-woong95.github.io](https://ki-woong95.github.io/), an academic website maintained with **Jekyll** and **GitHub Pages**.

## Main pages

| Page | Source file |
| --- | --- |
| About | `_pages/about.md` |
| Research & Publications | `_pages/research.md` |
| Teaching | `_pages/teaching.md` |
| Curriculum Vitae | `_pages/cv.md` |

The primary navigation is in `_data/navigation.yml`. Sidebar details and site description are in `_config.yml`. The PDF download is at `files/MKW_CV.pdf`.

**Research statuses need active maintenance:** do not describe submitted manuscripts as accepted or published. The site's Research page includes ongoing projects, but does not imply a finished publication.

## Maintaining the site

1. Edit the relevant page in `_pages/`.
2. When the CV changes, replace `files/MKW_CV.pdf` and update the date in `_pages/cv.md`.
3. Check manuscript and conference statuses after decisions arrive.
4. Commit and push changes to the configured GitHub Pages publishing branch.
5. Verify the live site after GitHub Pages finishes publishing.

To build locally, install Ruby/Bundler and the dependencies in `Gemfile`, then run `bundle exec jekyll build` or `bundle exec jekyll serve`.

The old `/portfolio/` route remains as an archive of earlier coursework and internship projects and is not linked from the primary navigation. The former `/publications/` route points visitors to the new combined Research & Publications page.

This repository was originally derived from the [Academic Pages](https://academicpages.github.io/) / [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme. Original licensing and theme components have been retained.

For a detailed list of the October 2026 edits, see [`SITE_UPDATE_NOTES.md`](SITE_UPDATE_NOTES.md).
