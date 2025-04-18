---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
## 👀 Hello! I’m **Ki Woong Moon**, 

Im a third-year Ph.D. student in **Linguistics** at the University of Arizona, where I’m also beginning my journey toward a **Master’s in Human Language Technology (HLT)**.

My research lives at the intersection of **phonetics**, **speech technology**, and **computational linguistics**. I’m especially interested in utilizing speech features into computational model to enhance the performance. Also, I am deeply intrigued in analyzing speech data using statistical methods to answer fascinating questions we haven't answered.

Lately, I’ve been diving deep into the phenomenon of **speech reduction** and exploring ways to adapt ASR models to more accurately interpret reduced and variable speech patterns. My goal is to bridge the gap between messy human speech and machine understanding.




What should be on this site
======
Remember that your professional website is intended to document what you did in your internship and in at least one other coding project, in order to share it with a technically savvy audience who might not be familiar with the details of the tools you used or the goals you were pursuing. That audience may include hiring managers, so this is your opportunity to demonstrate the marketable skills you've acquired through the MSHLT program. You want it to present your best professional self! Your website will also be evaluated by a faculty member and form part of your grade for your internship. Although the template repository will hold a stub for all the elements that are required, be sure to compare the site you build to the current evaluation rubric, which is available from the director of your program.

Getting started
======
1. See [README.md](https://github.com/uazhlt/professionalGHpages.github.io/blob/main/README.md) for getting set your repository set up and configured properly to render as a website.
1. Change this file to have the content you need for the *Professional introduction*--or, if you'd prefer to put that information onto a page that is *not* the landing page for your site, simply make it clear from this page where a reader could find that professional introduction. Think of it as a combination of a general-purpose cover letter in a job application and a personal statement for your application to the graduate program. It might briefly introduce your educational and work history, describe what you're currently working on, but also include the issues that motivate you and that you hope to work on in the future. Make sure your introduction meets the requirements outlined in the evaluation rubric.
1. Add information about your internship and at least one other coding project into files (either in Markdown or in HTML format) in the `_portfolio/` folder. You should find two stubs there, one in each format, but you can choose to have each of your portfolio files in whichever format you prefer. Make sure your write-ups meet the requirements outlined in the evaluation rubric.
1. Add the information that you'd like to have on your CV into `_pages/cv.md`. Make sure your CV meets the requirements outlined in the evaluation rubric.

Be sure to complete this information and submit your URL to the faculty evaluator by the deadline each term. The evaluator will prepare a feedback sheet for you to indicate how your site scored relative to the rubric. If there are significant points that need to be improved, you want to ensure you have time to receive that feedback and implement any revisions before the end of the term.

Site-wide configuration
------
The main configuration file for the site is in the base directory in [_config.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_config.yml), which defines the content in the sidebars and other site-wide features. You will need to replace the default values there with ones about yourself and your site's github repository; read this file carefully to make sure you've updated them all. The configuration file for the top menu is in [_data/navigation.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_data/navigation.yml). For example, if you want to add a section for *Talks* back into the site, you can add a navbar item for that (in navigation.yml) to add them back into the header. Make sure there's still a `talks.html` file in the _pages directory, and then add files for individual talks in the _talks directory. You'll follow a similar process to restore other navbar items or create your own.

Create content & metadata
------
For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For this template, only the _posts and _portfolio directories have been kept, but you can see the [original AcademicPages repository](https://github.com/academicpages/academicpages.github.io) for an example, if you'd like to add any of those categories back in.

If you choose to add some of these categories back to your site, you may find a lot of useful utilities to make updates to the site information work more smoothly. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

The creator of the AcademicPages template has also created [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the AcademicPages template. The sample CSVs in that directory are the ones used to create the site at stuartgeiger.com. His usual workflow is to keep a spreadsheet of publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository.

For more info
------
More info about configuring AcademicPages can be found in [the guide](https://academicpages.github.io/markdown/). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
