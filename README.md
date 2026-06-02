# vlasenkoalexey.github.io

Personal blog, published at <https://vlasenkoalexey.github.io>.

Built with [Jekyll](https://jekyllrb.com/) (minima theme) and hosted on GitHub
Pages. Posts were migrated from the original Blogger site
[alekseyv-dev.blogspot.com](https://alekseyv-dev.blogspot.com/).

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "Post title"
date: 2020-03-12
tags: [Tag1, Tag2]
---
```

Images live under `assets/images/<post-slug>/`.

## Running locally

```bash
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>. GitHub Pages rebuilds automatically on push
to the default branch.
