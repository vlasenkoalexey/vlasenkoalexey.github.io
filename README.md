# vlasenkoalexey.github.io

Personal blog, published at <https://vlasenkoalexey.github.io>.

Built with [Jekyll](https://jekyllrb.com/) and the
[Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme (dark mode),
built by GitHub Actions and hosted on GitHub Pages. Posts were migrated from the
original Blogger site [alekseyv-dev.blogspot.com](https://alekseyv-dev.blogspot.com/).

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "Post title"
date: 2020-03-12
categories: [Main, Sub]   # up to two levels, shown in the Categories tab
tags: [tag1, tag2]
---
```

Images live under `assets/images/<post-slug>/`.

## Running locally

Requires Ruby 3.x and Bundler.

```bash
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

## Deployment

Pushing to `main` triggers `.github/workflows/pages-deploy.yml`, which builds the
site with Chirpy and deploys it to GitHub Pages. The repository's **Settings →
Pages → Build and deployment → Source** must be set to **GitHub Actions**.
