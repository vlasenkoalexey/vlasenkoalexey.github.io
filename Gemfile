# frozen_string_literal: true

source "https://rubygems.org"

# The Chirpy theme (built by GitHub Actions, see .github/workflows/pages-deploy.yml).
gem "jekyll-theme-chirpy", "~> 7.5"

gem "html-proofer", "~> 5.0", group: :test

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:windows]
