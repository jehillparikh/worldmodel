#!/bin/bash

# Exit on error
set -e

# Define variables
REPO="https://github.com/jehillparikh/worldmodel.git"
BRANCH="gh-pages"  # GitHub Pages uses 'gh-pages' or 'main/docs'
BUILD_DIR="_book"
DEPLOY_DIR="docs"  # GitHub Pages serves from 'docs' folder

# Ensure we're in the correct directory
cd "$(dirname "$0")"

echo "🚀 Rendering Bookdown project..."
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

echo "📂 Moving output to GitHub Pages folder..."
rm -rf $DEPLOY_DIR  # Remove old docs
mv $BUILD_DIR $DEPLOY_DIR

echo "📤 Committing and pushing changes..."
git add .
git commit -m "Deploy updated Bookdown project"
git push origin main  # Change branch if needed

echo "✅ Deployment successful! Your book should be live soon."
