#!/bin/bash

# Step 1: Navigate to the R project directory
cd "$(dirname "$0")"  # This makes sure the script runs from the directory where it's located

# Step 2: Check if R is installed
if ! command -v R &> /dev/null
then
    echo "R is not installed. Please install R to proceed."
    exit 1
fi

# Step 3: Check if the gh-pages branch exists, if not, create it
if git show-ref --verify --quiet refs/heads/gh-pages; then
    echo "Switching to the gh-pages branch..."
    git checkout gh-pages
else
    echo "gh-pages branch does not exist. Creating and switching to gh-pages branch..."
    git checkout --orphan gh-pages
    git rm -rf .
    git commit -m "Initial commit for gh-pages"
    git push origin gh-pages
fi

# Step 4: Render the Rmd file to HTML using Bookdown
echo "Rendering the Rmd file to HTML..."
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

# Step 5: Ensure that the _book directory is created and contains the output
if [ ! -d "_book" ] || [ ! -f "_book/index.html" ]; then
    echo "Error: Rendering failed. No HTML file found in _book directory."
    exit 1
fi

# Step 6: Add the rendered content to the root of the repository (or to /docs if that's your chosen folder)
echo "Copying the rendered HTML files to the root of the repository..."
cp -r _book/* ./

# Step 7: Add changes to Git and commit them
echo "Committing changes to Git..."
git add .
git commit -m "Deploy book to GitHub Pages"

# Step 8: Push the changes to GitHub Pages
echo "Pushing changes to GitHub Pages..."
git push origin gh-pages

# Step 9: Confirmation message
echo "Deployment complete! Your site is now live at https://jehillparikh.github.io/worldmodel/"
