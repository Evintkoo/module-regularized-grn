#!/bin/bash
# Convert paper/manuscript.md to PDF using pandoc.
# Requires:
#   - pandoc:   brew install pandoc
#   - TeX Live: brew install --cask mactex  (or brew install basictex)
# Uses pdflatex (default, widely available). For Unicode/custom font support use --pdf-engine=xelatex.
set -e
pandoc paper/manuscript.md \
    --output paper/manuscript.pdf \
    --pdf-engine=pdflatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt
echo "PDF created: paper/manuscript.pdf"
