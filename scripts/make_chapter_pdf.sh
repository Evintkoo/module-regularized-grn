#!/bin/bash
# Render paper/chapter_phase8.md to paper/chapter_phase8.pdf
# Requires: pandoc, pdflatex (brew install pandoc mactex)
set -e
pandoc paper/chapter_phase8.md \
    --output paper/chapter_phase8.pdf \
    --pdf-engine=pdflatex \
    --variable geometry:margin=1in \
    --variable fontsize=12pt \
    --variable linestretch=1.5 \
    --variable documentclass=report
echo "PDF created: paper/chapter_phase8.pdf"
