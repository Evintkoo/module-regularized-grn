#!/usr/bin/env python3
"""Simple script to convert markdown to PDF using markdown2 and reportlab"""

import sys
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import markdown
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "reportlab", "markdown"])
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import markdown

# Read markdown
with open('paper/manuscript.md', 'r') as f:
    md_text = f.read()

# Create PDF
pdf_file = 'paper/manuscript.pdf'
doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)

# Container for the 'Flowable' objects
story = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor='#000000',
    spaceAfter=30,
    alignment=1,  # Center
)

# Simple conversion (not perfect for math, but works)
lines = md_text.split('\n')
for line in lines:
    line = line.strip()
    if not line:
        story.append(Spacer(1, 0.2*inch))
    elif line.startswith('# '):
        text = line[2:].replace('**', '<b>').replace('**', '</b>')
        story.append(Paragraph(text, title_style))
        story.append(Spacer(1, 0.3*inch))
    elif line.startswith('## '):
        text = line[3:].replace('**', '<b>').replace('**', '</b>')
        story.append(Paragraph(text, styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
    elif line.startswith('### '):
        text = line[4:].replace('**', '<b>').replace('**', '</b>')
        story.append(Paragraph(text, styles['Heading3']))
        story.append(Spacer(1, 0.1*inch))
    elif line.startswith('**') and line.endswith('**'):
        text = '<b>' + line[2:-2] + '</b>'
        story.append(Paragraph(text, styles['Normal']))
    elif line.startswith('- ') or line.startswith('* '):
        text = '• ' + line[2:].replace('**', '<b>').replace('**', '</b>')
        story.append(Paragraph(text, styles['Normal']))
    elif line.startswith('|'):
        # Skip table lines for now
        pass
    elif line.startswith('$$') or line.startswith('$'):
        # Skip math for now
        pass
    elif line == '---':
        story.append(Spacer(1, 0.3*inch))
    else:
        text = line.replace('**', '<b>').replace('**', '</b>')
        text = text.replace('*', '<i>').replace('*', '</i>')
        if text:
            story.append(Paragraph(text, styles['Normal']))

# Build PDF
doc.build(story)
print(f"PDF created successfully: {pdf_file}")
print(f"Note: Math equations and tables are simplified. For full formatting, use LaTeX.")
