# Build the preprint.  Figures are read from ../figures/, so the PDF always
# reflects whatever the reporting job last wrote -- there is no second copy of
# a figure to fall out of date.
#
#   make          build the PDF
#   make clean    remove LaTeX intermediates (keeps the PDF)

PAPER = main
OUT   = storm_outage_risk_michigan_preprint.pdf

.PHONY: all clean
all: $(OUT)

$(OUT): $(PAPER).tex refs.bib
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex   $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	cp $(PAPER).pdf $(OUT)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
