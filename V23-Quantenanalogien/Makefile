ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

all: build/finish.pdf

build/finish.pdf: build/plot.pdf content/aufbau.tex content/auswertung.tex content/diskussion.tex content/durchfuehrung.tex finish.tex header.tex content/theorie.tex title.tex content/ergebnisse.tex| build
	TEXINPUTS="$(call translate,build:)" \
	BIBINPUTS="$(call translate,$(pwd):)" \
	latexmk \
	 --lualatex \
	 --output-directory=build \
	 --interaction=nonstopmode \
	--halt-on-error \
	finish.tex

build/plot.pdf: scripts/plot.py scripts/data.txt | build
	TEXINPUTS="$(call translate,$(pwd):)" python scripts/plot.py

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
