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

build/finish.pdf: build/plot.pdf aufbau.tex auswertung.tex diskussion.tex durchfuehrung.tex finish.tex header.tex theorie.tex title.tex ergebnisse.tex| build
	TEXINPUTS="$(call translate,build:)" \
	BIBINPUTS="$(call translate,$(pwd):)" \
	latexmk \
	 --lualatex \
	 --output-directory=build \
	 --interaction=nonstopmode \
	--halt-on-error \
	finish.tex

build/plot.pdf: plot.py data.txt | build
	TEXINPUTS="$(call translate,$(pwd):)" python plot.py

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
