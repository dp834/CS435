

MATLAB_FILES = Homework4.m
MATLAB_MAIN  = Homework4.m

IMG_OUTPUT = images/generated/

LATEX_SRC  = Homework-4.tex
PDF_OUTPUT = Homework-4

run: $(MATLAB_FILES) $(MATLAB_MAIN)
	mkdir -p $(IMG_OUTPUT)
	matlab -nodisplay -nosplash -batch "run('$(MATLAB_MAIN)')"

pdf: $(LATEX_SRC)
	pdflatex -jobname $(PDF_OUTPUT) $(LATEX_SRC)

clean:
	@rm -rf $(IMG_OUTPUT)/* $(PDF_OUTPUT).pdf *.aux *.log &2> /dev/null || true
