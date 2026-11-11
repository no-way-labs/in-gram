# Paper source

`in-gram.pdf` is built from `main.tex`. To rebuild it:

```bash
latexmk -pdf -outdir=build main.tex
```

- `sections/` holds the text and `figures/` the TikZ and pgfplots sources of the figures.
- `generated/` holds the bodies of the tables (`.tex`), the data behind the figures (`.dat`) and the source of
  every number in the text (`numbers.txt`, `serving_numbers.txt`). The scripts in `../research/scripts` write
  these files from the run logs in `../research/runs`; see `../research/README.md` to regenerate them.
- `references.bib` is the bibliography.
