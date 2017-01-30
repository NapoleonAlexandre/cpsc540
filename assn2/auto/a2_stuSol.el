(TeX-add-style-hook
 "a2_stuSol"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "etoolbox"
    "fullpage"
    "color"
    "amsmath"
    "url"
    "verbatim"
    "graphicx"
    "parskip"
    "amssymb"
    "listings")
   (TeX-add-symbols
    '("alignStar" 1)
    '("code" 1)
    '("centerfig" 2)
    '("fig" 2)
    '("argmax" 1)
    '("argmin" 1)
    "blu"
    "gre"
    "red"
    "norm"
    "R"
    "items"
    "enum"
    "argmax"
    "argmin"
    "half")
   (LaTeX-add-labels
    "eq:FD2")
   (LaTeX-add-color-definecolors
    "blu"
    "gre"
    "red"))
 :latex)

