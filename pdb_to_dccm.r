library(bio3d, quietly=TRUE)
library(reticulate)
np <- import("numpy")
args <- commandArgs(trailingOnly=TRUE)
cat(args, sep = "\n")

quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 

pdb <- read.pdb(args[1])
modes <- suppressMessages(suppressWarnings(quiet(nma(pdb))))
cm <- suppressMessages(suppressWarnings(quiet(dccm(modes))))
b <- np$array(cm)
np$save(args[2], b)