library(bio3d, quietly=TRUE)
library(reticulate)
library(stringr)

np <- import("numpy")
args <- commandArgs(trailingOnly=TRUE)
cat(args, sep = "\n")

quiet <- function(x) { 
  sink(tempfile()) 
  on.exit(sink()) 
  invisible(force(x)) 
} 

if(str_equal(args[3], "all")){
  nmodes <- NULL
} else{
  nmodes <- as.integer(args[3]) 
}

pdb <- read.pdb(args[1])
modes <- suppressMessages(suppressWarnings(quiet(nma(pdb))))
cm <- suppressMessages(suppressWarnings(quiet(dccm(modes, nmodes=nmodes))))
b <- np$array(cm)
np$save(args[2], b)