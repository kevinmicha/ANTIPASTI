library(bio3d, quietly=TRUE)
library(reticulate)
library(stringr)

np <- import("numpy")
args <- commandArgs(trailingOnly=TRUE)
cat(args, sep = "\n")

pdb <- read.pdb(args[1]) 

getChainLenghts <- function(pdb) {
    pos_heavy <- 1
    pos_light <- sum(lengths(pdb$atom$chain))
    hchain <- pdb$atom$chain[pos_heavy]
    lchain <- pdb$atom$chain[pos_light]

    ca.h <- atom.select(pdb, "calpha", chain=hchain)
    hnum <- sum(lengths(ca.h$atom))

    if (hchain != lchain) {
    ca.l <- atom.select(pdb, "calpha", chain=lchain)
    lnum <- sum(lengths(ca.l$atom))
    } else {
        lnum <- 0
    }
    return(list(hnum, lnum))

}

list_chains <- do.call(getChainLenghts, list(pdb))
np$save("/Users/kevinmicha/Downloads/value.npy", list_chains)
