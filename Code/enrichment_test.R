enrichment_test <-
  function(myInterestingGenes, geneNames, geneID2GO, topNum, save_graph, graph_filename){
    # "myInterestingGenes" is the given set of genes
    # "geneNames" is the population of genes
    # "geneID2GO" is the mapping of gene IDs to GO terms
    # geneList <- factor(as.integer(geneNames %in% myInterestingGenes))
    
    geneList <- factor(as.integer(geneNames %in% myInterestingGenes), levels=c(0,1))
    names(geneList) <- geneNames
    GOdata <- new("topGOdata", ontology = "BP", allGenes = geneList, nodeSize = 1, annot = annFUN.gene2GO, gene2GO = geneID2GO)
    # GOdata
    resultFisher <- runTest(GOdata, algorithm = "classic", statistic = "fisher")
    # resultFisher
    # sort(score(resultFisher), decreasing = FALSE)
    allRes <- GenTable(GOdata, classicFisher = resultFisher, 
                       orderBy = "classicFisher", ranksOf = "classicFisher", topNodes = topNum)
    
    if (save_graph == TRUE){
      printGraph(GOdata, resultFisher, firstSigNodes = 5, fn.prefix = graph_filename, useInfo = "all", pdfSW = TRUE)
    }
    return(allRes)
  }




