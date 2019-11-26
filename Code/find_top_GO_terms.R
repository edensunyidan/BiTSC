find_top_GO_terms <-
  function(associated_gene_list, fb_all_genes, wb_all_genes, fb_all_genes_w_GO, wb_all_genes_w_GO,
           fb_geneID2GO, wb_geneID2GO, GO_dict, topNum, save_graph, graph_subfoldername){
    
    fb_myInterestingGenes = intersect(associated_gene_list$fb_gene_lists, fb_all_genes_w_GO)
    wb_myInterestingGenes = intersect(associated_gene_list$wb_gene_lists, wb_all_genes_w_GO)

    root_dir = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm'"
    ### consider myInterestingGenes == 0
    
    if ((length(fb_myInterestingGenes) == 0) & (length(wb_myInterestingGenes) == 0)){
      
      description_filepath = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result/description.txt", sep = "/")
      # write("", file = description_filepath)
      write(paste(graph_subfoldername, "fly and worm genes no GO annotation!"), file = description_filepath, append = TRUE)
  
      write(associated_gene_list$fb_gene_lists, file = description_filepath, append = TRUE)
      write(associated_gene_list$wb_gene_lists, file = description_filepath, append = TRUE)
      write("\n", file = description_filepath, append = TRUE)
      
      # find a better way to suppress the return message
      return()
    }

    if ((length(fb_myInterestingGenes) != 0) & (length(wb_myInterestingGenes) != 0)){
    fb_graph_filename = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "fly", sep = "/")
    wb_graph_filename = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "worm", sep = "/")
   
    fb_geneNames = intersect(fb_all_genes, fb_all_genes_w_GO)
    wb_geneNames = intersect(wb_all_genes, wb_all_genes_w_GO)
    
    fb_result = vector("list", length = 4)
    wb_result = vector("list", length = 4)
    names(fb_result) = c("test_result", "InterestGenes", "InterestTerms", "UnionTerms")
    names(wb_result) = c("test_result", "InterestGenes", "InterestTerms", "UnionTerms")
    # $test_result: a data frame containing the top 'topNum' GO terms identified by the Fisher algorithm,
    #               it also include some statistics on the GO terms and the p-values
    # $InterestGenes: a vector including interested genes
    # $InterestTerms: a list of interested genes to GO terms
    # $UnionTerms: a vector contain the union of all GO terms of interested genes
    
    fb_result$test_result = enrichment_test(fb_myInterestingGenes, fb_geneNames, fb_geneID2GO, topNum, save_graph, fb_graph_filename)
    wb_result$test_result = enrichment_test(wb_myInterestingGenes, wb_geneNames, wb_geneID2GO, topNum, save_graph, wb_graph_filename)
    
    fb_result$test_result$Term = sapply((1:length(fb_result$test_result$GO.ID)), FUN = function(y) {
      return(GO_dict[[fb_result$test_result$GO.ID[y]]]@Term)
    })
    
    wb_result$test_result$Term = sapply((1:length(wb_result$test_result$GO.ID)), FUN = function(y) {
      return(GO_dict[[wb_result$test_result$GO.ID[y]]]@Term)
    })
    
    fb_result$InterestGenes = fb_myInterestingGenes
    wb_result$InterestGenes = wb_myInterestingGenes
    
    ### we need consider the size of the co-cluster and size of genes of each species.
    if (length(fb_myInterestingGenes) < 15) {
      
      fb_myInterestingGenes2GOs = fb_geneID2GO[fb_myInterestingGenes]
      
      fb_result$InterestTerms = lapply(1:length(fb_myInterestingGenes2GOs), FUN = function(x){
        
        return(sapply((1:length(fb_myInterestingGenes2GOs[[x]])), FUN = function(y) {
          return(GO_dict[[fb_myInterestingGenes2GOs[[x]][y]]]@Term)
        }))
      })
      
      fb_result$UnionTerms = unique(unlist(fb_result$InterestTerms))
      
      names(fb_result$InterestTerms) = fb_myInterestingGenes
    } 
    
    if (length(wb_myInterestingGenes) < 15) {
  
      wb_myInterestingGenes2GOs = wb_geneID2GO[wb_myInterestingGenes]
      
      wb_result$InterestTerms = lapply(1:length(wb_myInterestingGenes2GOs), FUN = function(x){
        
        return(sapply((1:length(wb_myInterestingGenes2GOs[[x]])), FUN = function(y) {
          return(GO_dict[[wb_myInterestingGenes2GOs[[x]][y]]]@Term)
        }))
      })
      
      wb_result$UnionTerms = unique(unlist(wb_result$InterestTerms))
      
      names(wb_result$InterestTerms) = wb_myInterestingGenes
    }
    
    fb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "fb_output.xls", sep = "/")
    wb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "wb_output.xls", sep = "/")
    
    # fb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/result", graph_subfoldername, "fb_output.txt", sep = "/")
    # wb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/result", graph_subfoldername, "wb_output.txt", sep = "/")
    
    # write("", file = fb_output_file_path)
    # write(names(associated_gene_list)[i], file=output_file_path, append = TRUE)
    # write.table(fb_result$InterestGenes, file = fb_output_file_path, append = TRUE, quote = FALSE, sep = "\t")
    # write("\n", file = output_file_path, append = TRUE)
    
    # write("!fb_result$InterestGenes", file = fb_output_file_path, append = TRUE, sep = "")
    # write.table(fb_result$InterestGenes, file = fb_output_file_path, append = TRUE, sep = "")
    # write("!fb_result$test_result", file = fb_output_file_path, append = TRUE, sep = "")
    # write.table(fb_result$test_result, file = fb_output_file_path, append = TRUE, quote = FALSE, col.names = TRUE, sep = "\t")
    # 
    # write.csv(fb_result$test_result, file = fb_output_file_path, append = TRUE)
    # write.csv(fb_result$InterestGenes, file = fb_output_file_path, append = TRUE)
    
    # write("!fb_result$InterestGenes", file = fb_output_file_path, SheetNames="InterestGenes")
    # convert the list/vector to data.frame
    fly_test_result = fb_result$test_result
    fly_InterestGenes = as.data.frame(list(IntererstGenes_name = fb_result$InterestGenes))
    fly_UnionTerms = as.data.frame(list(UnionTerms_name = fb_result$UnionTerms))
    InterestTerms_list = lapply(fb_result$InterestTerms, FUN = paste, collapse = "\n")
    fly_InterestTerms = as.data.frame(matrix(InterestTerms_list, byrow = TRUE))
    if (dim(fly_InterestTerms)[1] > 0){
      fly_InterestTerms$GeneID = fb_result$InterestGenes
    }
    # consider the case that InterestTerms and UnionTerms could be empty
    # print(dim(InterestTerms))
    # colnames(InterestTerms) <- c("GeneGO","GeneID")
    WriteXLS(c("fly_test_result", "fly_InterestGenes", "fly_UnionTerms", "fly_InterestTerms"), ExcelFileName = fb_output_file_path)
    
    worm_test_result = wb_result$test_result
    worm_InterestGenes = as.data.frame(list(IntererstGenes_name = wb_result$InterestGenes))
    worm_UnionTerms = as.data.frame(list(UnionTerms_name = wb_result$UnionTerms))
    InterestTerms_list = lapply(wb_result$InterestTerms, FUN = paste, collapse = "\n")
    worm_InterestTerms = as.data.frame(matrix(InterestTerms_list, byrow = TRUE))
    if (dim(worm_InterestTerms)[1] > 0){
      worm_InterestTerms$GeneID = wb_result$InterestGenes
    }
    # consider the case that InterestTerms and UnionTerms could be empty
    # print(dim(InterestTerms))
    # colnames(InterestTerms) <- c("GeneGO","GeneID")
    WriteXLS(c("worm_test_result", "worm_InterestGenes", "worm_UnionTerms", "worm_InterestTerms"), ExcelFileName = wb_output_file_path)
  
    # result = list(fb = fb_result, wb = wb_result)
    # return(result)
    } else if (length(fb_myInterestingGenes) == 0) {
      
      description_filepath = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result/description.txt", sep = "/")
      # write("", file = fb_output_file_path)
      write(paste(graph_subfoldername, "fly gene(s) no GO annotation!"), file = description_filepath, append = TRUE)
      write(associated_gene_list$fb_gene_lists, file = description_filepath, append = TRUE)
      write("\n", file = description_filepath, append = TRUE)
      
      wb_graph_filename = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "worm", sep = "/")
      wb_geneNames = intersect(wb_all_genes, wb_all_genes_w_GO)
      wb_result = vector("list", length = 4)
      names(wb_result) = c("test_result", "InterestGenes", "InterestTerms", "UnionTerms")
      wb_result$test_result = enrichment_test(wb_myInterestingGenes, wb_geneNames, wb_geneID2GO, topNum, save_graph, wb_graph_filename)
      
      wb_result$test_result$Term = sapply((1:length(wb_result$test_result$GO.ID)), FUN = function(y) {
        return(GO_dict[[wb_result$test_result$GO.ID[y]]]@Term)
      })
      wb_result$InterestGenes = wb_myInterestingGenes
      
      if (length(wb_myInterestingGenes) < 15) {
        
        wb_myInterestingGenes2GOs = wb_geneID2GO[wb_myInterestingGenes]
        
        wb_result$InterestTerms = lapply(1:length(wb_myInterestingGenes2GOs), FUN = function(x){
          
          return(sapply((1:length(wb_myInterestingGenes2GOs[[x]])), FUN = function(y) {
            return(GO_dict[[wb_myInterestingGenes2GOs[[x]][y]]]@Term)
          }))
        })
        
        wb_result$UnionTerms = unique(unlist(wb_result$InterestTerms))
        
        names(wb_result$InterestTerms) = wb_myInterestingGenes
      }
      
      wb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "wb_output.xls", sep = "/")
      
      worm_test_result = wb_result$test_result
      worm_InterestGenes = as.data.frame(list(IntererstGenes_name = wb_result$InterestGenes))
      worm_UnionTerms = as.data.frame(list(UnionTerms_name = wb_result$UnionTerms))
      InterestTerms_list = lapply(wb_result$InterestTerms, FUN = paste, collapse = "\n")
      worm_InterestTerms = as.data.frame(matrix(InterestTerms_list, byrow = TRUE))
      if (dim(worm_InterestTerms)[1] > 0){
        worm_InterestTerms$GeneID = wb_result$InterestGenes
      }
      # consider the case that InterestTerms and UnionTerms could be empty
      # print(dim(InterestTerms))
      # colnames(InterestTerms) <- c("GeneGO","GeneID")
      WriteXLS(c("worm_test_result", "worm_InterestGenes", "worm_UnionTerms", "worm_InterestTerms"), ExcelFileName = wb_output_file_path)
      
    } else {
      # length(wb_myInterestingGenes) == 0
      
      description_filepath = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result/description.txt", sep = "/")
      # write("", file = fb_output_file_path)
      write(paste(graph_subfoldername, "worm gene(s) no GO annotation!"), file = description_filepath, append = TRUE)
      write(associated_gene_list$wb_gene_lists, file = description_filepath, append = TRUE)
      write("\n", file = description_filepath, append = TRUE)
      
      fb_graph_filename = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "fly", sep = "/")
      
      fb_geneNames = intersect(fb_all_genes, fb_all_genes_w_GO)
      
      fb_result = vector("list", length = 4)
  
      names(fb_result) = c("test_result", "InterestGenes", "InterestTerms", "UnionTerms")

      fb_result$test_result = enrichment_test(fb_myInterestingGenes, fb_geneNames, fb_geneID2GO, topNum, save_graph, fb_graph_filename)

      fb_result$test_result$Term = sapply((1:length(fb_result$test_result$GO.ID)), FUN = function(y) {
        return(GO_dict[[fb_result$test_result$GO.ID[y]]]@Term)
      })
      
      fb_result$InterestGenes = fb_myInterestingGenes
      
      ### we need consider the size of the co-cluster and size of genes of each species.
      if (length(fb_myInterestingGenes) < 15) {
        
        fb_myInterestingGenes2GOs = fb_geneID2GO[fb_myInterestingGenes]
        
        fb_result$InterestTerms = lapply(1:length(fb_myInterestingGenes2GOs), FUN = function(x){
          
          return(sapply((1:length(fb_myInterestingGenes2GOs[[x]])), FUN = function(y) {
            return(GO_dict[[fb_myInterestingGenes2GOs[[x]][y]]]@Term)
          }))
        })
        
        fb_result$UnionTerms = unique(unlist(fb_result$InterestTerms))
        
        names(fb_result$InterestTerms) = fb_myInterestingGenes
      } 
      
      fb_output_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, "fb_output.xls", sep = "/")
      
      fly_test_result = fb_result$test_result
      fly_InterestGenes = as.data.frame(list(IntererstGenes_name = fb_result$InterestGenes))
      fly_UnionTerms = as.data.frame(list(UnionTerms_name = fb_result$UnionTerms))
      InterestTerms_list = lapply(fb_result$InterestTerms, FUN = paste, collapse = "\n")
      fly_InterestTerms = as.data.frame(matrix(InterestTerms_list, byrow = TRUE))
      if (dim(fly_InterestTerms)[1] > 0){
        fly_InterestTerms$GeneID = fb_result$InterestGenes
      }
      # consider the case that InterestTerms and UnionTerms could be empty
      # print(dim(InterestTerms))
      # colnames(InterestTerms) <- c("GeneGO","GeneID")
      WriteXLS(c("fly_test_result", "fly_InterestGenes", "fly_UnionTerms", "fly_InterestTerms"), ExcelFileName = fb_output_file_path)
    }
 }
