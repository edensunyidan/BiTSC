write_top_GO_terms <-
  function(associated_gene_list, fb_all_genes, wb_all_genes, fb_all_genes_w_GO, wb_all_genes_w_GO,
           fb_geneID2GO=fb_sp_geneID2GO, wb_geneID2GO=wb_sp_geneID2GO, GO_dict, topNum, save_graph){
    
    # write("", file = output_file_path)
    # top_GO_all: returned value from sapply()
    sapply(1:length(associated_gene_list), FUN = function(i){
      find_top_GO_terms(associated_gene_list[[i]], fb_all_genes, wb_all_genes, fb_all_genes_w_GO, wb_all_genes_w_GO,
                        fb_geneID2GO, wb_geneID2GO, GO_dict, topNum, save_graph, graph_subfoldername=names(associated_gene_list)[i])
      # temp: returned result from find_top_GO_terms
      # write(paste("Sample:", names(associated_gene_list)[i]), file=output_file, append=TRUE)
      # write(names(associated_gene_list)[i], file=output_file_path, append = TRUE)
      # write.table(temp, file = output_file_path, append = TRUE, quote = FALSE, sep = "\t")
      # write("\n", file = output_file_path, append = TRUE)
      # return(temp)
      })
    # return(top_GO_all)
  }


### we should also store the information of asssociated_gene_list
# associated_gene_list1 = associated_gene_list[[333]]