find.top.GO.terms <-
  function(input_data, fb_GOmappingfile, wb_GOmappingfile, topNum=50, save_graph=TRUE){
  
  # root_dir = '/home/yidan/Dropbox/project_fly_worm/
  # input_data = fromJSON(file = paste(root_dir, "result_hc.json", sep = "/"), simplify = TRUE)

  # input_data: a list with 4 elements:
  # $tclust_id: a list where each element is a vector typeof character includes geneIDs in one co-cluster
  # $clust: a list where each element is a vector typeof double includes indexes of geneIDs in the gene population
  # $left_id: a character vector giving the universe/population of all fly genes  
  # $right_id: a character vector giving the universe/population of all worm genes 
  # GOmappingfile: a character giving the path of GO mapping .txt file, 
  #                where annotations are provided as a genes-to-GO mapping
  # output_file_path: a character specifying the name of a .txt file to store the output of this function:
  #              top enriched GO terms on the gene lists
  # topNum: a integer specifying the number of top GO terms to be included in the results. Defaults to 50.
  
  # readMappings(): return a named list of character vectors. The list names give the genes identifiers. 
  # Each element of the list is a character vector and contains the GO identifiers annotated to the specific gene
  
  ################################ Count and remove clusters with only one-side nodes ################################
  ################################ Count and remove clusters with size below 30       ################################
  input_data = result  

  length(input_data$tclust) #627
  fb_length <- length(input_data$left_id) #5414
  
  idx_tight_one_clusters <- c()
  idx_tight_small_clusters <- c()
  
  for (s in 1: length(input_data$tclust)) {
    if ((input_data$tclust[[s]][length(input_data$tclust[[s]])] <= fb_length-1) || (input_data$tclust[[s]][1] > fb_length-1)){
      idx_tight_one_clusters <- c(idx_tight_one_clusters, s)
    }
    if (length(input_data$tclust[[s]]) < 30){
      idx_tight_small_clusters <- c(idx_tight_small_clusters, s)
    }
  }
  
  length(idx_tight_one_clusters) #490
  length(idx_tight_small_clusters) #607
  length(setdiff(idx_tight_small_clusters, idx_tight_one_clusters)) #120
  length(setdiff(idx_tight_one_clusters, idx_tight_small_clusters)) #3
  length(intersect(idx_tight_one_clusters, idx_tight_small_clusters)) #487
  length(union(idx_tight_one_clusters, idx_tight_small_clusters)) #610
  
  # remove clusters with one-side nodes or with size below 30
  #input_data$tclust <- input_data$tclust[-union(idx_tight_one_clusters, idx_tight_small_clusters)] #17
  #input_data$tclust_id <- input_data$tclust_id[-union(idx_tight_one_clusters, idx_tight_small_clusters)] #17
  
  # remove clusters with one-side nodes
  input_data$tclust <- input_data$tclust[-idx_tight_one_clusters] #137
  input_data$tclust_id <- input_data$tclust_id[-idx_tight_one_clusters] #137

  ################################ Process the interesting genes ################################
  gene_indexes <- lapply(1:length(input_data$tclust), FUN=function(x) return(as.integer(input_data$tclust[[x]])+1))
  # all.equal(input_data$tclust, gene_indexes)
  gene_lists = input_data$tclust_id
  fb_all_genes = input_data$left_id
  wb_all_genes = input_data$right_id
  
  # separate fly genes with worm genes in a co-cluster
  #fb_length = length(fb_all_genes)
  
  associated_gene_list = lapply(1:length(gene_lists), FUN=function(x)
    return(list(fb_gene_lists = gene_lists[[x]][gene_indexes[[x]] <= fb_length], 
                wb_gene_lists = gene_lists[[x]][gene_indexes[[x]] >  fb_length])))
  
  names(associated_gene_list) = sapply(1:length(associated_gene_list), 
                                       FUN = function(x) return(paste("Co-cluster genes", x)))
  
  ################################ Select the clusters presented in the Result ################################
  # threshold on the number of genes in each species in one co-cluster
  associated_gene_length = lapply(1:length(associated_gene_list), FUN = function(i)
    return(list(fb_gene_length = length(associated_gene_list[[i]]$fb_gene_lists),
                wb_gene_length = length(associated_gene_list[[i]]$wb_gene_lists))))
  
  idx_selected_co_clusters <- c()
  for (s in 1: length(associated_gene_length)){
    if ((associated_gene_length[[s]]$fb_gene_length >=10) & (associated_gene_length[[s]]$wb_gene_length >=10 )){
      idx_selected_co_clusters <- c(idx_selected_co_clusters, s)
    }
  }
  
  ################################ Process the gene universe and GO annotations ################################
  #root_dir = '/home/yidan/Dropbox/project_fly_worm/
  #root_dir = "/Users/yidansun/Downloads"
  root_dir = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm'"
  
  path_convert_fb <- paste(root_dir, "GO_enrichment_analysis/convert_fb.py", sep="/")
  fb_GOmappingfile = paste(root_dir, "GO_enrichment_analysis/GO_annotations_2019/gene_association.fb", sep = "/")
  path_gene2go_fb <- paste(root_dir, "GO_enrichment_analysis/gene2go_2019/fb_geneid2go.map", sep="/")
  command <- paste("python", path_convert_fb, fb_GOmappingfile, path_gene2go_fb)
  response <- system(command)
  
  path_convert_wb <- paste(root_dir, "GO_enrichment_analysis/convert_wb.py", sep="/")
  wb_GOmappingfile = paste(root_dir, "GO_enrichment_analysis/GO_annotations_2019/gene_association.wb", sep = "/")
  path_gene2go_wb <- paste(root_dir, "GO_enrichment_analysis/gene2go_2019/wb_geneid2go.map", sep="/")
  command <- paste("python", path_convert_wb, wb_GOmappingfile, path_gene2go_wb)
  response <- system(command)
  
  path_gene2go_fb = gsub("'", "", path_gene2go_fb)
  path_gene2go_wb = gsub("'", "", path_gene2go_wb)
  
  fb_sp_geneID2GO <- readMappings(file = path_gene2go_fb)
  fb_all_genes_w_GO <- names(fb_sp_geneID2GO) #14462
  
  fb_sp_GO2geneID <- inverseList(fb_sp_geneID2GO)
  fb_sp_GO <- names(fb_sp_GO2geneID) #8574
  
  wb_sp_geneID2GO <- readMappings(file = path_gene2go_wb)
  wb_all_genes_w_GO <- names(wb_sp_geneID2GO) #14269 #the first geneID is ""
  
  wb_sp_GO2geneID <- inverseList(wb_sp_geneID2GO)
  wb_sp_GO <- names(wb_sp_GO2geneID) #6704
  
  #GO Term sets of 2 species
  #length(union(fb_sp_GO, wb_sp_GO)) #9921
  #length(intersect(fb_sp_GO, wb_sp_GO)) #5357
  
  ################################ filter GO annotations to keep BP GO terms for fly ################################
  #GOTERM #GOBPTerm #GOMFTerm #GOCCTerm
  #this data is for all species
  GOTERM_aslist <- as.list(GOTERM) #45050
  GO_in_GOTERM <- sapply(1:length(GOTERM_aslist), FUN=function(x) GOTERM_aslist[[x]]@GOID)
  
  fb_sp_GO_old <- fb_sp_GO #new 8574   #old 8371
  fb_sp_GO <- intersect(fb_sp_GO, GO_in_GOTERM) #new 8569   #old 8370
 
  # GOTERM_aslist[[1]]@Ontology # ls(GOBPTerm)
  GO_ontology <- Ontology(GOTERM[fb_sp_GO]) #8569
  fb_sp_GO <- fb_sp_GO[which(GO_ontology=="BP")] #new 5101 #old 4978
  
  fb_sp_GO2geneID <- fb_sp_GO2geneID[fb_sp_GO] #5101
  fb_sp_geneID2GO <- inverseList(fb_sp_GO2geneID) #12830
  
  ################################ filter GO annotations to keep BP GO terms for worm ################################
  #GOTERM #GOBPTerm #GOMFTerm #GOCCTerm
  # GOTERM_aslist <- as.list(GOTERM)
  # GO_in_GOTERM <- sapply(1:length(GOTERM_aslist), FUN=function(x) GOTERM_aslist[[x]]@GOID)
  wb_sp_GO_old <- wb_sp_GO #6704
  wb_sp_GO <- intersect(wb_sp_GO, GO_in_GOTERM) #6702
  
  # GOTERM_aslist[[1]]@Ontology
  GO_ontology <- Ontology(GOTERM[wb_sp_GO]) #6702
  wb_sp_GO <- wb_sp_GO[which(GO_ontology=="BP")] #3675
  
  wb_sp_GO2geneID <- wb_sp_GO2geneID[wb_sp_GO] #3675
  wb_sp_geneID2GO <- inverseList(wb_sp_GO2geneID) #9450
  
  # BP GO Term sets of 2 species
  #length(union(fb_sp_GO, wb_sp_GO)) #5981
  #length(intersect(fb_sp_GO, wb_sp_GO)) #2795
  
  # update the gene universe (background set) after we modify GO2geneID 
  fb_all_genes_w_GO <- names(fb_sp_geneID2GO) #12830
  wb_all_genes_w_GO <- names(wb_sp_geneID2GO) #9450
  
  # Q: check if the above update is necessary
  # Q: do we have to subset geneID2GO with gene universe, that is, all.equal(names(fb_sp_geneID2GO), fb_all_genes_w_GO)
  
  ################################ Create a file folder and subfolders to store the result of each co-cluster ####################
  dir.create(paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", sep = "/"))
   
  lapply(1:length(associated_gene_list), FUN = function(x) {
    graph_subfoldername = names(associated_gene_list)[x]
    dir.create(paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", graph_subfoldername, sep = "/"))
  })

  ################################ GO term enrichment analysis ################################
  # sp_topGo: returned value from write_top_GO_terms()
  # make sure to clean all of enviroment variables before run this function 
  topNum=50
  save_graph = FALSE
  write_top_GO_terms(associated_gene_list, fb_all_genes, wb_all_genes, 
                     fb_all_genes_w_GO, wb_all_genes_w_GO,
                     fb_geneID2GO=fb_sp_geneID2GO, wb_geneID2GO=wb_sp_geneID2GO, GO_dict=GOTERM_aslist,
                     topNum, save_graph)
  
  ################################ threshold the size of co-clusters which will be put in the table ################################
  ### .txt file one
  condition_table <- lapply (associated_gene_list, FUN=function(x){
  
    ((length(x$fb_gene_lists) > 15) & (length(x$wb_gene_lists) > 15))
  })
 
  associated_gene_list_table = associated_gene_list[unlist(condition_table)]
  # length(associated_gene_list[unlist(condition)])
  
  description_filepath = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/result/description.txt", sep = "/")
  write(paste("co-clusters whose both size of fly and size of worm are larger than 15"), file = description_filepath, append = TRUE)
  write(names(associated_gene_list_table), file = description_filepath, append = TRUE)
  write("\n", file = description_filepath, append = TRUE)
  
  
  ### .txt file two
  condition_result <- lapply (associated_gene_list, FUN=function(x){
    
    (length(x$fb_gene_lists) + length(x$wb_gene_lists) > 15)
  })
  
  associated_gene_list_result = associated_gene_list[unlist(condition_result)]
  # length(associated_gene_list[unlist(condition_result)])  
  
  description_filepath = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/result/description.txt", sep = "/")
  write(paste("co-clusters whose size is larger than 15"), file = description_filepath, append = TRUE)
  write(names(associated_gene_list_result), file = description_filepath, append = TRUE)
  write("\n", file = description_filepath, append = TRUE) 

  #return(sp_topGo)
}

################# List all genes which do not have BP GO terms for co-clusters
fly_gene_list_wn_go_all <- lapply(1:30, FUN = function(x)
  return(associated_gene_list[[x]]$fb_gene_lists[!associated_gene_list[[x]]$fb_gene_lists %in% fb_all_genes_w_GO]))

worm_gene_list_wn_go_all <- lapply(1:30, FUN = function(x)
  return(associated_gene_list[[x]]$wb_gene_lists[!associated_gene_list[[x]]$wb_gene_lists %in% wb_all_genes_w_GO]))

################# List all genes which do not have BP GO terms for the 16 selected co-clusters
fly_gene_list_wn_go <- lapply(idx_selected_co_clusters, FUN = function(x)
  return(associated_gene_list[[x]]$fb_gene_lists[!associated_gene_list[[x]]$fb_gene_lists %in% fb_all_genes_w_GO]))

worm_gene_list_wn_go <- lapply(idx_selected_co_clusters, FUN = function(x)
  return(associated_gene_list[[x]]$wb_gene_lists[!associated_gene_list[[x]]$wb_gene_lists %in% wb_all_genes_w_GO]))

#question: how to save as a data.frame
file_path = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm/GO_enrichment_analysis/genes_without_BPGOTerms.xls'"
WriteXLS(c("fly_gene_list_wn_go_all","worm_gene_list_wn_go_all"), ExcelFileName = file_path)

