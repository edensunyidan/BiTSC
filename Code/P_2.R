hypergeometric_test <- 
  function(result, fb_GOmappingfile, wb_GOmappingfile, output_file_p_values){
    
  # root_dir = '/home/yidan/Dropbox/project_fly_worm/
  # input_data = fromJSON(file = paste(root_dir, "result_hc.json", sep = "/"), simplify = TRUE)
  
  # input_data: a list with 4 elements:
  # $tclust_id: a list where each element is a vector typeof character includes geneIDs in one co-cluster
  # $clust: a list where each element is a vector typeof double includes indexes of geneIDs in the gene population
  # $left_id: a character verctor giving the universe/population of all fly genes  
  # $right_id: a character verctor giving the universe/population of all worm genes 
  # GOmappingfile: a character giving the path of GO mapping .txt file, 
  #                where annotations are provided as a genes-to-GO mapping
  # output_file: a character specifying the name of a .txt file to store the output of this function:
  #              top enriched GO terms on the gene lists
    
  
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
  input_data$tclust <- input_data$tclust[-union(idx_tight_one_clusters, idx_tight_small_clusters)] #17
  input_data$tclust_id <- input_data$tclust_id[-union(idx_tight_one_clusters, idx_tight_small_clusters)] #17
  
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
  fb_length = length(fb_all_genes)
  
  associated_gene_list = lapply(1:length(gene_lists), FUN=function(x)
    return(list(fb_gene_lists = gene_lists[[x]][gene_indexes[[x]] <= fb_length], 
                wb_gene_lists = gene_lists[[x]][gene_indexes[[x]] >  fb_length])))
  
  names(associated_gene_list) = sapply(1:length(associated_gene_list), 
                                       FUN = function(x) return(paste("Co-cluster genes", x)))
  
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
  fb_all_genes_w_GO <- names(fb_sp_geneID2GO)
  
  fb_sp_GO2geneID <- inverseList(fb_sp_geneID2GO)
  fb_sp_GO <- names(fb_sp_GO2geneID)
  
  wb_sp_geneID2GO <- readMappings(file = path_gene2go_wb)
  wb_all_genes_w_GO <- names(wb_sp_geneID2GO)
  
  wb_sp_GO2geneID <- inverseList(wb_sp_geneID2GO)
  wb_sp_GO <- names(wb_sp_GO2geneID)
  
  ################################ filter GO annotations to keep BP GO terms for fly ################################
  #GOTERM #GOBPTerm #GOMFTerm #GOCCTerm
  GOTERM_aslist <- as.list(GOTERM)
  GO_in_GOTERM <- sapply(1:length(GOTERM_aslist), FUN=function(x) GOTERM_aslist[[x]]@GOID)
  
  fb_sp_GO_old <- fb_sp_GO
  fb_sp_GO <- intersect(fb_sp_GO, GO_in_GOTERM)
  
  # GOTERM_aslist[[1]]@Ontology # ls(GOBPTerm)
  GO_ontology <- Ontology(GOTERM[fb_sp_GO])
  fb_sp_GO <- fb_sp_GO[which(GO_ontology=="BP")]
  
  fb_sp_GO2geneID <- fb_sp_GO2geneID[fb_sp_GO]
  fb_sp_geneID2GO <- inverseList(fb_sp_GO2geneID)
  
  ################################ filter GO annotations to keep BP GO terms for worm ################################
  #GOTERM #GOBPTerm #GOMFTerm #GOCCTerm
  # GOTERM_aslist <- as.list(GOTERM)
  # GO_in_GOTERM <- sapply(1:length(GOTERM_aslist), FUN=function(x) GOTERM_aslist[[x]]@GOID)
  wb_sp_GO_old <- wb_sp_GO
  wb_sp_GO <- intersect(wb_sp_GO, GO_in_GOTERM)
  
  # GOTERM_aslist[[1]]@Ontology
  GO_ontology <- Ontology(GOTERM[wb_sp_GO])
  wb_sp_GO <- wb_sp_GO[which(GO_ontology=="BP")]
  
  wb_sp_GO2geneID <- wb_sp_GO2geneID[wb_sp_GO]
  wb_sp_geneID2GO <- inverseList(wb_sp_GO2geneID)  
  
  # update the gene universe (background set) after we modify GO2geneID 
  fb_all_genes_w_GO <- names(fb_sp_geneID2GO)
  wb_all_genes_w_GO <- names(wb_sp_geneID2GO)
  
  # Q: check if the above update is necessary
  # Q: do we have to subset geneID2GO with gene universe, that is, all.equal(names(fb_sp_geneID2GO), fb_all_genes_w_GO)
  
  ################################ Combine the GO terms from both species together ################################
  
  fb_geneNames = intersect(fb_all_genes, fb_all_genes_w_GO)
  wb_geneNames = intersect(wb_all_genes, wb_all_genes_w_GO)
  fb_geneNames_GO = unlist(fb_sp_geneID2GO[fb_geneNames], use.names = FALSE)
  wb_geneNames_GO = unlist(wb_sp_geneID2GO[wb_geneNames], use.names = FALSE)
  # population/universe GO terms from both species
  geneNames_GO = union(fb_geneNames_GO, wb_geneNames_GO)
  
  myInterestingGenes_GO <- lapply(1:length(associated_gene_list), FUN = function(i) {
    
    fb_myInterestingGenes = intersect(associated_gene_list[[i]]$fb_gene_lists, fb_all_genes_w_GO)
    wb_myInterestingGenes = intersect(associated_gene_list[[i]]$wb_gene_lists, wb_all_genes_w_GO)
    
    return(list(fb_myInterestingGenes_GO=unlist(fb_sp_geneID2GO[fb_myInterestingGenes], use.names = FALSE),
                wb_myInterestingGenes_GO=unlist(wb_sp_geneID2GO[wb_myInterestingGenes], use.names = FALSE)))
  
  })
  
  
  # between species GO mapping 
  bs_GO_overlap <- sapply(1:length(myInterestingGenes_GO), FUN = function(k) {
    
    num <- length(intersect(myInterestingGenes_GO[[k]]$fb_myInterestingGenes_GO, 
                            myInterestingGenes_GO[[k]]$wb_myInterestingGenes_GO))
    m <- length(myInterestingGenes_GO[[k]]$fb_myInterestingGenes_GO)
    n <- length(myInterestingGenes_GO[[k]]$wb_myInterestingGenes_GO)
    
    sum(sapply(num:min(m,n), P_mass, m=m, n=n, N=length(geneNames_GO)))
  })
  
  # names(bs_GO_overlap) <- sapply(1:length(bs_GO_overlap), 
  #                                FUN = function(x) return(paste("P_values of co-cluster", x)))

  bs_GO_overlap_pvalues = as.data.frame(matrix(bs_GO_overlap))
  bs_GO_overlap_pvalues[[2]] = c(1:length(associated_gene_list))
  names(bs_GO_overlap_pvalues) = c('P value', "co-cluster ID")
  #pvalues_file_path = paste(gsub("'", "", root_dir), "GO_enrichment_analysis/onestep_result", "p_values.xls", sep = "/")
  #WriteXLS("bs_GO_overlap_pvalues", ExcelFileName = pvalues_file_path)
  
  WriteXLS("bs_GO_overlap_pvalues", ExcelFileName = output_file_p_values)
  # rownames(sp_overlap) <- names
  # colnames(sp_overlap) <- names
  # sp_overlap_old <- sp_overlap
  # sp_overlap[sp_overlap_old==0]=300
  # sp_overlap[sp_overlap_old!=0] <- -log(sp_overlap_old[sp_overlap_old!=0]*nrow(sp_overlap_old)*ncol(sp_overlap_old), base=10)
  # sp_overlap[sp_overlap<=0] <- 0
  # write.xlsx(sp_overlap, "within-species TROM scores.xlsx", row.names=TRUE, colNames=TRUE)
  
  }
