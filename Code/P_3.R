ortholog_enrichment <- function(gene_lists, sp1_sp2_orthologs){

  root_dir = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm'"
  gene_lists = associated_gene_list[1:100]
  load(file = gsub("'", "", paste(root_dir, "tsc_data/orthologs_data_uniq.Rdata", sep = "/")))
  links = orthologs_data_uniq
  sp1_sp2_orthologs = links[1:2] #11403     2
  
  sp1_interested_ortholog <- sapply(1:length(gene_lists), FUN=function(x)
    gene_lists[[x]]$fb_gene_lists[which(gene_lists[[x]]$fb_gene_lists %in% sp1_sp2_orthologs[,1])])
  sp2_interested_ortholog <- sapply(1:length(gene_lists), FUN=function(x)
    gene_lists[[x]]$wb_gene_lists[which(gene_lists[[x]]$wb_gene_lists %in% sp1_sp2_orthologs[,2])])
  
  
  # calculate the P-value for each co-cluster
  overlap<-sapply(1:length(gene_lists), FUN=function(x){
    sp1_ = sp1_interested_ortholog[[x]]
    sp2_ = sp2_interested_ortholog[[x]]
    num <- length(which(sp1_sp2_orthologs[,1] %in% sp1_ & sp1_sp2_orthologs[,2] %in% sp2_))
    m <- length(sp1_)
    n <- length(sp2_)
    sum(sapply(num:min(m,n), P_mass, m=m, n=n, N=nrow(sp1_sp2_orthologs)))
  })
  
  #overlap_old <- overlap
  #overlap[overlap_old==0]=300
  #overlap[overlap_old!=0] <- -log(overlap_old[overlap_old!=0]*nrow(overlap_old)*ncol(overlap_old), base=10)
  #overlap[overlap<=0] <- 0
  
  ortho_overlap_pvalues = as.data.frame(matrix(overlap))
  ortho_overlap_pvalues[[2]] = c(1:length(gene_lists))
  names(ortho_overlap_pvalues) = c('P value', "co-cluster ID")
  
  WriteXLS("ortho_overlap_pvalues", ExcelFileName = output_file_ortho_overlap_pvalues)
  
  
}
