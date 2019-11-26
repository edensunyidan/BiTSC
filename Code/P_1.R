#September 29, 2019

#####################################
# sessionInfo()
### check the version of the packages
packageVersion("biocManager") #Bioconductor version 3.9 (BiocManager 1.30.4)
packageVersion("GO.db") #‘3.8.2’
packageVersion("topGO") #‘2.36.0’

#####################################
#rm(list = ls())
library(GO.db)
library(topGO)
library(Rgraphviz)
library(rjson)
library(WriteXLS)

###########################################################################################################################################
#Find enriched gene ontology (GO) enrichment
#http://current.geneontology.org/products/pages/downloads.html

#the package contains a function to find the enriched Gene Ontology (GO) terms 
#or GO slim terms (cut-down versions of the GO ontologies containing a subset of the terms in the whole GO, 
#giving a broad overview of the ontology content without the detail of the specific fine grained terms) in 
#any user-specified genes, so users can interpret the observed transcriptome mapping patterns by looking into the corresponding overlapping genes’ biological functions.

#root_dir = '/home/yidan/Dropbox/project_fly_worm'
root_dir = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm'"
#root_dir = "'/Users/yidansun/Dropbox (LASSO)/project_fly_worm/result/data/sp_tsc_hc/timecourse'"

result = fromJSON(file = gsub("'", "", paste(root_dir, "result/data/sp_tsc_hc/timecourse/10itr10rsp_kmeans.json", sep = "/")), simplify = TRUE)

result = fromJSON(file = gsub("'", "", paste(root_dir, "result/data/sp_tsc_hc/timecourse/10itr10rsp_sklearnsp.json", sep = "/")), simplify = TRUE)

fb_GOmappingfile = paste(root_dir, "GO_enrichment_analysis/GO_annotations_2019/gene_association.fb", sep = "/")
wb_GOmappingfile = paste(root_dir, "GO_enrichment_analysis/GO_annotations_2019/gene_association.wb", sep = "/")
#output_file_GO_terms = paste(root_dir, "GO_enrichment_analysis/result/top_50_enriched_terms_in_fly_and_worm_co-cluster.txt", sep = "/")
#output_file_GO_path =  gsub("'", "", paste(root_dir, "GO_enrichment_analysis/result/top_GO_enriched_terms_of_co-cluster", sep = "/"))
output_file_p_values = gsub("'", "", paste(root_dir, "GO_enrichment_analysis/onestep_result/P_value_of_hypergeometric_test_of_co-cluster.xls", sep ="/"))
output_file_ortho_overlap_pvalues = gsub("'", "", paste(root_dir, "GO_enrichment_analysis/onestep_result/orthology_enrichment_Pvalues.xls", sep ="/"))

### the order of $tclust_id and $clust are kept in R
#find.top.GO.terms(result, fb_GOmappingfile, wb_GOmappingfile, output_file_GO_path, topNum=50, save_graph=TRUE) #p_threshold = 10**(-3)
find.top.GO.terms(result, fb_GOmappingfile, wb_GOmappingfile, topNum=50, save_graph=TRUE) #p_threshold = 10**(-3)
hypergeometric_test(result, fb_GOmappingfile, wb_GOmappingfile, output_file_p_values)

library("readxl")

p_values_file = gsub("'", "", paste(root_dir, "result/data/sp_tsc_hc/timecourse/onestep_result_result30_hc/P_value_of_hypergeometric_test_of_co-cluster.xls", sep ="/"))
df = read_excel(p_values_file, sheet = 1)
df

