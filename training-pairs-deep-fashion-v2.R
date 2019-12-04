library(dplyr)
library(jpeg)
library(stringr)
options(stringsAsFactors = FALSE)
options(scipen=999)


mother_folder <- "C:\\Users\\emilio\\Documents\\fashion-dataset" #where to find styles.csv
image_folder <- "C:\\Users\\emilio\\Documents\\fashion-dataset\\thesis\\images"
available_images <- str_match(list.files(image_folder), "(.*?).jpg")[ ,2]

setwd(mother_folder)

product_images_df <- read.csv("styles.csv") %>% 
                     filter(id %in% available_images) %>%
                     mutate(id = as.numeric(id)) %>% arrange(id) %>%
                     filter(gender %in% c("Men", "Women", "Boys", "Girls", "Unisex")) %>%
                     filter(!masterCategory %in% c("Free Items", "Home", "Personal Care")) %>%
                     filter(!subCategory %in% c("Jewellery", "Cufflinks", "Loungewear and Nightwear", 
                                                "Mufflers", "Accessories", "Wallets", "Stoles", "Saree"))  %>%
                     filter(!articleType %in% c("Salwar and Dupatta", "Churidar", "Kurtis", "Dupatta", 
                                                "Capris", "Earrings", "Wallets", 
                                                "Clothing Set", "Kurta Sets", "Travel Accessory", "Waist Pouch", 
                                                "Patiala", "Mobile Pouch", "Lehenga Choli", 
                                                "Clutches", "Trunk", "Tights", "Shrug", "Tablet Sleeve", "Briefs", 
                                                "Salwar", "Rompers")) %>%
                     select(-season, -year, -usage)

partition_in_groups <- product_images_df %>% 
                       group_by(masterCategory, subCategory, articleType, gender) %>%
                       summarise(n_items = n()) %>% ungroup() %>%
                       mutate(key_group = paste0(masterCategory, ".", subCategory, ".", articleType, ".", gender))

#filter jus the groups that ahve at least 5 observations
condition.for.sample <- partition_in_groups$n_items >= 5 
key_groups_df <- data.frame(key = partition_in_groups$key_group[condition.for.sample])
partition_in_groups <- partition_in_groups %>%
                       filter(key_group %in% key_groups_df$key)

#downsample the groups that have more than 50 observations
condition.for.downsampling <- partition_in_groups$n_items > 50
groups_to_extract_50_random_observations <- key_groups_df$key[condition.for.downsampling]
images_partition_in_groups <- partition_in_groups %>% 
                              left_join(product_images_df) %>%
                              mutate(img_filename = paste0(id, ".jpg")) %>% 
                              mutate(is.more.50 = key_group %in% groups_to_extract_50_random_observations)
aux_df <- images_partition_in_groups %>% group_by(key_group, is.more.50) %>%
          summarise(group_cardinality = length(key_group)) %>%
          filter(is.more.50) %>% ungroup()
generate_randon_numbers <- function(cardinality){return(sample(1:cardinality, 50))}
set.seed(1826)
list.ids_selected <- lapply(aux_df$group_cardinality, generate_randon_numbers)
names(list.ids_selected) <- aux_df$key_group
images_partition_in_groups <- images_partition_in_groups %>% 
                              left_join(aux_df) %>% group_by(key_group) %>%
                              mutate(id_within_group = 1:length(key_group)) %>% ungroup() %>%
                              mutate(is.selected = is.more.50 == FALSE) 
for(key in names(list.ids_selected)){
  images_partition_in_groups <- images_partition_in_groups %>%
                                mutate(is.selected = ifelse(key_group == key, 
                                                            ifelse(id_within_group %in% list.ids_selected[[key]], TRUE, FALSE), 
                                                            is.selected))
}
images_partition_in_groups_final <- images_partition_in_groups %>% filter(is.selected)

#at the end, we get the following groups
summary_groups <- as.data.frame(table(images_partition_in_groups_final$key_group)) %>%
                  mutate(n_sim_pairs = (Freq * (Freq - 1)) / 2)
max.qty.sim.pairs <- sum(summary_groups$n_sim_pairs)

#at the end, we filtered on the following items
images_partition_in_groups_final <- images_partition_in_groups_final %>%
                                    filter(key_group %in% summary_groups$Var1)




set.seed(307)
for(i in 1:100){
images_partition_in_groups_final <- images_partition_in_groups_final[sample(1:nrow(images_partition_in_groups_final)), ]
}

images_partition_in_groups_final_div <- images_partition_in_groups_final %>%
                                          group_by(key_group) %>%
                                          mutate(div = ifelse(which(key_group == key_group) %in% 1:round((length(key_group) * 0.599)), 
                                                        'train', 
                                                        ifelse(which(key_group == key_group) %in% (round((length(key_group) * 0.499))+1):round((length(key_group) * 0.8)), 
                                                               'val', 
                                                               'test'))) %>% ungroup() %>%
                                          group_by(key_group) %>%
                                          mutate(div = ifelse(rep(sum(table(div) == 1) > 0, length(div)), 
                                                              ifelse(div == 'train', 
                                                                     div, 
                                                                     ifelse(runif(1) > 0.5, 'val', 'test')), 
                                                              div))



images_partition_in_groups_final_train <- images_partition_in_groups_final_div %>% filter(div == 'train')
images_partition_in_groups_final_val <- images_partition_in_groups_final_div %>% filter(div == 'val')
images_partition_in_groups_final_test <- images_partition_in_groups_final_div %>% filter(div == 'test')






#construction of similar pairs train
sim_pairs_df_train <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_train$key_group)){
  
  df_group <- images_partition_in_groups_final_train %>% filter(key_group == key)
  
  df_group_sim_pairs <- as.data.frame(t(combn(df_group$img_filename, 2))) %>%
                        mutate(group = key)
  sim_pairs_df_train <- rbind(sim_pairs_df_train, df_group_sim_pairs)
  
}

#construction of non similar pairs train
nonsim_pairs_df_train <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_train$key_group)){
  
  df_group <- images_partition_in_groups_final_train %>% filter(key_group == key)
  
  n_sim_pairs <- nrow(df_group) * (nrow(df_group) - 1) / 2
  n_nonsim_products <- round(n_sim_pairs / nrow(df_group))
  
  df_products_nonsim <- images_partition_in_groups_final_train %>% 
                        filter(masterCategory != unique(df_group$masterCategory)
                               &
                               subCategory != unique(df_group$subCategory))
  df_group_nonsim_pairs <- data.frame(V1 = character(), V2 = character(), group = character())
  for(i in 1:nrow(df_group)){
    products_nonsim <- data.frame(V1 = df_group$img_filename[i],
                                  V2 = df_products_nonsim$img_filename[sample(1:nrow(df_products_nonsim), n_nonsim_products)], 
                                  group = df_group$key_group[i])
    df_group_nonsim_pairs <- rbind(df_group_nonsim_pairs, products_nonsim)
  }
  
  nonsim_pairs_df_train <- rbind(nonsim_pairs_df_train, df_group_nonsim_pairs)
}


df_pairs_train <- rbind(sim_pairs_df_train %>% mutate(label = 1), nonsim_pairs_df_train %>% mutate(label = 0))
for(i in 1:100){
  df_pairs_train <- df_pairs_train[sample(1:nrow(df_pairs_train)), ]
}










#construction of similar pairs val
sim_pairs_df_val <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_val$key_group)){
  
  df_group <- images_partition_in_groups_final_val %>% filter(key_group == key)
  
  df_group_sim_pairs <- as.data.frame(t(combn(df_group$img_filename, 2))) %>%
    mutate(group = key)
  sim_pairs_df_val <- rbind(sim_pairs_df_val, df_group_sim_pairs)
  
}

#construction of non similar pairs
nonsim_pairs_df_val <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_val$key_group)){
  
  df_group <- images_partition_in_groups_final_val %>% filter(key_group == key)
  
  n_sim_pairs <- nrow(df_group) * (nrow(df_group) - 1) / 2
  n_nonsim_products <- round(n_sim_pairs * 1.0000000001 / nrow(df_group))
  
  df_products_nonsim <- images_partition_in_groups_final_val %>% 
    filter(masterCategory != unique(df_group$masterCategory)
           &
             subCategory != unique(df_group$subCategory))
  df_group_nonsim_pairs <- data.frame(V1 = character(), V2 = character(), group = character())
  for(i in 1:nrow(df_group)){
    products_nonsim <- data.frame(V1 = df_group$img_filename[i],
                                  V2 = df_products_nonsim$img_filename[sample(1:nrow(df_products_nonsim), n_nonsim_products)], 
                                  group = df_group$key_group[i])
    df_group_nonsim_pairs <- rbind(df_group_nonsim_pairs, products_nonsim)
  }
  
  nonsim_pairs_df_val <- rbind(nonsim_pairs_df_val, df_group_nonsim_pairs)
}

df_pairs_val <- rbind(sim_pairs_df_val %>% mutate(label = 1), nonsim_pairs_df_val %>% mutate(label = 0))
for(i in 1:100){
  df_pairs_val <- df_pairs_val[sample(1:nrow(df_pairs_val)), ]
}










#construction of similar pairs test
sim_pairs_df_test <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_test$key_group)){
  
  df_group <- images_partition_in_groups_final_test %>% filter(key_group == key)
  
  df_group_sim_pairs <- as.data.frame(t(combn(df_group$img_filename, 2))) %>%
    mutate(group = key)
  sim_pairs_df_test <- rbind(sim_pairs_df_test, df_group_sim_pairs)
  
}

#construction of non similar pairs
nonsim_pairs_df_test <- data.frame(V1 = character(), V2 = character(), group = character())
for(key in unique(images_partition_in_groups_final_test$key_group)){
  
  df_group <- images_partition_in_groups_final_test %>% filter(key_group == key)
  
  n_sim_pairs <- nrow(df_group) * (nrow(df_group) - 1) / 2
  n_nonsim_products <- round(n_sim_pairs  * 1.0000000001/ nrow(df_group))
  
  df_products_nonsim <- images_partition_in_groups_final_test %>% 
    filter(masterCategory != unique(df_group$masterCategory)
           &
             subCategory != unique(df_group$subCategory))
  df_group_nonsim_pairs <- data.frame(V1 = character(), V2 = character(), group = character())
  for(i in 1:nrow(df_group)){
    products_nonsim <- data.frame(V1 = df_group$img_filename[i],
                                  V2 = df_products_nonsim$img_filename[sample(1:nrow(df_products_nonsim), n_nonsim_products)], 
                                  group = df_group$key_group[i])
    df_group_nonsim_pairs <- rbind(df_group_nonsim_pairs, products_nonsim)
  }
  
  nonsim_pairs_df_test <- rbind(nonsim_pairs_df_test, df_group_nonsim_pairs)
}


df_pairs_test <- rbind(sim_pairs_df_test %>% mutate(label = 1), nonsim_pairs_df_test %>% mutate(label = 0))
for(i in 1:100){
  df_pairs_test <- df_pairs_test[sample(1:nrow(df_pairs_test)), ]
}


images_available1 <- data.frame(V1 = list.files(image_folder), is.present = TRUE)
images_available2 <- data.frame(V2 = list.files(image_folder), is.present = TRUE)

df_pairs_train <- df_pairs_train %>% left_join(images_available1) %>% filter(is.present) %>% select(-is.present) %>%
                                     left_join(images_available2) %>% filter(is.present) %>% select(-is.present)

df_pairs_val <- df_pairs_val %>% left_join(images_available1) %>% filter(is.present)  %>% select(-is.present) %>%
                                     left_join(images_available2) %>% filter(is.present) %>% select(-is.present)

df_pairs_test <- df_pairs_test %>% left_join(images_available1) %>% filter(is.present) %>% select(-is.present) %>%
                                   left_join(images_available2) %>% filter(is.present) %>% select(-is.present)


write.csv(df_pairs_train, "training_pairs.csv")
write.csv(df_pairs_val, "validation_pairs.csv")
write.csv(df_pairs_test, "test_pairs.csv")
write.csv(data.frame(x = images_available1$V1[images_available1$V1 %in% unique(images_partition_in_groups_final$img_filename)]), "img_filenames.csv")



