library(ggplot2)
library(ggthemes)
library(gridExtra)
library(dplyr)
library(rethinking)
library(sf)
library(rnaturalearth)

plt_pos <- theme(plot.title.position = "plot")

#library(rnaturalearthhires)


# 48-94 powody

# Notes
# IDU - injective drug user
# STP - szybkie
# TPR - zwykłe 
# TPO - potwierdzone drugie (dodatni hiv)

pkd <- read.csv("PKDjoint.csv")

head(pkd)

colnames(pkd)

str(pkd)


unique(pkd$Powód)

causes_list <- strsplit(pkd$Powód, "\\|")
all_causes <- unlist(causes_list)
unique_causes <- unique(all_causes)

unique_causes <- unique_causes[complete.cases(unique_causes)]

pkd_small



pkd$separate_causes <- strsplit(pkd$Powód, "\\|")
cause_counts <- table(unlist(pkd$separate_causes))
sorted_cause_counts <- sort(cause_counts, decreasing = TRUE)



library(tidyr)
smalll <- pkd_small %>%
  separate_rows(Powód, sep = "\\|") %>%
  group_by(Powód, positive) %>%
  summarise(count = n())

smalll2 <- smalll %>%
  group_by(Powód) %>%
  mutate(percentage = count / sum(count) * 100)



smalll2 <- smalll2[complete.cases(smalll2),]





####

pkd$TPO_wynik[is.na(pkd$TPO_wynik)] <- 0 # 0 instead of NA, so no test


table(pkd$TPO_wynik, useNA = "ifany")      # okazja do zbadania czułości testu
table(pkd$Orientacja, useNA = "ifany")
unique(df$Orientacja)

# df_or <- df %>% filter(Orientacja == "biseksualna" | Orientacja ==
#                          "heteroseksualna" |  Orientacja  ==    "homoseksualna")
# mutate(Non_het_norm = ifelse(Orientacja == "biseksualna" | Orientacja == "homoseksualna", 1, 0)) %>% 


pkd_small <- pkd  %>% 
  mutate(positive = ifelse(TPO_wynik == "d", 1, 0 ))

pkd_small <- pkd_small[,c(2:19,48:94,120,121)]

names(pkd_small)

pkd_small <-pkd_small[complete.cases(pkd_small$Powód),]

table(pkd_small$Powód, useNA = "ifany")


# powód kontroli
ggplot(smalll2, aes(x= reorder(Powód, count), y= count, fill= as.factor(positive)))+ geom_col()+
  coord_flip()+  geom_text(data = subset(smalll2, positive  == 1),
                           aes(label = paste0(round(percentage, 1), "%")), hjust = -0.2)+
  theme_tufte()+ labs(fill = "Pozytywny:", title = "Powód wykonania badania",
                      subtitle = "PKD, dane sumaryczne", y = "", x = "Powód")+ plt_pos


# ilość partnerów




# rodzaj seksu
# kategoria gay passive | gay active

pkd_small

pkd_sex <- pkd_small %>% 
  mutate(anal = case_when(
    Kontakty_seks_rok_anal_active == 1 ~ 1,
    Kontakty_seks_rok_anal_passive == 1 ~ 1
    )) %>% 
  mutate(oral = case_when(
    Kontakty_seks_rok_oral_passive == 1 ~ 1,
    Kontakty_seks_rok_oral_active == 1 ~ 1  
    ))
           
           
           
           



mutate(category = case_when(
  positive == 1 ~ "Positive",
  positive == 0 ~ "Negative",
  TRUE ~ "Other"
))

