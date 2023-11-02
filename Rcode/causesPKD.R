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

pkd <- read.csv("C:/Users/nikod/Documents/PythonProjects/hivPKDproject/dataBits/PKDjoint.csv")


head(pkd)

colnames(pkd)

str(pkd)


pkd$TPO_wynik[is.na(pkd$TPO_wynik)] <- 0 # 0 instead of NA, so no test


table(pkd$TPO_wynik, useNA = "ifany")      # okazja do zbadania czułości testu
table(pkd$Orientacja, useNA = "ifany")


pkd_small <- pkd  %>% 
  mutate(positive = ifelse(TPO_wynik == "d", 1, 0 ))



pkd_small <-pkd_small[complete.cases(pkd_small$Powód),]

table(pkd_small$Powód, useNA = "ifany")


#### Here adding a column



library(tidyr)
smalll <- pkd_small %>%
  separate_rows(Powód, sep = "\\|") %>%
  group_by(Powód, positive) %>%
  summarise(count = n())

smalll2 <- smalll %>%
  group_by(Powód) %>%
  mutate(percentage = count / sum(count) * 100)




smalll2 <- smalll2[complete.cases(smalll2),]

write.csv(smalll2, 'groupedCausesPKD.csv')

# visualization


smalll2 <- read.csv("C:/Users/nikod/Documents/PythonProjects/hivPKDproject/dataBits/groupedCausesPKD.csv")

# changing variables names


translation_dict <- list(
  "IDU" = "IDU",
  "IDU+biseks." = "IDU+bisexual",
  "IDU+hetero" = "IDU+heterosexual",
  "IDU+homo" = "IDU+homosexual",
  "Zabieg medyczny" = "Medical procedure",
  "ciąża" = "Pregnancy",
  "ciąża u partnerki" = "Partner's pregnancy",
  "inne" = "Other",
  "k. biseksualne" = "Bisexual contacts",
  "k. heteroseksualne" = "Heterosexual contacts",
  "k. homoseksualne" = "Homosexual contacts",
  "krew" = "Blood",
  "krew+kontakty seksualne" = "Blood and sexual contacts",
  "namowa partnera" = "Partner's persuasion",
  "naruszenie skóry lub błony" = "Skin or mucous membrane damage",
  "objawy osłabionej odporności" = "Symptoms of weakened immunity",
  "początek nowego związku" = "New relationship",
  "seks MSM" = "MSM sex",
  "seks WSW" = "WSW sex",
  "sex worker" = "sex worker",
  "skierowanie przez lekarza" = "Doctor's recommendation",
  "uszkodzenie prezerwatywy" = "Condom damage",
  "wynik + partnera" = "Partner's positive HIV test",
  "życzenie klienta (brak ryzyka)" = "Client's wish (no risk)"
)


library(dplyr)

smalll2ENG <- smalll2 %>% 
  mutate(causes = ifelse(smalll2$Powód %in% names(translation_dict), 
                              translation_dict[smalll2$Powód], 
                              smalll2$Powód))


str(smalll2ENG)


smalll2ENG$causes <- unlist(smalll2ENG$causes)


new_row <- data.frame(
  X = 48,
  Powód  = "sex worker",
  positive = 1,
  count = 0,
  percentage =  0,
  causes = "sex worker"
)


smalll2ENG <- rbind(smalll2ENG, new_row)


causesPLT <- ggplot(smalll2ENG, aes(x= reorder(causes, count), y= count, fill= as.factor(positive)))+
  geom_col()+
  coord_flip()+ 
  geom_text(data = subset(smalll2ENG, positive  == 1),
                           aes(label = paste0(round(percentage, 1), "%")), hjust = -0.2)+
  theme_tufte()+ labs(fill = "HIV result:", title = "Causes of Testing", y = "", x = "") +
  theme(plot.title = element_text(hjust = 0.15, size = 16),
        legend.position = c(0.85, 0.15),
        legend.title = element_text(size = 14),  
        axis.title = element_text(size = 16),    
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12) )+
  scale_fill_manual(labels = c("negative", "positive"), values = c("#44b7c2", "#024b7a"))




ggsave(filename = "C:/Users/nikod/Documents/PythonProjects/hivPKDproject/visualizations/causesplott.pdf",
       plot = causesPLT, dpi = 300)





