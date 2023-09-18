library(ggplot2)
library(ggthemes)
library(gridExtra)
library(dplyr)
library(rethinking)

th <- theme_tufte(base_size = 14) + theme(plot.title.position = "plot")
plt_pos <- theme(plot.title.position = "plot")

# connecting nad loading data ---------------------------------------------



list.files("C:/Users/nikod/Documents/RProjects/hivProject/PZHdata")


df2 <- read.csv("PZHdata/HIV_dane indywidualne_(Rok rozpoznania, Województwo, Grupa wieku, Droga zakażenia, Licznik).csv",
                fileEncoding = "Windows-1250")

df4 <- read.csv("PZHdata/HIV_dane indywidualne_(Rok rozpoznania, Województwo, Płeć, Droga zakażenia, Licznik).csv",
                fileEncoding = "Windows-1250")

names(df2)
names(df4)

identical(df2$droga.zakazenia, df4$droga.zakazenia)
# as they are the same I will join the missing colum and stay with one dataset


df2$Plec <- df4$Plec 
  
table(df2$jednostka.chorobowa)
table(df2$licznik)

#therefore we have only data about 17241 cases from year 1999 to 2017 
HIVdata <- df2  %>% 
  filter(jednostka.chorobowa== "HIV" ) %>% 
  select(rok.rozpoznania, Wojewodztwo.zamieszkania, grupa.wieku, Plec, droga.zakazenia)
nrow(HIVdata)

HIVdata <- data.frame(lapply(HIVdata, function(x) iconv(x, from = "", to = "UTF-8")))

names(HIVdata) <- c("rok", "Województwo", "wiek", "plec", "droga_zakazenia")


HIVdata <- data.frame(lapply(HIVdata, function(x) ifelse(x == "brak danych", NA, x)))

HIVdata$Województwo <- toupper(HIVdata$Województwo)




HIVdata <- HIVdata %>% 
  mutate(droga_zakazenia = gsub(" ", "_", droga_zakazenia))




HIVdata$Województwo <- iconv(
  HIVdata$Województwo, "UTF-8", "ASCII//TRANSLIT")








write.csv(HIVdata, file = "HIV19992017PZH.csv", row.names = FALSE, )


table(HIVdata$droga_zakazenia)

sum(is.na(HIVdata$droga_zakazenia))



# starting analyzsis ------------------------------------------------------

HIVdata2 <- read.csv("HIV19992017PZH.csv")




table(HIVdata$rok.rozpoznania)
table(HIVdata$droga.zakazenia)



counts<- HIVdata %>% 
  group_by(rok) %>% 
  summarize(count = n())

counts$rok  <-  as.integer(counts$rok)




ggplot(counts, aes(x= rok, y= count)) + 
  geom_col(color= "white", fill= "steelblue")+ 
  scale_x_continuous(breaks = c(1999:2017))+ theme_clean() +
  plt_pos+ geom_smooth(se = FALSE, color= "red")+
  scale_y_continuous(breaks = seq(0, 1600, by= 200))+
  ylab("")+ xlab("")+ labs(title= "Number of detected HIV positive people",
                           subtitle = "Poland, 1999 to 2017")

HIVdata <- read.csv("HIV19992017PZH.csv")

counts<- HIVdata %>% 
  group_by(rok) %>% 
  summarize(count = n())

counts$rok  <-  as.integer(counts$rok)

ggplot(counts, aes(x= rok, y= count)) + geom_line()+
  scale_x_continuous(breaks = c(1999:2017))+ theme_clean()+
  scale_y_continuous(breaks = seq(0, 1600, by= 200), limits = c(0, 1600))+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



counts_voiv <- HIVdata %>% 
  group_by(rok, Województwo) %>% 
  summarize(count = n())






counts_voiv <- counts_voiv %>% filter(Województwo != "BRAK DANYCH")

counts_voiv$Województwo <- as.factor(counts_voiv$Województwo)



### what about relative to 100K citizens?
ggplot(counts_voiv, aes(x= rok.rozpoznania, y= count)) + 
  geom_col( fill= "steelblue")+ 
  scale_x_continuous(breaks = seq(1999, 2017, by=2))+ theme_fivethirtyeight() +
  plt_pos+ facet_wrap(~Województwo)+ geom_smooth(method= lm, se = FALSE, color= "red")+
  scale_y_continuous(breaks = seq(0, 1600, by= 200))+
  ylab("")+ xlab("")+ labs(title= "Number of detected HIV positive people",
                           subtitle = "Poland, 1999 to 2017")


counts_voiv_people<- merge(counts_voiv, ludnosc, by= "Województwo")

counts_voiv_people$count_per_100k <- counts_voiv_people$count / counts_voiv_people$population_size * 100000

ggplot(counts_voiv_people, aes(x= rok.rozpoznania, y= count_per_100k)) + 
  geom_col( fill= "steelblue")+ 
  scale_x_continuous(breaks = seq(1999, 2017, by=2))+ theme_fivethirtyeight() +
  plt_pos+ facet_wrap(~Województwo)+ geom_smooth(method= lm, se = FALSE, color= "red")+
 # scale_y_continuous(breaks = seq(0, 1600, by= 200))+
  ylab("")+ xlab("")+ labs(title= "Detected HIV posit. people per 100k citizens",
                           subtitle = "Poland, 1999 to 2017")


########################################
counts <- HIVdata %>% 
  group_by(rok.rozpoznania) %>% 
  summarize(count = n())

ggplot(counts, aes(x= rok.rozpoznania, y= count)) + 
  geom_line(color = "steelblue") + geom_point()+
  scale_x_continuous(breaks = c(1999:2017)) + 
  theme_fivethirtyeight()+ plt_pos


# New thingy on PKD hours -------------------------------------------------


pkdData <- read.csv("dataAboutPKD.csv")
ludnos_100k <- read.csv("wojPer100kTests.csv")
poland_map <- readRDS(file= "poland_mapDF.RDS")

pkdData_mod <- pkdData %>% mutate(hPerWeek= dniwTyg * godzPerDzien)

pkdData_mod$wojewodztwo <- as.factor(pkdData_mod$wojewodztwo )


pkdData_mod_WOJ <- pkdData_mod %>%
  group_by(wojewodztwo) %>%
  summarise(hours_perWeek = sum(hPerWeek), num_points = n()) %>%
  arrange(desc(hours_perWeek)) %>% 
  rename(Województwo = wojewodztwo)
  ungroup()


pkdData_mod_WOJ <- as.data.frame(pkdData_mod_WOJ)
pkdData_mod_WOJ$Województwo <- trimws(pkdData_mod_WOJ$Województwo)
merged_regions_count <- merge(pkdData_mod_WOJ, ludnos_100k, by= "Województwo")


merged_regions_countL <- merge(merged_regions_count, 
                          df_posit_region_group_percSUBSET, by= "Województwo")

# for 2015 - 2022
Woj_large <- merged_regions_countL %>% 
  select(- c(count, positive)) %>% 
  rename(perc_posit_tests= percent)
Woj_large$hours_perWeek_100k <- Woj_large$hours_perWeek / Woj_large$population_size * 100000
saveRDS(Woj_large, "woj_large.csv")

############ Map vis

Woj_large_map <- merge(poland_map, Woj_large, by= "Województwo")

map_Hours_perWeek_100k <- ggplot() +
  geom_sf(data = Woj_large_map, aes(fill = hours_perWeek_100k)) + 
  geom_sf_label(data = Woj_large_map,
                aes(label = paste0(round(hours_perWeek_100k, 2))) , size = 3, fontface = "bold")+
  scale_fill_gradient(low = "white", high = "skyblue", na.value = "red") + # ,  limits = c(0.5, 4)
  labs(title = "PKD weekly open hours per 100k citizens") +
  theme_void()+
  theme(legend.position = "none")
map_Hours_perWeek_100k



ggplot(Woj_large, aes(x= reorder(factor(Województwo), hours_perWeek_100k), y= hours_perWeek_100k))+ 
  geom_col(fill= "steelblue", color="white")+ coord_flip()+
  labs(title= "Amount of weakly opening hours of PKD's per 100k region citizens ",x= "Województwo",
       y= "hours per week, per 100k citizens")+ th
  
# dostępność -> więcej testów  -> mniej zakażeń na 100k (?) | mniej procentowo
# może porównać roczne różnice per województwo?

correlation100KWekk <- cor(Woj_large$testsPer100k, Woj_large$hours_perWeek)

ggplot(Woj_large, aes(x= testsPer100k, y= hours_perWeek))+ geom_point()+ th+
  geom_smooth(method = lm, color= "red")+ labs(title= "Tests per 100k vs weekly opening hours", subtitle = paste(
    "Correlation =", round(correlation100KWekk, 2)))


ggplot(Woj_large, aes(x= testsPer100k, y= hours_perWeek_100k))+ geom_point()+ th+
  geom_smooth(method = lm, color= "red")+ labs(title= "Tests per 100k vs opening hours per 100K")


ggplot(Woj_large, aes(x= hours_perWeek_100k, y=  perc_posit_tests))+ geom_point()+ th+
  geom_smooth(method = lm, color= "red")









plt4 <- ggplot(counts, aes(x= rok, y= count)) + geom_line(linewidth= 1.5)+
  scale_x_continuous(breaks = c(1999:2017))+
  scale_y_continuous(breaks = seq(0, 1600, by= 200), limits = c(0, 1600))+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text = element_text(size = 18),  
    axis.title = element_text(size = 20),  
    legend.text = element_text(size = 18),  
    legend.title = element_text(size = 20), title = element_text(size = 22)) + 
  labs(title = "Number of detected HIV positive people",
       subtitle = "Poland, 1999 to 2017", x = "", y= "")+
  theme(plot.title.position = "plot")+

  theme(
    panel.background = element_rect(fill = "lightblue"),
    panel.grid.major = element_line(linewidth = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white"),
    
    plot.background = element_rect(fill = "lightblue")
  )  
  
plt4








