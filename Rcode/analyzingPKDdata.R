library(ggplot2)
library(ggthemes)
library(gridExtra)
library(dplyr)
library(rethinking)
library(sf)
library(rnaturalearth)
#library(rnaturalearthhires)


# 48-94 powody

# Notes
# IDU - injective drug user
# STP - szybkie
# TPR - zwykłe 
# TPO - potwierdzone drugie (dodatni hiv)

df <- read.csv("PKDjoint.csv")

head(df)

colnames(df)

str(df)


unique(df$Powód)

causes_list <- strsplit(df$Powód, "\\|")
all_causes <- unlist(causes_list)
unique_causes <- unique(all_causes)




# basicVis ----------------------------------------------------------------



df$Data_przeprowadzenia_rozmowy <- as.Date(df$Data_przeprowadzenia_rozmowy)

df$Year <- format(df$Data_przeprowadzenia_rozmowy, "%Y")

occurencesByYear <- ggplot(df, aes(x= factor(Year), y= after_stat(count))) + 
  geom_bar(fill= "steelblue", width = 0.8) + theme_tufte()+
  ggtitle("Number of observations by year") +
  xlab("Year") +
  ylab("Count")
occurencesByYear



# n tests from 2015 to 2019


df$Year <- as.numeric(df$Year)

df_year_smaller <- df %>% filter(Year < 2020) %>% 
  group_by(Year) %>% 
  summarise(count = n())

df_year <- df %>% 
  group_by(Year) %>% 
  summarise(count = n())

# fit a linear model
model <- lm(count ~ Year, data = df_year_smaller)

# make predictions for the next 5 years
new_data <- data.frame(Year = 2020:2024)
predictions <- predict(model, newdata = new_data)

new_data$predict <- predictions 

df_year$pred <- c(rep(1,5), predictions[-(4:5)] ) 

# plot the results
ggplot(df_year_smaller, aes(x = Year, y = count)) +
  geom_col(width = 0.1) +
  geom_line(aes(y = predict(model), color = "Linear Model"), linewidth= 2) +
  geom_line(data = new_data, aes(y = predictions, color = "Predictions"), linetype = "dashed") +
  scale_color_manual(values = c("Linear Model" = "red", "Predictions" = "red")) +
  labs(title = "Linear Model", x = "Year", y = "Count")+
  scale_x_continuous(breaks = 2015:2025)+ th




subset(df, subset=rownames(df) == 'r1') 

OccurencesByAge <- ggplot(df, aes(x = Wiek)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color= "white") +
  ggtitle("Occurrences by Age") +
  xlab("Age") +
  ylab("Frequency")+ 
  xlim(0, 83)+ theme_tufte()
OccurencesByAge



# wrangling ---------------------------------------------------------------


# Powód
unique(df$Powód)
# seks MSM, k. biseksualne, k. homoseksualne    - regex needed (>600 uniques)

# Orientacja
unique(df$Orientacja)
# homoseksualna, biseksualna

str(df)
df$Orientacja <- factor(df$Orientacja)
df$TPO_wynik[is.na(df$TPO_wynik)] <- 0 # 0 instead of NA, so no test


table(df$TPO_wynik, useNA = "ifany")      # okazja do zbadania czułości testu


table(df$Orientacja, useNA = "ifany")
unique(df$Orientacja)

df_or <- df %>% filter(Orientacja == "biseksualna" | Orientacja ==
                         "heteroseksualna" |  Orientacja  ==    "homoseksualna")



df_posit_orient <- df_or  %>% 
  mutate(positive = ifelse(TPO_wynik == "d", 1, 0 )) %>% 
  mutate(Non_het_norm = ifelse(Orientacja == "biseksualna" | Orientacja == "homoseksualna", 1, 0)) %>% 
  select(Płeć, Wiek, Year, positive, Non_het_norm, Województwo )

table(df_posit_orient$positive, useNA = "ifany")
table(df_posit_orient$Orientacja, useNA = "ifany")



subsetPLUS_df_posit_orient <- subset(df_posit_orient, positive == 1)
positNominalYear <- ggplot(subsetPLUS_df_posit_orient, 
                           aes(x= Year, y= after_stat(count)))+ 
  geom_bar(width = 0.7)+ theme_tufte() +
  ggtitle("HIV positive by Year") +
  xlab("Year") +
  ylab("Count")
positNominalYear

grid.arrange(occurencesByYear, positNominalYear, ncol = 2)










table(df_posit_orient$Płeć)
subsetGENDER_df_posit_orient <- subset(df_posit_orient, Płeć %in% c("M", "K"))
SexIdentityGender <- ggplot(subsetGENDER_df_posit_orient) +
                                      geom_bar(aes(x= factor(Płeć),
                                       y= after_stat(count), 
                                       fill= factor(Non_het_norm)),
 width = 0.6)+
scale_fill_brewer(palette="Purples")+ # Pastel2
  theme_tufte() + ggtitle("Sexual identity and gender")+
  labs(x = "Gender", y = "Count", fill = "Non-hetero")
SexIdentityGender




# percentage orientation and positive vis ---------------------------------


df_posit_orient_group <- df_posit_orient %>%
  group_by(Non_het_norm, positive) %>%
  summarise(count = n()) %>%
  ungroup()

df_posit_orient_group_perc <- df_posit_orient_group %>% 
  group_by(positive) %>% 
  mutate(percent = count / sum(count) * 100)





hivResultSexIdentity <- ggplot(df_posit_orient_group_perc,
                               aes(x = factor(positive), y= percent, 
                                   fill = factor(Non_het_norm))) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  ggtitle("HIV test result and sexual identity")+
  labs(x = "HIV positive", y = "Count in %", fill = "Non-hetero") +
  scale_y_continuous(limits = c(0, 100))+
  scale_fill_brewer(palette="Purples")+ # Pastel2
  geom_text(aes(label = paste0(round(percent, 1), "%")),
            position = position_dodge(width = 0.7),
            vjust = -0.5)+
  theme_tufte()
hivResultSexIdentity









df_yearly_percentages <- df_posit_orient %>%
  group_by(Year) %>%
  summarize(positive_pct = mean(positive == 1) * 100)

# Create the bar chart
ggplot(df_yearly_percentages, aes(x = Year, y = positive_pct)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
  labs(title = "Percentage of HIV positive test results",
       x = "Year",
       y = "Count in %") +
 scale_y_continuous(limits = c(0, 4))+ 
  geom_text(aes(label = paste0(round(positive_pct, 1), "%")),
            position = position_dodge(width = 0.7),
            vjust = -0.5)+
  theme_tufte()


# mapa polski i procent pozytywnych, albo procent testujących się na 100 k mieszkańców


str(df_or$TPO_wynik, useNA = "ifany")



df_posit_region <- df_or  %>% 
  mutate(positive = ifelse(TPO_wynik == "d", 1, 0 )) %>% 
  select(Płeć, Wiek, Year, positive, Województwo )

df_posit_region$Województwo <- iconv(
  df_posit_region$Województwo, "UTF-8", "ASCII//TRANSLIT")



df_posit_region <- na.omit(df_posit_region)

table(df_posit_region$Województwo, useNA = "ifany")


df_posit_region_group <- df_posit_region %>%
  group_by(Województwo, positive) %>%
  summarise(count = n()) %>%
  ungroup()

df_posit_region_group_perc <- df_posit_region_group %>% 
  group_by(Województwo) %>% 
  mutate(percent = count / sum(count) * 100)




poland_map <- ne_states(country = "Poland", returnclass = "sf")
df_posit_region_group_percSUBSET <- subset(df_posit_region_group_perc, positive == 1)

poland_map$name_alt <- toupper(poland_map$name_alt)

saveRDS(df_posit_region_group_perc, "df_posit_region_group_perc.RDS")



poland_map$name_alt <- iconv(
  poland_map$name_alt, "UTF-8", "ASCII//TRANSLIT")

df_posit_region_group_percSUBSET$Województwo <- iconv(
  df_posit_region_group_percSUBSET$Województwo, "UTF-8", "ASCII//TRANSLIT")

poland_map$name_alt 
df_posit_region_group_percSUBSET$Województwo

poland_map <- poland_map %>% rename(Województwo = name_alt)

saveRDS(poland_map, "poland_mapDF.RDS")



map_percent_positive <- merge(poland_map, df_posit_region_group_percSUBSET, by = "Województwo")

length(map_percent_positive$Województwo)





map_perc_positive_region <- ggplot() +
  geom_sf(data = map_percent_positive, aes(fill = percent)) + 
  geom_sf_label(data = map_percent_positive, aes(label = paste0(round(percent, 1), "%"))
                , size = 3, fontface = "bold")+
  scale_fill_gradient(low = "white", high = "red", na.value = "blue") + # ,  limits = c(0.5, 4)
  labs(title = "Percentage of Positive PKD test results", subtitle = "Summary data from 2015 to 2022") +
  theme_void() +
  theme(legend.position = "none")
map_perc_positive_region


# GUS data for year 2021 https://www.polskawliczbach.pl/Wojewodztwa
ludnosc <- read.csv("WojewodztwaLudnosc.csv")
write.csv(merged_regions_count, "wojPer100kTests.csv", row.names = FALSE)

df_region_count <- df_posit_region %>% 
  group_by(Województwo) %>% 
  summarise(testsCount = n())


merged_regions_count <- merge(ludnosc, df_region_count, by= "Województwo")


testsPerCapita <- merged_regions_count$testsCount / merged_regions_count$population_size
merged_regions_count$testsPer100k <- testsPerCapita * 100000


merged_regions_count_map <- merge(poland_map, merged_regions_count, by= "Województwo" )


saveRDS(merged_regions_count_map, "merged_regions_count_map.RDS")

 map_testper100k <- ggplot() +
  geom_sf(data = merged_regions_count_map, aes(fill = testsPer100k)) + 
   geom_sf_label(data = merged_regions_count_map,
                 aes(label = paste0(round(testsPer100k, 0))) , size = 3, fontface = "bold")+
  scale_fill_gradient(low = "white", high = "green", na.value = "blue") + 
  labs(title = "Number of tests per 100k citizens") +
  theme_void() +
  theme(legend.position = "none")
 map_testper100k
 
 
grid.arrange(map_perc_positive_region, map_testper100k, ncol = 2)
 



 correlation_urb_test100k <- cor(merged_regions_count$urbanization_level, 
     merged_regions_count$testsPer100k, method = "spearman")
 
 ggplot(merged_regions_count, aes(x = urbanization_level, y = testsPer100k)) +
   geom_point() +
   geom_smooth(method = "lm") +
   labs(title= "Urbanization vs test per 100k citizens per Województwo",
        subtitle = paste("Spearman Correlation =", round( correlation_urb_test100k, 2)),
 x = "Urbanization level", y = "Tests per 100k people") + theme_tufte()

 pkdData_mod_WOJ
 
 
 
 
 map_testper100k <- ggplot() +
   geom_sf( data, aes(fill = testsPer100k)) + 
   geom_sf_label(data,
                 aes(label = paste0(round(testsPer100k, 0))) , size = 3, fontface = "bold")+
   scale_fill_gradient(low = "white", high = "green", na.value = "blue") + 
   labs(title = "Number of tests per 100k citizens") +
   theme_void() +
   theme(legend.position = "none")
 map_testper100k 
 
 
 
