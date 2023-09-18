library(readxl)


# Set the directory where your Excel files are located
setwd("C:/Users/nikod/Documents/RProjects/hivproject/pkdData/first")


# Get a list of all the .xlsx files in the folder
files <- list.files(pattern = "\\.xlsx$")

# Create an empty data frame to store the combined data
combined_data <- data.frame()

# Loop through each file and read its data into a data frame
for (file in files) {
  data <- read_excel(file)
  combined_data <- rbind(combined_data, data)
}



# 2022 file has different number of columns,
# therefore it has to be dealt differently
setwd("C:/Users/nikod/Documents/RProjects/hivproject/pkdData")
df2022 <- read_excel("2022.xlsx")


colnames(combined_data)
colnames(df2022)

common_cols <- intersect(names(combined_data), names(df2022))
not_common_cols <- setdiff(names(combined_data), names(df2022))
not_common_cols2 <- setdiff(names(df2022), names(combined_data))
# Bind the common columns together into a single data frame
df_all <- bind_rows(combined_data[, common_cols], df2022[, common_cols])

colnames(df_all)



names(df_all) <- gsub("\\s+", "_", names(df_all))


write.csv(df_all, file = "PKDjoint.csv", row.names = FALSE)

##############

# Define the table as a string
table_string <- "PKDmiasto, dniwTyg, godzPerDzien, pora, wojewodztwo
Białystok, 3, 2.5, po_15, podlaskie
Bydgoszcz, 2, 3.5, po_15, kujawsko-pomorskie
Chorzów, 5, 2.2, przed_15, śląskie
Częstochowa, 1, 3, po_15, śląskie
Gdańsk, 3, 4, po_15, pomorskie
Gdynia, 2, 4, po_15|przed_15, pomorskie
Jelenia_góra, 2, 2.5, po_15, dolnośląskie
Kraków, 5, 3, po_15, małopolskie
Kielce, 2, 2.5, po_15|przed_15, świętokrzyskie
Koszalin, 1, 4, po_15, zachodniopomorskie
Lublin, 3, 3, po_15, lubelskie
Łódź, 3, 3, po_15, łódzkie
Olsztyn, 2, 3, po_15, warmińsko-mazurskie
Opole, 2, 3, po_15, opolskie
Płock, 2, 3, po_15, mazowieckie
Poznań, 4, 4, po_15, wielkopolskie
Rzeszów, 2, 3, po_15|przed_15, podkarpackie
Szczecin, 3, 3, po_15, zachodniopomorskie
Toruń, 2, 3, po_15, kujawsko-pomorskie
Wałbrzych, 1, 4, po_15, dolnośląskie
Warszawa1, 5, 4, po_15, mazowieckie
Warszawa2, 5, 10, przed_15|po_15, mazowieckie
Warszawa3, 5, 3, po_15, mazowieckie
Warszawa4, 5, 4, po_15, mazowieckie
Wrocław, 5, 4, po_15, dolnośląskie
Zgorzelec, 1, 2, przed_15, dolnośląskie
Zielona_góra, 2, 2, po_15, lubuskie"

# Convert the table string to a data frame
df_PKD <- read.csv(text = table_string, stringsAsFactors = FALSE)

# Print the data frame
df_PKD



write.csv(df_PKD, file = "dataAboutPKD.csv", row.names = FALSE)


















