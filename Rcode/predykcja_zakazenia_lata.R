library(ggplot2)
library(ggthemes)
library(gridExtra)
library(dplyr)
library(rethinking)

th <- theme_tufte(base_size = 14) + theme(plot.title.position = "plot")
plt_pos <- theme(plot.title.position = "plot")


priors_check <- function(model, data, count){
  
  priorsYU <- extract.prior(model)
  priorLinkYU <- link(model,post = priorsYU)
  
  
  str(priorLinkYU)
  priorMeansYU <- apply(priorLinkYU, 2, mean)
  priorHPDIYU <- data.frame(t(apply(priorLinkYU, 2, HPDI)))
  names(priorHPDIYU) <- c("low", "high") 
  
  len_numb <- nrow(data)
  
  return(
    ggplot()+geom_point(data = data, aes(x = 1:len_numb, y = count))+th+
      geom_line(aes(x = 1:len_numb, y = priorMeansYU))+
      geom_ribbon(aes(x = 1:len_numb, ymin = priorHPDIYU$low, ymax = priorHPDIYU$high),
                  alpha = .5, fill = "skyblue")
  )
}


posteriors_check <- function(model, data, count){

posteriorLinkYU <- link(model)

posteriorMeansYU <- apply(posteriorLinkYU, 2, mean)
posteriorHPDIYU <- data.frame(t(apply(posteriorLinkYU, 2, HPDI)))
names(posteriorHPDIYU) <- c("low", "high") 

len_numb <- nrow(data)

return(
ggplot()+geom_point(data = data, aes(x = 1:len_numb, y = count))+th+
  geom_line(aes(x = 1:len_numb, y = posteriorMeansYU))+
  geom_ribbon(aes(x = 1:len_numb, 
                  ymin = posteriorHPDIYU$low,
                  ymax = posteriorHPDIYU$high),
              alpha = .5, fill = "skyblue")
)
}


# -------------------------------------------------------------------------

HIVdata <- read.csv("HIV19992017PZH.csv")


head(HIVdata)

counts <- HIVdata %>% 
  group_by(rok) %>% 
  summarize(count = n())

years_empty <- data.frame(rok= c(2018:2025),
                          count= rep(NA, 8))

counts_joint <- rbind(counts, years_empty)
counts_joint$yearsStand <- as.numeric(standardize(counts_joint$rok))

counts_mod <- counts_joint %>% filter(rok <= 2017)


set.seed(123)
years_model <- ulam(
  alist(
    count ~ dnorm( mu , sigma ) ,
    mu <- base + m  * yearsStand,
    m ~ dnorm(200, 200),
    base ~ dnorm(500, 200),
    sigma ~ dunif( 0 , 225)
  ) , data=counts_mod, log_lik = TRUE )

precis(years_model)

# prior check

hell <- priors_check(years_model, counts_mod, counts_mod$count)

hell_post <- posteriors_check(years_model, counts_mod, counts_mod$count)

dens(priorsYU$base)
dens(priorsYU$m)
dens(priorsYU$sigma)


predict_year_counts <- sim(years_model,  counts_joint)


predict_year_counts_cols <- data.frame(t(apply(predict_year_counts, 2, HPDI)))

priorMeansYU2222 <-  mean(predict_year_counts_cols$low, predict_year_counts_cols$high)

names(predict_year_counts_cols) <- c("low_pred", "high_pred")

predict_year_counts_cols <- predict_year_counts_cols %>% mutate(mean_val= (
  (predict_year_counts_cols$low + predict_year_counts_cols$high)/ 2))


counts_joint_filled <- cbind(counts_joint, predict_year_counts_cols)



# awesome work, try to repeat that with Exponential Model
ggplot(counts_joint_filled)+
  geom_line(aes(x = rok, y = mean_val))+
  geom_ribbon(aes(x = rok, ymin = low, ymax = high),
              alpha = .5, fill = "gray")+ theme_fivethirtyeight()+
  plt_pos+ scale_y_continuous(breaks = seq(0, 2000, by= 200))+
  geom_col(aes(x = rok, y = count), fill= "steelblue", width = 0.4)+
  scale_x_continuous(breaks = 1999:2025)+ theme(
    axis.text.x = element_text(angle = 45))+
  labs(title= "Prediction of new HIV positive diagnosis",
       subtitle= "HIV prognosis, Poland 2018-2025")




























