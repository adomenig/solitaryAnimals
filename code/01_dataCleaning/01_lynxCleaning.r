################################################
############# Data Import Script ##############
##############################################
pkgs = c('tidyverse', 'lubridate')
sapply(pkgs, require, character = TRUE)


### Getting the home directory so we can work with relative paths
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Home directory must be supplied as argument")
}

home_dir <- args[1]
setwd(home_dir)

wd <- "./data/Complete_20231231"

### Importing the data and formatting --------------------------------------------------------------------------
###Modify this list with the collars that you want imported.
SNs =  list.files(wd, pattern = "*.csv", full.names = T) 

##Creates a blank list to fill with collar data
LYNX  = list()

##Loop that adds each collar to the list as its own dataframe
##Make sure to change the file suffix in the paste command to whatever it needs to be changed to.
for(n in 1:length(SNs)){
  LYNX[[substring(SNs[n], nchar(wd) + 2, nchar(wd) + 7)]] = with(read.csv(skip=22, SNs[n],
                                                                          header = TRUE, na.strings=c("",NA)),
                                                                 data.frame(Time = round_date(ymd_hms(GPS.Fix.Time), unit = "hour"),
                                                                            Long = GPS.Longitude, Lat = GPS.Latitude,
                                                                            Type = GPS.Fix.Attempt,
                                                                            Easting = GPS.UTM.Easting, Northing = GPS.UTM.Northing,
                                                                            Activity = c(Activity.Count[-1],NA), Mortality = c(Mortality[-1], 'NA'))) %>%
    #    mutate(CTN = substring(SNs[n], nchar(wd) + 2, nchar(wd) + 7)) %>%
    mutate(CTN = substring(basename(SNs[n]), 1, 6)) %>%
    #####Subsets for only resolved fixes
    filter(Type == "Resolved QFP" | Type == "Uncertain QFP")  %>%
    #####Adds an ID column to the front of the dataframe
    dplyr::select(CTN, everything()) %>%
    #####Adds TimeDiff column used for package BBMM
    mutate(TimeDiff = c(NA, round(diff(as.numeric(Time))/3600)))  %>%
    mutate(Mortality = as.factor(ifelse(Mortality == "Yes", "2",
                                        ifelse(Mortality == "No", "1", NA))))
  
}  


## Removing data after a lynx mortality from the dataset
LYNX = LYNX %>%  map(function(x){
  ##Determining when the first "Yes" occurs in the dataset
  mort.date = ifelse(
    (sum(x$Mortality[-c(1:10)] == "2", na.rm = T)) == 0, nrow(x),
    (min(which(x$Mortality[-c(1:10)] == "2"))))  
  
  ##Removing all data after mortality
  slice(x, 1:(mort.date+40))
})


##Creating the data frames and adding in refuge, sex, and age covariates
CTN.key = read.csv(paste(wd, "/ctn_key/ctn_refuges.csv", sep = "")) %>%
  mutate_at('CTN', as.character)
lynx.df = bind_rows(LYNX) %>%
  ##Merging the key with the dataframe to add these covariate columns
  dplyr::left_join(CTN.key[, c("ID", "CTN", "Refuge", "Sex", "Age", "Mass",
                               "Year_Collared", "Multiple_Cats")], by = "CTN", relationship = "many-to-many") %>%
  ## Filtering out CTN's that have been on multiple lynx
  #dplyr::filter(Multiple_Cats == "N") %>%
  ## Moving ID column to the front
  dplyr::select(ID, everything())

## Dealing with lynx that have multiple collars
c.700528 = filter(lynx.df, CTN == "700528") %>% distinct(Time, .keep_all = TRUE) %>% mutate(ID = ifelse(Time < "2019-09-26", "WSM022", "WSM107"))
c.704787 = filter(lynx.df, CTN == "704787") %>% distinct(Time, .keep_all = TRUE) %>% mutate(ID = ifelse(Time < "2019-11-14", "WSM012", "WSM108")) %>%
  mutate(Sex = ifelse(Time < "2019-11-14", "M", "F")) %>% filter(Lat < 68) %>% filter(Lat > 67)
c.704796 = filter(lynx.df, CTN == "704796") %>% distinct(Time, .keep_all = TRUE) %>% mutate(ID = ifelse(Time < "2019-04-10", "WSM010", "WSM022"))
c.704794 = filter(lynx.df, CTN == "704794") %>% distinct(Time, .keep_all = TRUE) %>% mutate(ID = ifelse(Time < "2020-03-01", "WSM019", "WSM105"))


lynx.df = filter(lynx.df, CTN != "700528" & CTN != "704787" & CTN != "704796" & CTN != "704794") %>%
  rbind(., c.700528, c.704787, c.704796, c.704794) %>%
  filter(is.na(ID) == FALSE) %>% filter(Lat >= 40)

rm(LYNX)
rm(CTN.key)
rm(c.700528, c.704787, c.704796, c.704794)

## Adulting out Juveniles that survive to the following year
## Create two dataframe for adults and juveniles
lynx.df.a = lynx.df %>% filter(Age == "Adult")
lynx.df.j = lynx.df %>% filter(Age == "Juvenille")
## Exract Juvenille ID's
juvs = unique(lynx.df.j$ID)

## Replace age after 300 days with adult category
for(i in juvs){
  ## Extract capture date
  cpd = range(filter(lynx.df.j, ID == i)$Time)[1]
  ## Find total range of data from that individual
  end = range(filter(lynx.df.j, ID == i)$Time)[2] 
  ## Skip lynx that didn't survive to adulthood
  if(end >= paste0(year(cpd)+1,'-03-31')){next}
  ## Replace age category with Adult after March 31 of next year days
  lynx.df.j[lynx.df.j$ID == i,]$Age = ifelse(lynx.df.j[lynx.df.j$ID == i,]$Time >= paste0(year(cpd)+1,'-03-31'),
                                             "Adult", "Juvenille")
}


## Rebind lynx.df dataframe from separate dataframes
lynx.df = rbind(lynx.df.a, lynx.df.j)

## Remove separate dataframes
rm(lynx.df.a)
rm(lynx.df.j)

### Remove any repeat observations in the data and filter out telonics headquarters locations
lynx.df = distinct(lynx.df, ID, Time, .keep_all = TRUE)
output_path <- "./data/processed"
output_file <- file.path(output_path, "lynx_initial_clean.csv")

# create the directory if it doesn't exist
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
}

# save the CSV
lynx.df$Time <- format(as.POSIXct(lynx.df$Time, tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
write.csv(lynx.df, file = output_file, row.names = FALSE, quote = FALSE)
