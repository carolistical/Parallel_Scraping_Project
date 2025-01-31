# Project For the American University Of Beirut - Library Dept.
- GOAL: streamlit APP where you can upload database (an excel sheet of data) with required column :'search term'
- URL associated with Search term will be scraped LEGALLY from website : "https://www.loc.gov".
- New excel sheet with added column of URL will be able to be downloaded from streamlit app once scraping is completed.
- Multiple Graphs are also generated in the APP such as % of URI's found Vs Not Found for associated search term. 

# Technical Process: 
- Scraping will be done through calling functions concurrently using ThreadPoolExecutor package.
- BeautifulSoup library used for parsing HTML and XML webpages and REGEX used to match with corresponding tags of the information required to be scraped. 
- Pandas and Matplotlib was used for data manipulation and creation of bar charts generated in the APP.
- Webscraping action was done using selenium driver using Chrome Web.
- Error logging was maintained and monitored by package "logging".
