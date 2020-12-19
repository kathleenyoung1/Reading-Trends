# Reading-Trends

I started reading Marcus Aurelius last spring. I'd never had much interest in philosophy, but I was desperate to find a way of managing the stress of isolating in my apartment while the pandemic raged through New York City. I wasn't sleeping much, and I was open to any wisdom that might help. So I gave philosophy a try.  

I didn't make it to a state of stoic acceptance. It turns out I don't like philosophy books. But I did start to wonder whether there might be others who were reacting to the pandemic by reading in an entirely new way. What do we read in a pandemic? Philosophy? Light entertainment? Cookbooks? Divorce manuals? I was very curious, so I started this project.

I wanted to understand whether the pandemic had impacted the books that people choose to read, and I hoped that doing so would give me some insight into how others were experiencing the pandemic. In order to create a dataset that captured what people were reading, I built a distributed webscraper which pulled data from Goodreads.com. In September, I decided to use the project as my term project for my Introduction to Data Science class at the NYU Courant Center for Data Science. Three of my amazing classmates, Preston Harry, Sabrina Sheu, and Kathleen Young, became my teammates. Together, we managed the data creation process and then used machine learning models to measure and interpret what people were reading.

If you'd like to take a look, I would recommend starting with the final paper file in the documentation folder. It contains a full write-up of our methodology and findings. The document style is very specific to how our professor likes papers; at some point, I would like to create a version for a more general audience. Other key files are as follows:
   - The distributed_data_collection folder contains the code for the distributed scraper which created the Goodreads dataset. Additional documentation for the scraper can be found in the documentation folder under scraper_one_page_diagram.
   - The url_analysis folder contains two analyses of scraped data, designed to understand the relationship between URL and publication date. These analyses allowed us to more accurately scrape data only from the periods of interest.
   - The subject_matching folder contains the code with which we processed and vectorized subject & genre information from Open Library.
   - The modeling folder contains the models that we built based on the data: the base_model file represents the first baseline model that we built. The monthly_2020_models represent a more nuanced approach to the problem: the feature_selection file shows how we decided which features to include, and the other monthly_2020 files capture versions of the models built with different feature sets. The aggregator file supports the modeling process by cleaning and aggregating data sources.

Language: Python

Packages: Beautiful Soup (bs4), Bottle, Json, Matplotlib, Natural Language Toolkit (nltk), Numpy, Pandas, Regular Expressions, Requests, Seaborn, Sklearn.
