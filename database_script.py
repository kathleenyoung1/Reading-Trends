#import libraries
import pandas as pd

#Classes

class Database():

    def __init__(self, file_name):
        self.file_name = "databases/" + file_name + ".csv"
        self.df = pd.read_csv(self.file_name)
        self.df.dropna(inplace = True)
        self.df.drop_duplicates(inplace = True)

        self.df.reset_index(inplace = True, drop = True)

    def get_df(self):

        return self.df 

class Review_Database(Database):

    def __init__(self, file_name):
        super().__init__(file_name)

        self.df = self.df[self.df.is_URL_valid == True]

        self.df.sort_values(by = "ID", inplace = True)
        self.df.reset_index(inplace = True, drop = True)

    def drop_unrated(self):

        self.df = self.df[self.df.rating != "None"]
        self.df.reset_index(inplace = True, drop = True)

    def limit_dates(self, min_year, max_year):

        self.df["year"] = (pd.to_datetime(self.df["review_publication_date"]).apply(lambda date: date.year))
        self.df = self.df[(self.df.year >= min_year) & (self.df.year <= max_year)]
        self.df.drop(columns = ["year"], inplace = True)
        self.df.reset_index(inplace = True, drop = True)

    def generate_review_count_by_book(self):

        self.df_by_book = self.df.groupby(["book_id", "book_title"]).ID.count().reset_index()
        self.df_by_book.rename(columns = {"ID": "review_count"}, inplace = True)
        self.df_by_book.sort_values(by = "review_count", ascending=False, inplace = True)
        self.df_by_book.reset_index(inplace = True, drop = True)

    def generate_book_id_list(self):

        self.generate_review_count_by_book()
        book_id_list = self.df_by_book.book_id.unique()
        return book_id_list

class Book_Database(Database):

    def __init__(self, file_name):
        super().__init__(file_name)

        self.df.sort_values(by = "book_id", inplace = True)
        self.df.reset_index(inplace = True, drop = True)
        

## TESTING

review_database = Review_Database("review_data_sample")
review_database.drop_unrated()
review_database.limit_dates(2017, 2020)
#review_database.generate_review_count_by_book()
book_list = review_database.generate_book_id_list()

#print(book_list)
