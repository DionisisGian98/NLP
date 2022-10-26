from bs4 import BeautifulSoup
import requests
import pandas as pd
import urllib.request,sys,time
from sqlalchemy import create_engine

def get_url(url):
    page = requests.get(url)

    try:
         # this might throw an exception if something goes wrong.
         page = requests.get(url) 
         # this describes what to do if an exception is thrown 
    except Exception as e:    
    
        # get the exception information
        error_type, error_obj, error_info = sys.exc_info()      
    
        #print the link that cause the problem
        print ('ERROR FOR LINK:',URL)
    
        #print error info and line that threw the exception                          
        print (error_type, 'Line:', error_info.tb_lineno)

    
    return page

url1 = 'https://www.buzzfeednews.com/section/world'
page1 = get_url(url1)
print("the response for page1 is:",page1)

def return_summary(page,cls,cls_name):
    c = page.content
    # Set as Beautiful Soup Object
    soup = BeautifulSoup(c,'html.parser')# Go to the section of interest
    summary = soup.find_all(cls,{'class':cls_name})
    #print(len(summary))
    return summary
summary = return_summary(page1,'a','newsblock-story-card__link xs-flex')

#collect titles and articles
articles = []
titles =[]
for j in range(0,len(summary)):
    page = requests.get(summary[j]['href'])
    temp_url = get_url(summary[j]['href'])
    temp_summary = return_summary(page,'div','subbuzz subbuzz-text xs-mb4 xs-relative')
    temp_link = summary[j]['href']
    title = summary[j].get_text()
    titles.append(title)
    temp_list = []
    
    for i in range(0,len(temp_summary)):
        temp = temp_summary[i].find('p').get_text()
        temp1 = ''.join(temp)
        temp_list.append(temp1)
    articles.append(temp_list)
print(articles)
print(titles)

#article trnsformation
new_article = []
for i in articles:
    i = ' '.join(map(str, i))
    new_article.append(i)
    #print(i)
print(new_article)

#articles to db
dict = {'Title': titles, 'Article': new_article}
df = pd.DataFrame(dict)
engine = create_engine(r'sqlite:///C:\Users\gian5\OneDrive\Desktop\NLP\save_pandas.db', echo=True)
df.to_sql('articles_info', con=engine,if_exists='append')
