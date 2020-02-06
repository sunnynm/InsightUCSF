from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import pandas as pd 
import time
start_time = time.time()
df = pd.read_csv("./MutliKeywordDistFULL.csv") 
alpha = df.loc[(df['ManRefVal'] >= .66)]
beta = df.loc[(df['ManRefVal'] >= .66) & (df['ManRefVal'] != 1)]
  
# iterate through the csv file 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',
                collocations=False,
                min_font_size = 10,
                regexp = '\w+(?:-\w+)*|\$[\d.]+|\S+').generate(" ".join(list(beta.Phrase.values))) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
#plt.show() 

plt.savefig('wordcloudFULL.png')
plt.savefig('wordcloudFULL.pdf')
print("--- %s seconds ---" % (time.time() - start_time))