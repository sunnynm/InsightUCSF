import pandas as pd
all1 = pd.read_csv('../Data/AllAbstracts.csv')

all1["Total"] = (all1["Title"] + all1["Abstract"]).str.lower()

mllist = ["machine learning","deep learning","artificial intelligence", " ai "]
cardlist = ['cardiology', 'cardiovascular','cardiac','heart']

import csv
counter = 0
with open('../Data/SubAbstracts.csv', mode='w', encoding='UTF-8', newline='') as file:
    ewrite = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ewrite.writerow(['Source', 'Identifier', 'Title', 'Abstract'])
    
    for index, row in all1.iterrows():
        ml = False
        card = False
        for i in mllist:
            if i in row["Total"]:
                ml = True
        
        for i in cardlist:
            if i in row["Total"]:
                card = True
        if card and ml:
            ewrite.writerow([row["Source"], row["Identifier"], row["Title"], row["Abstract"]])
            counter += 1
print(counter)