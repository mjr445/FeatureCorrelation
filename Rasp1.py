import pandas as pd

file = open("livedata.txt")
data =[]
ColumnNames=[]
for line in file:
    if len(line)> 100 and not line.startswith(" ID"):  # removes all unnecessary lines from the data
        for word in line.split():
            data.append(word)
    elif line.startswith(" ID"):
        for word in line.split():
            ColumnNames.append(word)

#  The following line coverts the list to a nested list with each sublist having 26 elements
data = [data[i:i + 25] for i in range(0, len(data), 25)]
ColumnNames=ColumnNames[0:26]
#  The following 3 lines remove a repeat column header and combine it into one
ColumnNames.pop(17)
ColumnNames.pop(17)
ColumnNames.insert(17,"Sig-Strength")

df = pd.DataFrame(data, columns=ColumnNames)
print(df.shape)

df.to_csv("DataframeTest.txt", sep="\t")

run = True
while run:
    if file.readline() != "":
        if len(line) > 100 and not line.startswith(" ID"):
            df2 = pd.DataFrame(line.split(), columns=ColumnNames)
            df = df.append(df2)
            df.to_csv("DataframeTest")
        elif line.startswith(" 15"):
            run = False

file.close()
