import sys, csv
from pprint import pprint

fileName = sys.argv[1]

confusion = [
    [0,0,0],
    [0,0,0]
]

hateTypes = {'racism': 0, 'misoginy': 0, 'political': 0, 'homophobia': 0, 'other': 0, 'A': 0, 'N/A': 0}

countNoData = 0

with open(fileName) as tsvFile:
    tsvReader = csv.DictReader(tsvFile, delimiter='\t')
    for line in tsvReader:
        hateLabel = int(line['HS'])
        offensiveLabel = line['OF']
        for i in range(2):
            if hateLabel == i:
                if offensiveLabel == 'A':
                    confusion[i][2] += 1
                elif offensiveLabel == 'N/A':
                    countNoData += 1
                    print(line['id'])
                elif int(offensiveLabel) == 1:
                    confusion[i][1] += 1
                else:
                    confusion[i][0] += 1
        
        hateTypes[line['HT']] += 1

hateTypes['N/A'] -= sum(confusion[0])

print(confusion[0])
print(confusion[1])
print("No data: ", countNoData)

pprint(hateTypes)