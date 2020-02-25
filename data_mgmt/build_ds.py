import krippendorff, csv, sys
import pprint as pp
import numpy as np

dataFileName = sys.argv[1]
valueCounts = {}
ids = set()

with open(dataFileName) as tsvFile:
    tsvReader = csv.DictReader(tsvFile, delimiter="\t")
    for line in tsvReader:
        if line['tweet_id'] not in valueCounts:
            valueCounts[line['tweet_id']] = [0,0]
        valueCounts[line['tweet_id']][int(line['is_hateful'])] = int(line['count(v.is_hateful)'])
    tsvFile.seek(0)
    next(tsvFile)
    with open('datasets/idors.tsv', 'w') as idorsFile:
        fieldNames = ['id',	'text', 'HS']
        writer = csv.DictWriter(idorsFile, fieldnames=fieldNames, delimiter="\t")
        writer.writeheader()
        for line in tsvReader:
            if line['tweet_id'] in ids:
                continue
            ids.add(line['tweet_id'])
            counts = valueCounts[line['tweet_id']]
            if counts[0] == counts[1]:
                print(line['text'])
                continue
            label = counts.index(max(counts))
            writer.writerow({'id': line['tweet_id'], 'text': line['text'], 'HS': str(label)})
            

print("\nNominal:", krippendorff.alpha(value_counts=np.array(list(valueCounts.values())), level_of_measurement='nominal'))
# pp.pprint(list(filter(lambda v : abs(v[0] - v[1]) > 1, valueCounts)))



