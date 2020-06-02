import krippendorff, csv, sys
import pprint as pp
import numpy as np
import paramiko
import base64
import json

dataFileName = sys.argv[1]

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('odioelodio.com', username=f'{sys.argv[2]}')
stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{sys.argv[3]} -e "use pgodio;select v.tweet_id, t.text, count(v.is_hateful) as count, v.is_hateful from tweets t right join votesIsHateful v on t.id = v.tweet_id group by v.tweet_id, v.is_hateful;"')
with open(dataFileName, "w") as tsvFile:
    tsvFile.writelines(stdout)
client.close()

desambigEntries = {}
with open('db_data/desambiguados.csv') as desambiguados:
    csvReader = csv.DictReader(desambiguados)
    for line in csvReader:
        if line['text'] not in desambigEntries:
            desambigEntries[line['text']] = {"written": False, "labels": list()}
        desambigEntries[line['text']]['labels'].append(line['label'])

valueCounts = {}
ids = set()

countAmbiguous = 0
with open(dataFileName) as tsvFile:
    tsvReader = csv.DictReader(tsvFile, delimiter="\t")
    for line in tsvReader:
        #if line['tweet_id'] == "1034785678893232128":
        #    b = 0
        if line['tweet_id'] not in valueCounts:
            valueCounts[line['tweet_id']] = [0,0]
        valueCounts[line['tweet_id']][int(line['is_hateful'])] = int(line['count'])
    
    tsvFile.seek(0)
    next(tsvFile)
    
    with open("db_data/ambiguous.json", "w") as ambiguousJson:
        with open('datasets/idors.tsv', 'w') as idorsFile:
            fieldNames = ['id',	'text', 'HS']
            writer = csv.DictWriter(idorsFile, fieldnames=fieldNames, delimiter="\t")
            writer.writeheader()
            
            for line in tsvReader:
                if line['tweet_id'] in ids:
                    continue
                
                ids.add(line['tweet_id'])
                counts = valueCounts[line['tweet_id']]
                
                label = None
                if abs(counts[0] - counts[1]) <= 1:
                    entry = None
                    if line['text'].strip() in desambigEntries:
                        entry = desambigEntries[line['text'].strip()]

                    if entry and not entry['written'] and ('1' in entry['labels']):
                        label = 1
                    elif entry and not entry['written'] and ('2' in entry['labels']):
                        label = 0
                    else:
                        toWrite = json.dumps({"tweet_id": line['tweet_id'], "text": line['text'].replace("\\n", "\n")})
                        ambiguousJson.write(toWrite + "\n")
                        countAmbiguous += 1
                
                if counts[0] == counts[1]:
                    print(line['text'])
                    continue
                
                if not label:
                    label = counts.index(max(counts))
                writer.writerow({'id': line['tweet_id'], 'text': line['text'], 'HS': str(label)})
            

print("\nAmbiguous:", countAmbiguous)
print("\nKrippendorff:", krippendorff.alpha(value_counts=np.array(list(valueCounts.values())), level_of_measurement='nominal'))



