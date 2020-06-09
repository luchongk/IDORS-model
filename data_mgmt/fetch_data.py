import krippendorff, csv, sys
import pprint as pp
import numpy as np
import paramiko
import base64
import json

def determineHateLabel(tweet_id, tweet_text, counts, desambigEntries, ambiguousJson):                
    label = None
    if abs(counts[0] - counts[1]) <= 1:
        entry = None
        if tweet_text.strip() in desambigEntries:
            entry = desambigEntries[tweet_text.strip()]

        if entry and not entry['written'] and ('1' in entry['labels']):
            label = 1
        elif entry and not entry['written'] and ('2' in entry['labels']):
            label = 0
        else:
            toWrite = json.dumps({"tweet_id": tweet_id, "text": tweet_text.replace("\\n", "\n")})
            ambiguousJson.write(toWrite + "\n")
    
    if not label:
        if counts[0] == counts[1]:
            print(line['text'])
            return -1
        
        label = counts.index(max(counts))

    return label

def determineOffensiveLabel(tweet_id, tweet_text, counts, desambigEntries):                
    label = None
    if abs(counts[0] - counts[1]) <= 1:
        entry = None
        if tweet_text.strip() in desambigEntries:
            entry = desambigEntries[tweet_text.strip()]

        if entry and not entry['written'] and ('3' in entry['labels']):
            label = 1
        elif entry and not entry['written'] and ('4' in entry['labels']):
            label = 0
    
    if not label:
        if counts[0] == counts[1]:
            return "A"
        
        label = counts.index(max(counts))

    return label

def determineHateTypeLabel(tweet_id, tweet_text, counts, desambigEntries):                
    label = None

    isAmbiguous = len(set(counts.values())) <= 1

    if isAmbiguous:
        entry = None
        if tweet_text.strip() in desambigEntries:
            entry = desambigEntries[tweet_text.strip()]

        if entry and not entry['written']:
            if '5' in entry['labels']:
                label = 'racism'
            elif '6' in entry['labels']:
                label = 'misoginy'
            elif '7' in entry['labels']:
                label = 'political'
            elif '8' in entry['labels']:
                label = 'homophobia'
            else:
                label = "other"
    
    if not label:
        if isAmbiguous:
            return "A"
        
        maxIndex = list(counts.values()).index(max(counts.values()))
        label = list(counts.keys())[maxIndex]

    return label


dataFileName = sys.argv[1]

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('odioelodio.com', username=f'{sys.argv[2]}')
stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{sys.argv[3]} -e "use pgodio;select v.tweet_id, t.text, count(v.is_hateful) as count, v.is_hateful from tweets t right join votesIsHateful v on t.id = v.tweet_id group by v.tweet_id, v.is_hateful;"')
with open(dataFileName, "w") as tsvFile:
    tsvFile.writelines(stdout)

stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{sys.argv[3]} -e "use pgodio;select v.tweet_id, t.text, count(v.is_offensive) as count, v.is_offensive from tweets t right join votesIsHateful v on t.id = v.tweet_id group by v.tweet_id, v.is_offensive;"')
offensiveReader = csv.DictReader(list(stdout), delimiter="\t")

stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{sys.argv[3]} -e "use pgodio;select v.tweet_id, t.text, count(v.hate_type) as count, v.hate_type from tweets t right join votesHateType v on t.id = v.tweet_id group by v.tweet_id, v.hate_type;"')
hateTypesReader = csv.DictReader(list(stdout), delimiter="\t")
client.close()

desambigEntries = {}
with open('db_data/desambiguados.csv') as desambiguados:
    csvReader = csv.DictReader(desambiguados)
    for line in csvReader:
        if line['text'] not in desambigEntries:
            desambigEntries[line['text']] = {"written": False, "labels": list()}
        desambigEntries[line['text']]['labels'].append(line['label'])

valueCounts = {'hate': {}, 'offensive': {}, 'hateTypes': {}}
ids = set()

countAmbiguous = 0
with open(dataFileName) as tsvFile:
    tsvReader = csv.DictReader(tsvFile, delimiter="\t")
    for line in tsvReader:
        #if line['tweet_id'] == "1034785678893232128":
        #    b = 0
        if line['tweet_id'] not in valueCounts['hate']:
            valueCounts['hate'][line['tweet_id']] = [0,0]
        valueCounts['hate'][line['tweet_id']][int(line['is_hateful'])] = int(line['count'])

    for line in offensiveReader:
        if line['tweet_id'] not in valueCounts['offensive']:
            valueCounts['offensive'][line['tweet_id']] = [0,0]
        valueCounts['offensive'][line['tweet_id']][int(line['is_offensive'])] = int(line['count'])

    for line in hateTypesReader:
        if line['tweet_id'] not in valueCounts['hateTypes']:
            valueCounts['hateTypes'][line['tweet_id']] = {'racism': 0, 'political': 0, 'homophobia': 0, 'misoginy': 0, 'other': 0}
        valueCounts['hateTypes'][line['tweet_id']][line['hate_type']] = int(line['count'])
    
    tsvFile.seek(0)
    next(tsvFile)
    
    with open("db_data/ambiguous.json", "w") as ambiguousJson:
        with open('datasets/idors.tsv', 'w') as idorsFile:
            fieldNames = ['id',	'text', 'HS', 'OF', 'HT']
            writer = csv.DictWriter(idorsFile, fieldnames=fieldNames, delimiter="\t")
            writer.writeheader()
            
            for line in tsvReader:
                if line['tweet_id'] in ids:
                    continue
                
                ids.add(line['tweet_id'])
                labelHate = determineHateLabel(line['tweet_id'], line['text'], valueCounts['hate'][line['tweet_id']], desambigEntries, ambiguousJson)
                if labelHate == -1:
                    countAmbiguous += 1
                    continue

                labelOffensive = None
                if line['tweet_id'] in valueCounts['offensive']:
                    labelOffensive = determineOffensiveLabel(line['tweet_id'], line['text'], valueCounts['offensive'][line['tweet_id']], desambigEntries)
                else:
                    labelOffensive = "N/A"

                if labelHate == 1:
                    labelHateType = None
                    if line['tweet_id'] in valueCounts['hateTypes']:
                        labelHateType = determineHateTypeLabel(line['tweet_id'], line['text'], valueCounts['hateTypes'][line['tweet_id']], desambigEntries)
                    else:
                        labelHateType = "N/A"
                else:
                    labelHateType = "N/A"

                writer.writerow({'id': line['tweet_id'], 'text': line['text'], 'HS': str(labelHate), 'OF': str(labelOffensive), 'HT': labelHateType})
            

print("\nAmbiguous:", countAmbiguous)
print("\nKrippendorff:", krippendorff.alpha(value_counts=np.array(list(valueCounts['hate'].values())), level_of_measurement='nominal'))