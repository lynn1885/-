import os

labels = []
with open('event_schema.json', 'r', encoding='utf-8') as fp:
    event_schema_data = fp.read().strip()
    for line in event_schema_data.split('\n'):
        line = eval(line)
        labels.append(line['event_type'])


with open('../final_data/labels.txt', 'w', encoding='utf-8') as fp:
    for label in labels:
        fp.write(label + '\n')
