from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import csv

print(tf.__version__)

with open('datasets/dev_es.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    for row in [r['text'] for r in reader]:
        print(row)