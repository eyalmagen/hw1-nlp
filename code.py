import csv
import math
import os
from pandas import DataFrame
import numpy

SPLITTER = "    "
CODE_PATH = os.path.abspath(__name__)
FOLDER_PATH = os.path.dirname(os.path.dirname(CODE_PATH))
DATA_SETS_PATH = os.path.join(FOLDER_PATH, u"data")


def create_counter(text, n):
    counter = {}
    index = 0
    while index < len(text) - n:
        gram = text[index: index + n]
        if gram in counter:
            counter[gram] += 1
        else:
            counter[gram] = 1
        index += 1
    return counter


def lm(corpus_file):
    text = corpus_file
    lms = {}
    counter_1 = create_counter(text, 1)

    for key, val in counter_1.items():
        lms[key] = float(val) / len(text)

    for n in range(2, 4):
        counter = create_counter(text, n)
        for gram in counter:
            log_p = float(counter[gram]) / counter_1[gram[-1]]
            lms[gram] = log_p
    return lms


def calc_perplexity(testset, model, weights):
    w1, w2, w3 = weights
    perplexity = 0
    N = len(testset) - 3
    index = 0
    while index < len(testset) - 3:
        gram = testset[index: index + 3]
        try:
            p1 = float(w1 * (model[gram]))
            p2 = float(w2 * (model[gram[:-2]]))
            p3 = float(w3 * (model[gram[:-1]]))
        except:
            # bonus done.
            p1 = 0.01
            p2 = p3 = 0

        p_interpulation = p1 + p2 + p3
        perplexity += math.log((1 / p_interpulation), 2)
        index += 1

    perplexity = 2 ** (perplexity / float(N))
    return round(perplexity, 3)


def eval(test, model, weights):
    perplexity = calc_perplexity(test, model, weights)
    return perplexity


class LanguageDataSet():
    def __init__(self, name, test, train):
        self.name = name.replace(".csv", "")
        self.test = test
        self.train = train

    @classmethod
    def prepare(cls, path):
        lines = []
        with open(path, "rb") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lines.append("{}{}{}".format("\x00", row['tweet_text'], "\x01"))
        numpy.random.shuffle(lines)
        ten_percent = len(lines) / 10
        return cls(os.path.basename(path), lines[:ten_percent], lines[ten_percent:])


def handle_train_with_others(train_db, data_files, aggregate):
    model = lm("".join(train_db.train))
    for test_db in data_files:
        perplexity = eval("".join(test_db.test), model, [0.4, 0.3, 0.3])
        aggregate[train_db.name].append(perplexity)


def calc_table(base):
    data_files = [LanguageDataSet.prepare(os.path.join(base, data_set))
                  for data_set in os.listdir(base)]
    headers = [file.name for file in data_files]
    aggregate = {i: [] for i in headers}
    for train_db in data_files:
        handle_train_with_others(train_db, data_files, aggregate)

    data = {
        ' ': headers,
        }
    aggregate.update(data)
    columns = [' ']
    columns.extend(headers)
    df = DataFrame(aggregate, columns=columns)

    print(df)


def main():
    calc_table(DATA_SETS_PATH)


if __name__ == "__main__":
    main()
