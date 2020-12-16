import json
import csv

def action_str_process(actions):
    if not actions:
        return []
    actions = actions.split(';')
    ret = []
    for ac in actions:
        ac = ac.split(' ')
        n = int(ac[0][1:])  # class number
        start = float(ac[1]) # start time
        end = float(ac[2]) # end time
        ac = [n, start, end]
        ret.append(ac)
    return ret



class MyData(object):
    def __init__(self, subset, actions, duration):
        self.actions = actions
        self.duration = duration
        self.subset = subset

def csv2json(train_csv, test_csv):
    dic = {}
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # skip 1st line
            if i == 0 or row[11] != 'Yes': # not egocentric
                continue
            id = row[0]
            actions = action_str_process(row[9])
            duration = float(row[10])
            data = MyData('training', actions, duration)
            dic[id] = data.__dict__
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # skip 1st line
            if i == 0 or row[11] != 'Yes': # not egocentric
                continue
            id = row[0]
            actions = action_str_process(row[9])
            duration = float(row[10])
            data = MyData('testing', actions, duration)
            dic[id] = data.__dict__
    with open("charades_ego.json", 'w+') as f:
        json.dump(dic, f)

if __name__ == '__main__':

    # csv2json('../CharadesEgo/CharadesEgo_v1_test.csv', 'testing')
    csv2json(train_csv='../CharadesEgo/CharadesEgo_v1_train.csv', test_csv='../CharadesEgo/CharadesEgo_v1_test.csv')
