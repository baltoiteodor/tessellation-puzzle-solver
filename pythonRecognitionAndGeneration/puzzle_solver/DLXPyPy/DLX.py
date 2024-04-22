from dlx import DLX


# Taken from here:
# https://gist.github.com/marekyggdrasil/a8e63be8e34e000f2507bdb5e0755dda

def genInstance(labels, rows) :
    columns = []
    indices_l = {}
    for i in range(len(labels)) :
        label = labels[i]
        indices_l[label] = i
        columns.append(tuple([label,0]))
    return labels, rows, columns, indices_l

def solveInstance(instance) :
    labels, rows, columns, indices_l = instance
    instance = DLX(columns)
    indices = {}
    for l, i in zip(rows, range(len(rows))) :
        h = instance.appendRow(l, 'r'+str(i))
        indices[str(hash(tuple(sorted(l))))] = i
    sol = instance.solve()
    lst = list(sol)
    selected = []
    if len(lst) == 0:
        return None
    for i in lst[0]:
        l = instance.getRowList(i)
        l2 = [indices_l[label] for label in l]
        idx = indices[str(hash(tuple(sorted(l2))))]
        selected.append(idx)
    return selected

def printColumnsPerRow(instance, selected) :
    labels, rows, columns, indices_l = instance
    print('covered columns per selected row')
    for s in selected :
        A = []
        for z in rows[s] :
            c, _ = columns[z]
            A.append(c)
        print(s, A)

def printInstance(instance) :
    labels, rows, columns, indices_l = instance
    print('columns')
    print(labels)
    print('rows')
    print(rows)

