from dlx import DLX

# Function to generate the instance with labels, rows, columns, and label indices.
def genInstance(labels, rows):
    columns = []
    indicesL = {}
    for i in range(len(labels)):
        label = labels[i]
        indicesL[label] = i
        columns.append((label, 0))  # Initialize columns with label and 0.
    return labels, rows, columns, indicesL

# Function to solve the given instance using DLX algorithm.
def solveInstance(instance):
    labels, rows, columns, indicesL = instance
    dlxInstance = DLX(columns)
    indices = {}

    # Append each row to the DLX instance and store its hash in indices.
    for i, row in enumerate(rows):
        h = dlxInstance.appendRow(row, f'r{i}')
        indices[str(hash(tuple(sorted(row))))] = i

    solution = dlxInstance.solve()
    solutionList = list(solution)
    selectedRows = []

    # If no solution is found, return None.
    if not solutionList:
        return None

    # Process the solution to find the selected rows.
    for i in solutionList[0]:
        rowList = dlxInstance.getRowList(i)
        rowIndices = [indicesL[label] for label in rowList]
        rowIndex = indices[str(hash(tuple(sorted(rowIndices))))]
        selectedRows.append(rowIndex)

    return selectedRows

def printColumnsPerRow(instance, selectedRows):
    labels, rows, columns, indicesL = instance
    print('Covered columns per selected row:')

    for s in selectedRows:
        coveredColumns = []
        for z in rows[s]:
            c, _ = columns[z]
            coveredColumns.append(c)
        print(s, coveredColumns)

def printInstance(instance):
    labels, rows, columns, indicesL = instance
    print('Columns:')
    print(labels)
    print('Rows:')
    print(rows)
