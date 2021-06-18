import numpy as np

# Softmax algorithm
def softmax(mtx:np.ndarray):
    newMtx = np.empty((0, np.shape(mtx)[1]), float)
    for i in range(len(mtx)):
        newMtx = np.vstack((newMtx, np.divide(np.exp(mtx[i]), np.sum(np.exp(mtx[i])))))
    return newMtx

# Import CSV data
def readCSV(filePath:str):
    csv = np.genfromtxt(filePath, delimiter=";", dtype=float)
    csv = csv[:,~np.all(np.isnan(csv), axis=0)]
    return softmax(csv)

# Beam Search Calculation
def calcBeamSearch(mtx:np.ndarray, w:int, c:np.ndarray):
    indexes = np.empty((0,w), int) # Top [beamWidth] biggest possibilities's indexes of each rows in matrix
    
    # Get top [beamWidth] values' indexes
    for i in range(len(mtx)):
        rawIndexes = np.argsort(mtx[i])[:-w-1:-1]
        indexes = np.vstack((indexes, rawIndexes))
    paths = indexes[0].reshape(w,1) # Result paths holder
    
    # Get the indexes of top [beamWidth] biggest probability
    for i in range(1, len(indexes)):
        # np.argsort always increase sort. Therefore, it needs to be reversed by using [:-beamWidth-1:-1]
        topIndexes = np.argsort(np.concatenate([mtx[i][indexes[i]] * x for x in mtx[i-1][paths.T[-1]]]), kind="mergesort")[:-w-1:-1]
        nxt = topIndexes % w # Next indexes position
        prev = (topIndexes - nxt)//w # Previous indexes position
        paths = np.concatenate((paths[prev], indexes[i][nxt].reshape(w,1)), axis=1)
    return [''.join(res) for res in c[paths]]

def main():
    classes = np.append(np.array(list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")), "") # Classes
    beamWidth = 3 # Beam width
    filePath = "BeamSearch/assets/data/line/rnnOutput.csv" # File path
    
    # Import CSV
    mtx = readCSV(filePath)

    # Validations
    if beamWidth < 1: 
        raise("Invalid Beam Width: Must be more than 0")
    if len(mtx) < 1:
        raise("Empty data")
    if len(classes) != len(mtx[0]):
        raise("Different length for classes and data rows")
    if np.isnan(mtx).any():
        raise("Data contains non-number")

    # Beam Search
    print(calcBeamSearch(mtx, beamWidth, classes))


main()
