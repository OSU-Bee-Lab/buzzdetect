import os

def concatfile(fpath):

    results = [("start","end","classification")]
    runningStart = -1
    lastEnd = -1
    
    with open(fpath) as f:
        next(f)
        #print( f.readlines() )
        
        fileArr = f.read().splitlines()
        for i, line in enumerate(fileArr):
            vals = line.split(",")
            
            if (runningStart == -1):
                runningStart = float(vals[0])
                lastEnd = float(vals[1])
            

            if (lastEnd < float(vals[0])):
                results.append( ( runningStart, lastEnd, "bee" ) )
                runningStart = float(vals[0])
                
            
            if (i == len(fileArr) - 1):
                results.append( ( runningStart, float(vals[1]), "bee" ) )
                runningStart = -1
                lastEnd = -1
            
            lastEnd = float(vals[1])

    # results.pop(0)
    print(results)

    open(fpath, "w").close()
    
    with open(fpath, 'a') as f:
        for line in results:
            f.write(f"{line[0]},{line[1]},{line[2]}\n")
            #print(f"{line[0]},{line[1]},{line[2]}")
