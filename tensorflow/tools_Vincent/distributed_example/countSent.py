import sys
import json
import subprocess
import os

import random
import string

def parseSentData(fileNames, outFile=""):
    for jsonF in fileNames:
        #TODO: Error non-JSON files
        sentBytes = dict()
        print("Opening trace file: " + jsonF)
        with open(jsonF) as f:
            chrome_trace = json.load(f)
            this_device = ""
            for event in chrome_trace['traceEvents']:
                if event['name'] == "_Send" and event["send_device"]:
                    if this_device == "":
                        this_device = event["send_device"]
                    else:
                        assert(this_device == event["send_device"])
                    sentBytes[event["recv_device"]] = sentBytes.setdefault( event["recv_device"], 0) + event["sendsize"]
        print("This worker: %s" % this_device)
        print("Sent:")
        for recv, datasize in sentBytes.iteritems():
            print("Receiving worker: %s\tBytes: %d %s" % (recv, datasize, "\t(ME)" if recv==this_device else "") )


machines = ['engelbart.cs.ubc.ca', 'micali.cs.ubc.ca']
userID = 'vhui'
srcDir = ['/home/vhui/Documents/distributed-tensorflow-example/']
destDir = ['/home/vhui/Documents/distributed-tensorflow-example/']
fileRegex = "dist_mnistTimeline_worker*"

if __name__ == "__main__":
    print("JSON files to parse: %d" % (len(sys.argv)-1) )
    outFile = "" #stdout for now
    #parseSentData(sys.argv[1:], outFile)
    
    #Generate random directory for download
    randomDir = 'tmp_' + ''.join(random.sample(string.lowercase + string.digits,8))
    destPath = destDir[0] + randomDir
    print("Creating random directory: " + randomDir)
    subprocess.Popen(["mkdir", destPath], stdout=subprocess.PIPE).communicate()   
 
    for machine in machines: #TODO: assume stale file overwritten
        srcPath = "%s@%s:%s%s" % (userID, machine, srcDir[0], fileRegex)
        print("Downloading via SCP")
        p = subprocess.Popen(["scp", srcPath, destPath], stdout=subprocess.PIPE)
        p.communicate()
   
    #os.listdir(destPath) 
    allTraces = os.listdir(destPath)
    allTraces = [destPath + "/" +tracefile for tracefile in allTraces]
    print(allTraces)

    #parseSentData(allTraces, outFile)
    parseSentData([x for x in allTraces if ":2" not in x and "MERGE" not in x ], outFile)

