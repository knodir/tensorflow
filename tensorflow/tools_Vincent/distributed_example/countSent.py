import sys
import json
import subprocess
import os

import random
import string

def buildConfigMap(fileName="testConfig.json"):
    clusterConfig = dict()
    with open(fileName) as config:
        clusterConfig = json.load(config)

    flatConfig = dict()
    for jobType, jobsDict in clusterConfig.iteritems():
        flatConfig.update( jobsDict )
    return flatConfig

def parseSentData(fileNames, config="", outFile=""):
    for jsonF in fileNames:
        #TODO: Error non-JSON files
        commMap = dict()
        aliases = dict()

        print("Opening trace file: " + jsonF)
        with open(jsonF) as f:
            chrome_trace = json.load(f)
            for event in chrome_trace['traceEvents']:
                if event['name'] == "_Send" and event["send_device"]:
                    if event["send_device"] not in commMap:
                        commMap[event["send_device"]] = dict()
                    dests = commMap[event["send_device"]]
                    dests[event["recv_device"]] = dests.setdefault( event["recv_device"], 0) + event["sendsize"]

            if config:
                def createAlias():  #shorten to config aliases 
                        for event in chrome_trace['traceEvents']:
                            if "send_device" in event and event["send_device"] not in aliases:
                                shortName = ""
                                nameSections = event["send_device"].split("/")
                                for section in nameSections[1:]:
                                    sectionName = section.split(":")[0]
                                    ID = section.split(":")[1]
                                    if sectionName == "job" or sectionName == "task":
                                        shortName += ID

                                aliases[event["send_device"]] = shortName

                            if "recv_device" in event and event["recv_device"] not in aliases:
                                shortName = ""
                                nameSections = event["recv_device"].split("/")
                                for section in nameSections[1:]:
                                    sectionName = section.split(":")[0]
                                    ID = section.split(":")[1]
                                    if sectionName == "job" or sectionName == "task":
                                        shortName += ID
     
                                aliases[event["recv_device"]] = shortName
                createAlias()

        for sender, dests in commMap.iteritems():
            sender = config[aliases[sender]] if config else sender
            print("Sender: %s" % sender)
            print("Sent to:")
            for recv, datasize in dests.iteritems():
                recv = config[aliases[recv]] if config else recv
                print("\tReceiving worker: %s\tBytes: %d %s" % (recv, datasize, "\t(ME)" if recv==sender else "") )


machines = ['engelbart.cs.ubc.ca', 'micali.cs.ubc.ca']
userID = 'vhui'
srcDir = '/home/vhui/Documents/distributed-tensorflow-example/'
destDir = '/home/vhui/Documents/distributed-tensorflow-example/'
fileRegex = "dist_mnistTimeline_worker*"

if __name__ == "__main__":
    #print("JSON files to parse: %d" % (len(sys.argv)-1) )
    #parseSentData(sys.argv[1:], outFile)
    
    #Generate random directory for download
    randomDir = 'tmp_' + ''.join(random.sample(string.lowercase + string.digits,8))
    destPath = destDir + randomDir
    print("Creating random directory: " + randomDir)
    subprocess.Popen(["mkdir", destPath], stdout=subprocess.PIPE).communicate()   
 
    for machine in machines: #TODO: assume stale file overwritten
        srcPath = "%s@%s:%s%s" % (userID, machine, srcDir, fileRegex)
        print("Downloading via SCP")
        p = subprocess.Popen(["scp", srcPath, destPath], stdout=subprocess.PIPE)
        p.communicate()
   
    allTraces = os.listdir(destPath)
    allTraces = [destPath + "/" +tracefile for tracefile in allTraces]
    #print(allTraces)

    parseSentData(allTraces, config=buildConfigMap() )
    #parseSentData([x for x in allTraces if ":2" not in x and "MERGE" not in x ], outFile)

