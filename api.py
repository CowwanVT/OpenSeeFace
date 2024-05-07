from websockets.sync.client import connect
import json
import os
import time
import queue



class VtubeStudioAPI():

    customParameterList = ["JawOpen", "MouthPressLipOpen", "MouthFunnel", "MouthPucker", "EyeSquintR", "EyeSquintL", "MouthX", "MouthOpen", "EyeOpenLeft", "EyeOpenRight", "MouthSmile", "BrowLeftY", "BrowRightY", "Brows", "FaceAngleY", "FaceAngleZ", "FaceAngleX", "EyeRightX", "EyeRightY", "EyeLeftX", "EyeLeftY"]

    def __init__(self):

        self.requestID = 0
        self.port = 8001
        self.ip = ""
        self.vtsWebsocket = None
        self.authKey = None
        self.authenticated = False
        self.authFail = False
        self.featureQueue = None

    def sendRequest(self, request):
        request["requestID"] = self.requestID
        self.requestID += 1
        response = None
        while response is None:
            try:
                self.vtsWebsocket.send(json.dumps(request))
                response = json.loads(self.vtsWebsocket.recv())
            except:
                self.connectToVTS()

        return response

    def start(self):
        self.connectToVTS()
        self.createParameters()
        while True:
            parameterList = self.featureQueue.get()
            if self.authenticated:
                self.setParameters(parameterList)

    def connectToVTS(self):
        self.vtsWebsocket = None
        while self.vtsWebsocket is None:
            try:
                self.vtsWebsocket = connect("ws://" + self.ip + ":" + str(self.port))
            except:
                time.sleep(1)

        self.readKeyFile()
        return


    def authenticate(self):
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "1",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "OpenSeeFace but not really",
                "pluginDeveloper": "Cowwan",
                "authenticationToken": ""
                        }
                    }

        request["data"]["authenticationToken"] = self.authKey
        request["requestID"] = self.requestID
        self.requestID += 1
        self.vtsWebsocket.send(json.dumps(request))
        response = json.loads(self.vtsWebsocket.recv())
        if response["data"]["authenticated"]:
            self.authenticated = True
            return
        else:
            self.getNewAuthKey()
            return

    def readKeyFile(self):
        if os.path.isfile("./apiKey"):
            self.keyFile = open("./apiKey", "r")
            authKey = self.keyFile.read()
            self.keyFile.close()
            if authKey == "":
                self.getNewAuthKey()
                return
            else:
                self.authKey = authKey
                self.authenticate()
                return
        else:
            self.getNewAuthKey()
            return

    def getNewAuthKey(self):
        self.keyFile = open("./apiKey", "w")
        request = {"apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "1",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "OpenSeeFace but not really",
                "pluginDeveloper": "Cowwan",
                        }
                    }
        request["requestID"] = self.requestID
        self.requestID += 1
        self.vtsWebsocket.send(json.dumps(request))
        response = json.loads(self.vtsWebsocket.recv())
        if "authenticationToken" in response["data"]:
            self.authKey = response["data"]["authenticationToken"]
            self.keyFile.write(self.authKey )
            self.keyFile.close()
            self.authenticated = True
            return
        else:
            print("VTS Authentication Failed")
            return

    def requestParameterList(self):
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "SomeID",
            "messageType": "InputParameterListRequest"
            }
        response  = self.sendRequest(request)
        existingParameterList = []
        for parameter in response["data"]["customParameters"]:
            existingParameterList.append(parameter["name"])
        return existingParameterList


    def createParameters(self):
        existingParameterList = self.requestParameterList()
        for parameter in self.customParameterList:
            if parameter not in existingParameterList:
                self.createCustomParamter(parameter)
        return

    def createCustomParamter(self, parameterName):
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "SomeID",
            "messageType": "ParameterCreationRequest",
            "data": {
                "parameterName": "MyNewParamName",
                "explanation": "This is my new parameter.",
                "min": -1,
                "max": 1,
                "defaultValue": 0
            }
        }

        request["data"]["parameterName"] = parameterName
        response  = self.sendRequest(request)

        return

    def parameterValueEntry(self, parameterName, value):
        parameter = {
				"id": "",
				"value": 0
                }
        parameter["id"] = parameterName
        parameter["value"] = value
        return parameter

    def setParameters(self, parameterList):
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "SomeID",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": True,
                "mode": "set",
                "parameterValues": []
                }
            }
        for parameter in parameterList:
            if parameter[0] in self.customParameterList:
                parameterEntry = self.parameterValueEntry(parameter[0], parameter[1])
                request["data"]["parameterValues"].append(parameterEntry)

        if len(request["data"]["parameterValues"]) > 0:
            self.sendRequest(request)
        return
