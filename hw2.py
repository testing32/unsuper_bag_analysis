import glob
import re
import scipy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import Image
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cross_validation
import cPickle as pickle

ALL_ITEM_CATEGORIES = ('black','brown','red','silver','gold', 'unknown')
ITEM_COLOR_CATEGORIES = ('black','brown','red','silver','gold')
NO_WORDS = ('white','ivory','navy','purple','green','orange','blue','grey','gray','amber','ruby','olive','bronze','violet','eggshell', 'snake', 'croc', 'hardware')
# ,'amber','ruby','olive','bronze','violet','eggshell'

class Item(object):

    def __init__(self, name, dicLoc):
        self.name = name
        self.dicLoc = dicLoc

    def isGreyScale(self):
        im = Image.open(self.getFullImageLocation()).convert('RGB')
        w,h = im.size
        for i in range(w):
            for j in range(h):
                r,g,b = im.getpixel((i,j))
                if r != g != b: return False
        return True
    
    # get the image file name
    def getImageName(self):
        return "img_" + self.name + ".jpg"
    
    def getFullImageLocation(self):
        return self.dicLoc + self.getImageName()
    
    # gets the next file name
    def getTextName(self):
        return "descr_" + self.name + ".txt"
    
    # gets the full description
    def getText(self):
        queryDescr = open(self.dicLoc + self.getTextName(), "r")
        return queryDescr.readlines()[0].lower()
    
    # list of words in the text description
    def getWordList(self):
        return [s for s in re.findall(r"[A-Za-z']+",self.getText())]
    
    def getHSVHistogram(self):
        
        img = cv2.imread(self.getFullImageLocation())
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1, 2], None, [10, 10, 10], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        histsum = sum(hist)
        return [float(val)/histsum for val in hist]
    
    def getColorHistogram(self):
        img = cv2.imread(self.getFullImageLocation(),0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        hist = cv2.normalize(hist).flatten()
        #hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
        return hist

class ItemCollection(object):
    def __init__(self, dicLoc,noWords=None):
        self.bagTypes = {}
        for category in ALL_ITEM_CATEGORIES:
            self.bagTypes[category] = []
        
        fileNamesList = [re.search('descr_(.+?)\.txt', s.lower()).group(1) for s in glob.glob(dicLoc + "descr_*.txt")]
        
        # for each bag
        for fileName in fileNamesList:
            
            #create an item object and get a list of words
            # in the description of the object
            newItem = Item(fileName, dicLoc)
            
            if newItem.isGreyScale():
                continue
            
            text = newItem.getWordList()
            
            colorResult = []
            for color in ITEM_COLOR_CATEGORIES:
                colorResult.append(color in text)
            
            if sum(colorResult) != 1 or ((noWords is not None) and len(set(text)&set(noWords))>0):
                self.bagTypes['unknown'].append(newItem)
            else:         
                for i in range (0, len(ITEM_COLOR_CATEGORIES)):
                    if colorResult[i]:
                        self.bagTypes[ITEM_COLOR_CATEGORIES[i]].append(newItem)
                        break
                
    def getBagTypes(self):
        return self.bagTypes

    def moveFromUnknown(self, category, item):
        self.bagTypes[category].append(item)
        self.bagTypes['unknown'].remove(item)

    def getXTraining(self):
        xTraining = []
        
        for key in ITEM_COLOR_CATEGORIES:
            for item in self.bagTypes[key]:
                xTraining.append(item.getHSVHistogram())
        return xTraining

    def getYTraining(self, color):
        
        yTraining = []
        for key in ITEM_COLOR_CATEGORIES:
            for i in range(0, len(self.bagTypes[key])):
                yTraining.append((key==color)*2-1)
        
        return yTraining


class SVMCollection(object):
    def __init__(self, itemCollection, xTraining=None):
        self.svmDict = {}
        if xTraining is None:
            xTraining = itemCollection.getXTraining()

        for category in ITEM_COLOR_CATEGORIES:
            svmModel, scaler = self.createSklearnSVM(xTraining, itemCollection.getYTraining(category))
            #svmModel, scaler = self.crossValidateSVM(xTraining, itemCollection.getYTraining(category))
            self.svmDict[category] = {'svm': svmModel, 'scaler': scaler}
    
    
    def createSklearnSVM(self, xValues, yValues):
        
        # fix up x values
        scaler = StandardScaler()
        xValues = scaler.fit_transform(xValues)
        
        # create the svm
        svmModel = svm.SVC(C=5.0, probability=True)
        svmModel.fit(xValues,yValues)
        
        return svmModel, scaler
        
    def crossValidateSVM(self, xValues, yValues):
             
        # fix up x values
        scaler = StandardScaler()
        xValues = scaler.fit_transform(xValues)
        
        # Cross Validation Code
        xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xValues,yValues,test_size=.7,random_state=0)
        
        svmModel = svm.SVC(C=5.0).fit(xTrain,yTrain)
        
        print(svmModel.score(xTest,yTest))
        return svmModel, scaler
      
    def predictProb(self, hist, color):
        xValues = self.svmDict[color]['scaler'].transform(hist)
        return self.svmDict[color]['svm'].predict_proba(xValues)

    def predict(self, hist, color):
        xValues = self.svmDict[color]['scaler'].transform(hist)
        return self.svmDict[color]['svm'].predict(xValues)

def generateHTMLFile(itemList, fileName):
    # generates an html files which contains the images
    # this made it easy to add pictures to the hw document
    html = "<html><body>"
    
    for item in itemList:
        html += '<img src="' + item.getFullImageLocation() + '" />'

    html += "</body></html>"
    myHtmlDoc = open(fileName, 'w')
    myHtmlDoc.write(html)
    myHtmlDoc.close()     
   
def generateHTMLCountFile(itemList, fileName):
    # generates an html files which contains the images
    # this made it easy to add pictures to the hw document
    html = "<html><body>"
    count = 1
    for item in itemList:
        html += '<figure>'
        html += '<img src="' + item.getFullImageLocation() + '" />'
        html += '<figcaption>' + str(count) + '</figcaption>'
        html += '</figure>'
        count += 1
        
    html += "</body></html>"
    myHtmlDoc = open(fileName, 'w')
    myHtmlDoc.write(html)
    myHtmlDoc.close()
   
def generateHTMLCaptionFile(itemList, fileName):
    # generates an html files which contains the images
    # and the text associated with the image, this makes for easier debugging
    html = "<html><body>"
    
    for item in itemList:
        html += '<figure>'
        html += '<img src="' + item.getFullImageLocation() + '" />'
        html += '<figcaption>' + item.getText() + '</figcaption>'
        html += '</figure>'
    
    html += "</body></html>"
    myHtmlDoc = open(fileName, 'w')
    myHtmlDoc.write(html)
    myHtmlDoc.close() 
    

def partOne(dictionaryLoc):
    #itemCollection = ItemCollection(dictionaryLoc)
    #pickle.dump(itemCollection, open('itemCollection',"wb"))
    itemCollection = pickle.load(open('itemCollection',"rb"))
    itemBagTypes = itemCollection.getBagTypes()
    
    for cat in ALL_ITEM_CATEGORIES:
        generateHTMLFile(itemBagTypes[cat], cat + ".html")

def partTwo(dictionaryLoc):
    #itemCollection = ItemCollection(dictionaryLoc)
    #pickle.dump(itemCollection, open('itemCollection',"wb"))
    itemCollection = pickle.load(open('itemCollection',"rb"))
    itemBagTypes = itemCollection.getBagTypes()
    
    print itemBagTypes['brown'][1].getHSVHistogram()

def partThreeAndFour(dictionaryLoc):
    
    # Create a collection, get the X training data, create the SVMs
    # the lines are commented out to save time
    
    #itemCollection = ItemCollection(dictionaryLoc)
    #pickle.dump(itemCollection, open('itemCollection',"wb"))
    itemCollection = pickle.load(open('itemCollection',"rb"))
    #pickle.dump(itemCollection.getXTraining(), open('bagXTraining',"wb"))
    #svmCollection = SVMCollection(itemCollection, xTraining=pickle.load(open('bagXTraining',"rb")))
    #pickle.dump(svmCollection, open('svmCollection',"wb"))
    svmCollection = pickle.load(open('svmCollection',"rb"))
    
    # setup
    predictions = {}
    for category in ITEM_COLOR_CATEGORIES:
        predictions[category] = []
        
    # make a prediction for each unknown type for each svm we have
    for item in itemCollection.getBagTypes()['unknown']:
        hist = item.getHSVHistogram()
        for color in ITEM_COLOR_CATEGORIES:
            probTrue = svmCollection.predictProb(hist, color)[0][1]
            predictions[color].append({'item':item,'probTrue':probTrue })
    
    # generate sorted, based on the prediction, html files     
    for category in ITEM_COLOR_CATEGORIES:
        generateHTMLFile([dictItem['item'] for dictItem in sorted(predictions[category], key=itemgetter('probTrue'),reverse=True)[:200]], 'predicted' + category + '.html')

def partFive(dictionaryLoc):
    #itemCollection = ItemCollection(dictionaryLoc,noWords=NO_WORDS)
    #pickle.dump(itemCollection, open('partFiveItemCollection',"wb"))
    itemCollection = pickle.load(open('partFiveItemCollection',"rb"))
    
    #pickle.dump(itemCollection.getXTraining(), open('partFivebagXTraining',"wb"))
    #svmCollection = SVMCollection(itemCollection, xTraining=pickle.load(open('partFivebagXTraining',"rb")))
    #pickle.dump(svmCollection, open('partFivesvmCollection',"wb"))
    svmCollection = pickle.load(open('partFivesvmCollection',"rb"))
    
    # setup
    predictions = {}
    for category in ITEM_COLOR_CATEGORIES:
        predictions[category] = []
        
    # make a prediction for each unknown type for each svm we have
    for item in itemCollection.getBagTypes()['unknown']:
        hist = item.getHSVHistogram()
        for color in ITEM_COLOR_CATEGORIES:
            probTrue = svmCollection.predictProb(hist, color)[0][1]
            predictions[color].append({'item':item,'probTrue':probTrue })
    
    # generate sorted, based on the prediction, html files     
    #for category in ITEM_COLOR_CATEGORIES:
    #    generateHTMLFile([dictItem['item'] for dictItem in sorted(predictions[category], key=itemgetter('probTrue'),reverse=True)[:200]], 'predicted' + category + '.html')
    for category in ITEM_COLOR_CATEGORIES:
        sortedList = [dictItem['item'] for dictItem in sorted(predictions[category], key=itemgetter('probTrue'),reverse=True)[:20]]
        for item in sortedList:
            itemCollection.moveFromUnknown(category, item)

    pickle.dump(itemCollection, open('partFiveItemCollectionTwo',"wb"))
    itemCollection = pickle.load(open('partFiveItemCollectionTwo',"rb"))
    pickle.dump(itemCollection.getXTraining(), open('partFivebagXTrainingTwo',"wb"))
    svmCollection = SVMCollection(itemCollection, xTraining=pickle.load(open('partFivebagXTrainingTwo',"rb")))
    pickle.dump(svmCollection, open('partFivesvmCollectionTwo',"wb"))
    svmCollection = pickle.load(open('partFivesvmCollectionTwo',"rb"))
    
    # setup
    predictions = {}
    for category in ITEM_COLOR_CATEGORIES:
        predictions[category] = []
        
    # make a prediction for each unknown type for each svm we have
    for item in itemCollection.getBagTypes()['unknown']:
        hist = item.getHSVHistogram()
        for color in ITEM_COLOR_CATEGORIES:
            probTrue = svmCollection.predictProb(hist, color)[0][1]
            predictions[color].append({'item':item,'probTrue':probTrue })
     
    for category in ITEM_COLOR_CATEGORIES:
        generateHTMLFile([dictItem['item'] for dictItem in sorted(predictions[category], key=itemgetter('probTrue'),reverse=True)[:200]], 'predicted' + category + '.html')
    
def test():
    itemCollection = pickle.load(open('itemCollection',"rb"))
    
    bagTypes = itemCollection.getBagTypes()
    
    for i in bagTypes.keys():
        print i + " " + str(len(bagTypes[i]))

def test2(dicLoc):
    itemCollection = pickle.load(open('itemCollection',"rb"))
    bagTypes = itemCollection.getBagTypes()
    
    for image in bagTypes['silver']:
        img = scipy.misc.imread(image.getFullImageLocation())
        array=np.asarray(img)
        arr=(array.astype(float))/255.0
        img_hsv = colors.rgb_to_hsv(arr[...,:3])
        
        lu1=img_hsv[...,0].flatten()
        plt.subplot(1,3,1)
        plt.hist(lu1*360,bins=10,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
        plt.title("Hue")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        
        lu2=img_hsv[...,1].flatten()
        plt.subplot(1,3,2)                  
        plt.hist(lu2,bins=10,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
        plt.title("Saturation")   
        plt.xlabel("Value")    
        plt.ylabel("Frequency")
        plt.legend()
        
        lu3=img_hsv[...,2].flatten()
        plt.subplot(1,3,3)                  
        plt.hist(lu3*255,bins=10,range=(0.0,255.0),histtype='stepfilled', color='b', label='Intesity')
        plt.title("Intensity")   
        plt.xlabel("Value")    
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

def test3(dicLoc):
    itemCollection = pickle.load(open('partFiveItemCollection',"rb"))
    itemBagTypes = itemCollection.getBagTypes()
    
    for cat in ALL_ITEM_CATEGORIES:
        #print len(itemBagTypes[cat])
        generateHTMLFile(itemBagTypes[cat], cat + ".html")

def compareColorHist(x1, x2):
    return 1/cv2.compareHist(x1, x2, cv2.cv.CV_COMP_INTERSECT)

def test4():
    
    from sklearn import cluster
    
    itemCollection = pickle.load(open('itemCollection',"rb"))
    itemBagTypes = itemCollection.getBagTypes()
    xTraining = []
        
    for item in itemBagTypes['black']:
        xTraining.append(item.getColorHistogram())
        #xTraining.append(item.getHSVHistogram())

    #myCluster = cluster.AgglomerativeClustering(n_clusters=6,linkage='average')
    #myCluster = cluster.KMeans(n_clusters=3)
    #prediction = myCluster.fit_predict(xTraining)
    myCluster = cluster.DBSCAN(metric=compareColorHist, eps=.65,algorithm='brute')
    prediction = myCluster.fit_predict(np.array(xTraining))
    predictionDivision = {-1:[],0:[],1:[],2:[],3:[],4:[],5:[]}
    for i in range(0, len(prediction)):
        predictionDivision[prediction[i]].append(itemBagTypes['black'][i])
    
    for i in range(-1,6):
        generateHTMLFile(predictionDivision[i], 'clustered' + str(i) + '.html')

if __name__ == "__main__":
    dictionaryLoc = "/home/testing32/Downloads/shopping/bags/"
    
    #partOne(dictionaryLoc)
    #partTwo(dictionaryLoc)
    #partThreeAndFour(dictionaryLoc)
    #partFive(dictionaryLoc)
    #test()
    #test2(dictionaryLoc)
    #test3(dictionaryLoc)
    test4()
    