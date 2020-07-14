

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
from numpy import inf
import random

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:


        """
        #print "   "
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        for i in range(X.shape[1]):
            v_,score_,dy,dn = self.evaluate_numerical_attribute (X[:,i],Y) 
            if (score_<scor):
                splitpoint = v_
                scor = score_
                Dy = dy
                Dn = dn
                find = i
        Dy = np.array(Dy)
        Dn = np.array(Dn)

        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#	
        
        if (X<=split):
            return True
        return False
	
    
        #---------End of Your Code-------------------------#

    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        
        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        
        # YOUR CODE HERE
        
        u = np.unique (f)
        
        
        mid = u[:-1] + np.diff(u)/2
           
        entropy = [] 

        for var in mid:
 
            ny = float(np.sum ([f < var]))
            nn = float(np.sum ([f > var]))
            n = float(nn+ny)
            
            t,tc = np.unique (sY[f < var],return_counts = True)
            b,bc = np.unique (sY[f > var],return_counts = True)

            
            top = (ny/n )*-1*np.sum ( np.log2(tc/ny)*(tc/ny))
            bot = (nn/n )*-1*np.sum ( np.log2(bc/nn)*(bc/nn))
            
            entropy.append((top + bot))
            
           
        v = mid[np.argmin (entropy)]

        ind = np.arange(0,Y.shape[0])
        i = ind[np.argmin (entropy)]
        
        mingain= np.min (entropy)
        
        Xlidx = f<v
        Xridx = f>v
        

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using
        a random set of features from the given set of features...

    """

    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        self.fidx=-1
        self.split=-1

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        #print "Inside the train of Random"
        nexamples,nfeatures=X.shape

        #print "Train has X of length ", X.shape


        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        columns = []
        for i in range(self.nrandfeat):
      
            columns.append(np.random.randint((X.shape[1])))
        
        
        bestentropy = np.inf
        bestsplit = np.inf
        Dy = []
        Dn = []
        bfeat = -1
        
        for col in columns:
            split,entropy,Xlidx,Xridx = self.findBestRandomSplit(X[:,col],Y)

            if (bestentropy > entropy):
                bestsplit = split
                bestentropy = entropy
                Dy = Xlidx
                Dn = Xridx
                bfeat = col

        
        
        self.bsplit = split

        return bfeat, bestsplit, bestentropy, np.array(Dy), np.array(Dn)
	
    

        #---------End of Your Code-------------------------#
    

    def findBestRandomSplit(self,feat,Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        X = feat
        entropy = []
        s = []
        for n in range(self.nsplits):
            split = X[np.random.randint (X.shape[0])]
            Dy = X <= split
            entropy.append (self.calculateEntropy(Y,Dy))
            s.append(split)
        s = np.array(s)
        entropy = np.array(entropy)
        splitvalue = s[np.argmin(entropy)]
        minscore = np.min (entropy)
	
        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """
        
        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl))
        hr= -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr
       
        return sentropy



    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
    
        #print X.shape, self.fidx, "xshape"
        
        if (X[f1]<= bsplit):
            return True
        return False
	

        #---------End of Your Code-------------------------#




# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.F1=0
        self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self,nsplits)

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        bestsplit = ()
        bestentropy = np.inf

        bx= -1
        by = -1

        for i in range(X.shape[1]) :
            x = np.random.randint (X.shape[1])
            y = np.random.randint (X.shape[1])
          
            for n in range(self.nsplits):
                c_split = X[np.random.randint (X.shape[0]),y]
                m = np.random.normal(-3,3)

                xx = X[:,x]
                yy = X[:,y]
                equation = yy - np.dot(m,xx) - c_split
                dy = equation <= 0 
                entropy = self.calculateEntropy (Y,dy)

                if (bestentropy > entropy):
                    bestsplit = (c_split,m)
                    bestentropy = entropy
                    Dy = equation <= 0
                    Dn = equation > 0
                    bx = x
                    by = y
        

        return (bx,by),bestsplit, bestentropy, np.array(Dy), np.array(Dn)
	
	    
        #---------End of Your Code-------------------------#
#         return 0, minscore, bXl, bXr

#     self,parameters,fx,fy,X
    def evaluate(self,parameters,fx,fy,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        if (X[fy]-np.dot(X[fx],parameters[1]) - parameters[0]  <= 0):
            return True
        return False
	

        #---------End of Your Code-------------------------#


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.d=0
        self.e=0
        self.f=0
        self.F1=0
        self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        a, b, c, d, e, f = np.random.uniform(-3, 3, (6,))
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        bestsplit = ()
        bestentropy = np.inf
        bx = -1
        by = -1

        for i in range(X.shape[1]) :
            xx = np.random.randint (nfeatures)
            yy = np.random.randint (nfeatures)
            entropy = []
            s = []
            for n in range(self.nsplits):
                A = np.random.normal (-3,3)
                B = np.random.normal (-3,3)
                C = np.random.normal (-3,3)
                D = np.random.normal (-3,3)
                E = np.random.normal (-3,3)
                F = np.random.normal (-3,3)

                x = X[:,xx]
                y = X[:,yy]
                #build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
                equation = A*x**2+B*y**2+C*x*y+D*x+E*y+F

                dy = equation <= 0
                entropy = self.calculateEntropy(Y,dy)

                if (bestentropy > entropy):
                    bestsplit = (A,B,C,D,E,F)
                    bestentropy = entropy
                    Dy = equation <= 0
                    Dn = equation > 0
                    bx = xx
                    by = yy

            return (bx,by),bestsplit, bestentropy, np.array(Dy), np.array(Dn)
	
        #---------End of Your Code-------------------------#
#         return 0, minscore, bXl, bXr

    def evaluate(self,parameters,fx,fy,X):
        """
        Evalute the trained weak learner  on the given example...
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        

        x = X[fx]
        y = X[fy]

        if ( parameters[0]*x**2+parameters[1]*y**2+parameters[2]*x*y+parameters[3]*x+parameters[4]*y+parameters[5]<= 0):
            return True
        return False
	
        #---------End of Your Code-------------------------#
