import numpy as np
from scipy.stats import skew
import cv2
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dolphin_detector.config import MAX_FREQUENCY, MAX_TIME
except ImportError:
    from config import MAX_FREQUENCY, MAX_TIME

def buildHeader(tmpl, maxT=MAX_FREQUENCY, maxTime=MAX_TIME):
    """ Build a header

        Build the header for the metrics

        Args:
            tmpl: templateManager object

        Returns:
            header string as csv
    """
    hdr_ = []
    prefix_ = ['max','xLoc','yLoc']
    for p_ in prefix_:
        for i in range(tmpl.size):
            hdr_.append(p_+'_'+str(tmpl.info[i]['file']))
    for p_ in prefix_:
        for i in range(tmpl.size):
            hdr_.append(p_+'H_'+str(tmpl.info[i]['file']))

    # Add time metrics
    for i in range(maxTime):
        hdr_ += ['centTime_%04d'%i]
    for i in range(maxTime):
        hdr_ += ['bwTime_%04d'%i]
    for i in range(maxTime):
        hdr_ += ['skewTime_%04d'%i]
    for i in range(maxTime):
        hdr_ += ['tvTime_%04d'%i]

    # Add high frequency metrics
    hdr_ += ['CentStd','AvgBwd','hfCent','hfBwd']
    hdr_ += ['hfMax','hfMax2','hfMax3']
    return ','.join(hdr_)

def computeMetrics(P, tmpl, bins, maxF, maxTime):
    """ Compute a bunch of metrics

        Perform template matching and time stats

        Args:
            P: 2-d numpy array
            tmpl: templateManager object
            bins: time bins
            maxT: maximum frequency slice for time stats

        Returns:
            List of metrics
    """
    # Use full image dimensions for sliding window
    Q = slidingWindowV(P, inner=3, outer=64, maxM=maxF, norm=True)
    W = slidingWindowH(P, inner=3, outer=32, maxM=maxF, norm=True)
    out = templateMetrics(Q, tmpl)    
    out += templateMetrics(W, tmpl)    
    out += timeMetrics(P, bins, maxM=maxTime)
    out += highFreqMetrics(P, bins)
    return out

def matchTemplate(P, template):
    """ Max correlation and location

        Calls opencv's matchTemplate and returns the
        max correlation and location

        Args:
            P: 2-d numpy array to search
            template: 2-d numpy array to match

        Returns:
            maxVal: max correlation
            maxLoc: location of the max
    """
    m, n = template.shape
    p_height, p_width = P.shape
    
    # Check if template is larger than image
    if m > p_height or n > p_width:
        print(f"Warning: Template size ({m}x{n}) is larger than image ({p_height}x{p_width}). Skipping template.")
        return 0.0, 0, 0
    
    mf = cv2.matchTemplate(P.astype('float32'), template, cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
    return maxVal, maxLoc[0], maxLoc[1]

def slidingWindowV(P, inner=3, outer=64, maxM=MAX_FREQUENCY, norm=True):
    """ Enhance the constrast vertically (along frequency dimension)

        Cut off extreme values and demean the image
        Utilize numpy convolve to get the mean at a given pixel
        Remove local mean with inner exclusion region

        Args:
            P: 2-d numpy array image
            inner: inner exclusion region 
            outer: length of the window
            maxM: size of the output image in the y-dimension
            norm: boolean to cut off extreme values

        Returns:
            Q: 2-d numpy contrast enhanced vertically
    """
    Q = P.copy()
    m, n = Q.shape
    if norm:
        mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    for i in range(n):
        Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)

    return Q

def slidingWindowH(P, inner=3, outer=32, maxM=MAX_FREQUENCY, norm=True):
    """ Enhance the constrast horizontally (along temporal dimension)

        Cut off extreme values and demean the image
        Utilize numpy convolve to get the mean at a given pixel
        Remove local mean with inner exclusion region

        Args:
            P: 2-d numpy array image
            inner: inner exclusion region 
            outer: length of the window
            maxM: size of the output image in the y-dimension
            norm: boolean to cut off extreme values

        Returns:
            Q: 2-d numpy contrast enhanced vertically
    """
    Q = P.copy()
    m, n = Q.shape
    if norm:
        mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    for i in range(min(maxM, m)):
        Q[i,:] = Q[i,:] - (np.convolve(Q[i,:],wOuter,'same') - np.convolve(Q[i,:],wInner,'same'))/(outer - inner)

    return Q

def timeMetrics(P, b, maxM=50):
    """ Calculate statistics for a range of frequency slices

        Calculate centroid, width, skew, and total variation
            let x = P[i,:], and t = time bins
            centroid = sum(x*t)/sum(x)
            width = sqrt(sum(x*(t-centroid)^2)/sum(x))
            skew = scipy.stats.skew
            total variation = sum(abs(x_i+1 - x_i))

        Args:
            P: 2-d numpy array image
            b: time bins 

        Returns:
            A list containing the statistics

    """
    m, n = P.shape
    cf_ = [np.sum(P[i,:]*b)/np.sum(P[i,:]) for i in range(maxM)]
    bw_ = [np.sum(P[i,:]*(b - cf_[i])*(b - cf_[i]))/np.sum(P[i,:]) for i in range(maxM)]
    sk_ = [skew(P[i,:]) for i in range(maxM)]
    tv_ = [np.sum(np.abs(P[i,1:] - P[i,:-1])) for i in range(maxM)]
    return cf_ + bw_ + sk_ + tv_

        
def highFreqTemplate(P, tmpl):
    """ High frequency template matching

        Apply horizontal contrast enhancement and
        look for strong vertical features in the image.
        Cut out the lower frequencies

        Args:
            P: 2-d numpy array image
            tmpl: 2-d numpy array template image

        Returns:
            Maximum correlation as a list
    """
    Q = slidingWindowH(P, inner=7, maxM=MAX_FREQUENCY, norm=True)[200:,:]   
    mf = cv2.matchTemplate(Q.astype('float32'), tmpl, cv2.TM_CCOEFF_NORMED)
    return [mf.max()]

def highFreqMetrics(P, bins):
    """ High frequency statistics
        
        Calculate statistics of features at higher frequencies
        This is designed to capture false alarms that occur
        at frequencies higher than typical dolphin clicks.

        Also sum accross frequencies to get an average temporal
        profile. Then return statistics of this profile

        Args:
            P: 2-d numpy array image
            bins: time bins

        Returns:
            A list containing the standard deviation of the
            centroid, the mean of the width, and then 
            the moments of the collapsed slices

    """
    Q = slidingWindowH(P, inner=7, maxM=MAX_FREQUENCY, norm=True)[200:,:]    
    m, n = Q.shape
    cf_ = np.empty(m)
    bw_ = np.empty(m)
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    
    for i in range(m):
        mQ = Q[i,:]
        min_, max_ = mQ.min(), mQ.max()
        # Add epsilon to denominator to prevent division by zero
        mQ = (mQ - min_)/(max_ - min_ + epsilon)
        cf_[i] = np.sum(mQ*bins)/np.sum(mQ + epsilon) 
        bw_[i] = np.sqrt(np.sum(mQ*(bins-cf_[i])*(bins-cf_[i]))/(np.sum(mQ) + epsilon)) 

    mQ = np.sum(Q[50:,:], 0)  # Suggested adjustment
    min_, max_ = mQ.min(), mQ.max()
    # Add epsilon to denominator to prevent division by zero
    mQ = (mQ - min_)/(max_ - min_ + epsilon)
    cfM_ = np.sum(mQ*bins)/(np.sum(mQ) + epsilon) 
    bwM_ = np.sqrt(np.sum(mQ*(bins - cfM_)*(bins - cfM_))/(np.sum(mQ) + epsilon)) 

    return [np.std(cf_), np.mean(bw_), cfM_, bwM_]

def templateMetrics(P, tmpl):
    """ Template matching

        Perform template matching for a list of templates

        Args:
            P: 2-d numpy array image
            tmpl: templateManager object

        Returns:
            List of correlations, x and y pixel locations of the max 
    """
    maxs, xs, ys = [], [], []
    for k in range(tmpl.size):
        mf, y, x = matchTemplate(P, tmpl.templates[k])
        maxs.append(mf)
        xs.append(x)
        ys.append(y)
    return maxs + xs + ys 
