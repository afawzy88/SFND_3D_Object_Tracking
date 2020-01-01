
#include <iostream>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <set>

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    
    vector<cv::DMatch> selectedMatches;
    vector<cv::DMatch> tempMatches;

    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::Point2f keyPointCurr = kptsCurr[it->trainIdx].pt;
        if (boundingBox.roi.contains(keyPointCurr))
        {
            tempMatches.push_back(*it);
        }
    }

    //Vector of matches and corresponding euclidean distance for each keypoints match pair
    vector<pair<cv::DMatch,double>> eucliDistAndMatchVec;
    // Vector of euclidean distances only
    vector<double> eucliDistVec;
    
    for (auto it = tempMatches.begin(); it != tempMatches.end(); ++it)
    {
        cv::Point2f point1 = kptsCurr[it->trainIdx].pt;
        cv::Point2f point2 = kptsPrev[it->queryIdx].pt;
        double x1,y1,x2,y2,dist;
        x1 = point1.x;
        y1 = point1.y;
        x2 = point2.x;
        y2 = point2.y;
        dist = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
        eucliDistVec.push_back(dist);
        eucliDistAndMatchVec.push_back({*it,dist});
    } 

    double median, variance = 0; //median is a robust mean
    double stdDeviation;

    sort(eucliDistVec.begin(),eucliDistVec.end());

    if(eucliDistVec.size()%2==1) //Number of elements are odd
    {
        median = eucliDistVec[eucliDistVec.size()/2];
    }
    else // Number of elements are even
    {
        int index = eucliDistVec.size()/2;
        median = (eucliDistVec[index-1] + eucliDistVec[index])/2;
    }
    for (auto it = eucliDistVec.begin(); it != eucliDistVec.end(); ++it)
    {
        variance += pow(*it - median, 2);
    }
    variance = variance / eucliDistVec.size();
    stdDeviation = sqrt(variance);

    for (auto it = eucliDistAndMatchVec.begin(); it != eucliDistAndMatchVec.end(); ++it)
    {
        //Consider only the matches with euclidean distance between (mio - sigma) and (mio + sigma) i.e not far from the median
        if (it->second >= (median - stdDeviation) && it->second <= (median + stdDeviation))
            selectedMatches.push_back(it->first);
    }

    boundingBox.kptMatches = selectedMatches;

    //cout << "Size of boundingBox.kptMatches = " << boundingBox.kptMatches.size() <<endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    double dT = 1 / frameRate;
    
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());

    double medianDistRatio;
    if (distRatios.size()%2==1) //Number of elements are odd
    {
        medianDistRatio = distRatios[distRatios.size()/2];
    }
    else // Number of elements are even
    {
        int index = distRatios.size()/2;
        medianDistRatio = (distRatios[index-1] + distRatios[index])/2;
    }

    TTC = -dT / (1 - medianDistRatio);

    //cout << "TTC Camera = " << TTC << " seconds" <<endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, string option)
{
    double dT = 1 / frameRate; // time between two measurements in seconds

    if (option.compare("Average") == 0)
    {
        double sumOfXElementsPrev, averageXPrev = 0;
        double sumOfXElementsCurr, averageXCurr = 0;


        for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it)
        {
        sumOfXElementsPrev += it->x;
        }
        averageXPrev = sumOfXElementsPrev/lidarPointsPrev.size();

        for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it)
        {
            sumOfXElementsCurr += it->x;
        }
        averageXCurr = sumOfXElementsCurr/lidarPointsCurr.size();

        // compute TTC from both measurements
        TTC = averageXCurr * dT / (averageXPrev-averageXCurr);
    }
    else if (option.compare("Median") == 0)
    {
        vector<double> xPrevVec;
        vector<double> xCurrVec;
        double medianXPrev,medianXCurr;

        for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it)
        {
            xPrevVec.push_back(it->x);
        }
        sort(xPrevVec.begin(),xPrevVec.end());
        if(xPrevVec.size()%2==1) //Number of elements are odd
        {
            medianXPrev = xPrevVec[xPrevVec.size()/2];
        }
        else // Number of elements are even
        {
            int index = xPrevVec.size()/2;
            medianXPrev = (xPrevVec[index-1] + xPrevVec[index])/2;
        }

        for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it)
        {
            xCurrVec.push_back(it->x);
        }
        sort(xCurrVec.begin(),xCurrVec.end());
        if(xCurrVec.size()%2==1) //Number of elements are odd
        {
            medianXCurr = xCurrVec[xCurrVec.size()/2];
        }
        else // Number of elements are even
        {
            int index = xCurrVec.size()/2;
            medianXCurr = (xCurrVec[index-1] + xCurrVec[index])/2;
        }

        // compute TTC from both measurements
        TTC = medianXCurr * dT / (medianXPrev-medianXCurr);

    }

    //cout << "TTC LiDAR = " << TTC << " seconds" <<endl;
}


void matchBoundingBoxesOption2(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int,int> potentialBBMatches;

    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        cv::Point2f keyPointPrev = prevFrame.keypoints[it->queryIdx].pt;
        cv::Point2f keyPointCurr = currFrame.keypoints[it->trainIdx].pt;
        int boxIDPrev = -1;
        int boxIDCurr = -1;
    
        for(auto it1=prevFrame.boundingBoxes.begin(); it1!=prevFrame.boundingBoxes.end(); ++it1)
        {
            if (it1->roi.contains(keyPointPrev))
                boxIDPrev = it1->boxID;
        }
        for(auto it2=currFrame.boundingBoxes.begin(); it2!=currFrame.boundingBoxes.end(); ++it2)
        {
            if (it2->roi.contains(keyPointCurr))
                boxIDCurr = it2->boxID;
        }
        if (!(boxIDPrev == -1 || boxIDCurr == -1))
        {
            potentialBBMatches.insert({boxIDPrev,boxIDCurr});
        }
    }
    
    //Set holding each key (boxIDPrev) from potentialBBMatches along with its number of occurences
    set<pair<int,int>> boxIDPrevAndNumOfOccur;
    for(auto it3=potentialBBMatches.begin(); it3!=potentialBBMatches.end(); ++it3)
    {
        boxIDPrevAndNumOfOccur.insert({potentialBBMatches.count(it3->first),it3->first});
    }

    //Loop over the list of most occurring keys/boxIDPrev starting from last element (the most occuring)
    for (set<pair<int,int>>::reverse_iterator rit = boxIDPrevAndNumOfOccur.rbegin(); rit!=boxIDPrevAndNumOfOccur.rend(); ++rit)
    {
        int key = rit->second;
        vector<int> corresBoxIDsCurrVec;
        set<pair<int,int>> boxIDCurrAndNumOfOccur;

        for(auto it=potentialBBMatches.begin(); it!=potentialBBMatches.end(); ++it)
        {
            if (it->first == key)
            {
                //Store all "values" for that "key"
                corresBoxIDsCurrVec.push_back(it->second);
            }
        }
        for(int i = 0; i < corresBoxIDsCurrVec.size(); i++)
        {
            boxIDCurrAndNumOfOccur.insert({count(corresBoxIDsCurrVec.begin(),corresBoxIDsCurrVec.end(),corresBoxIDsCurrVec[i]), corresBoxIDsCurrVec[i]});
        }
        int value = boxIDCurrAndNumOfOccur.rbegin()->second;
        
        bbBestMatches.insert({key,value});
    }

    cout << "bbBestMatches are:- " <<endl;
    for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
    {
        cout << it->first << ":" << it->second <<endl;
    }
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int,int> potentialBBMatches;

    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        cv::Point2f keyPointPrev = prevFrame.keypoints[it->queryIdx].pt;
        cv::Point2f keyPointCurr = currFrame.keypoints[it->trainIdx].pt;
        int boxIDPrev = -1;
        int boxIDCurr = -1;
    
        for(auto it1=prevFrame.boundingBoxes.begin(); it1!=prevFrame.boundingBoxes.end(); ++it1)
        {
            if (it1->roi.contains(keyPointPrev))
                boxIDPrev = it1->boxID;
        }
        for(auto it2=currFrame.boundingBoxes.begin(); it2!=currFrame.boundingBoxes.end(); ++it2)
        {
            if (it2->roi.contains(keyPointCurr))
                boxIDCurr = it2->boxID;
        }
        // Avoid storing boxIDs when points not enclosed by boxes
        if (!(boxIDPrev == -1 || boxIDCurr == -1))
        {
            potentialBBMatches.insert({boxIDPrev,boxIDCurr});
        }
    }

    for (auto it = potentialBBMatches.begin(); it != potentialBBMatches.end(); ++it)
    {
        int key = it->first;
        //Iterator to range of pairs having the same "key"
        auto rangeOfPairsWithKey = potentialBBMatches.equal_range(key);

        vector<int> valuesOfKeyVec;
        set<pair<int,int>> valueAndCount;

        for (auto itr = rangeOfPairsWithKey.first; itr != rangeOfPairsWithKey.second; ++itr)
        {
            //Store each "value" corresponding to "key" within the range of pairs in vector
            valuesOfKeyVec.push_back(itr->second);
        }
        for(int i = 0; i < valuesOfKeyVec.size(); i++)
        {
            //Store a pair of  {"value", "its number of occurrences"} in sorted set
            valueAndCount.insert({count(valuesOfKeyVec.begin(),valuesOfKeyVec.end(),valuesOfKeyVec[i]), valuesOfKeyVec[i]});
        }
        //Last element of the set is the "value" with highest number of occurrences
        int value = valueAndCount.rbegin()->second;

        bbBestMatches.insert({key,value});
    }

    /*cout << "bbBestMatches are:- " <<endl;
    for (auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
    {
        cout << it->first << ":" << it->second <<endl;
    }*/
}