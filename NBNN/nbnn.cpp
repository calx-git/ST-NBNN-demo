//----------------------------------------------------------------------
//		File:			nbnn.cpp
//		Programmer:		Junwu Weng
//		Last modified:	03/21/2018
//		Description:	NBNN for Skeleton-based Action Recognition
//----------------------------------------------------------------------

#include <cstdlib>						                       
#include <cstdio>						                         
#include <cstring>						                            
#include <string>
#include <sstream>
#include <fstream>						                         
#include <iostream>
#include <numeric>
#include <vector>
#include <ANN.h>						                            	// ANN declarations

using namespace std;					                          

#define NAME_DATASET         "MHAD"	
#define DIR_DATASET          "./Dataset"  
#define NAMELIST_SAMPLES     "sampleList-MHAD.txt"				
#define NUM_SAMPLES          659									
#define NUM_DATA_SPLIT       1										 

#define NUM_JOINTS			 35
#define NUM_CLASS            11								 
#define NUM_SUBJECT          12								

#define HIP_INDEX            0								 
#define NUM_MAX_FEATURE      50000							

#define ACTION_SET 			 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}




#define RANGE_ACTION_SNIPPET 	 10
#define DIM_FEATURE          	 NUM_JOINTS*3*RANGE_ACTION_SNIPPET*2	// NUM_JOINTS(20) * NUM_COORD(3) * RANGE_ACTION_SNIPPET(10)
#define NUM_NEARNEIGH        	 1
#define EPS                  	 0										// Error Bound
#define NUM_STAGES				 15
#define OUT_DIR					 "MHAD"


typedef vector< vector< vector< double > > > STM;


struct sample
{
	int cls;
	int sbj;
	int frm;
	int ID;
	int seq;
	string name;
	vector<double*> actSnip; // length is the number of frames of video sample
};


void loadOriFeature(vector<sample> &samples, vector<sample> &samplesOri)
{
	ifstream sampleList;
	sampleList.open(NAMELIST_SAMPLES, ios::in);
	samples.resize(NUM_SAMPLES);
	samplesOri.resize(NUM_SAMPLES);

	printf("%s", "Loading Samples ...");

	for (int fileIdx = 0; fileIdx < NUM_SAMPLES; ++fileIdx)
	{
		sample newSam, newSamOri;
		string sampleName;
		string slash = "/";
		sampleList >> sampleName;


		int aIdx, sIdx, eIdx;
		const char *cSample = sampleName.c_str();
		sscanf( cSample, "a%02d_s%02d_e%02d", &aIdx, &sIdx, &eIdx );

		newSam.ID   = fileIdx;
		newSam.cls  = aIdx - 1;
		newSam.sbj  = sIdx - 1;
		newSam.seq  = (sIdx-1)*2 + (eIdx-1);
		newSam.name = sampleName;

		newSamOri.ID   = fileIdx;
		newSamOri.cls  = aIdx - 1;
		newSamOri.sbj  = sIdx - 1;
		newSamOri.seq  = (sIdx-1)*2 + (eIdx-1);
		newSamOri.name = sampleName;

		sampleName  = DIR_DATASET + slash + NAME_DATASET + slash + sampleName + "_skeleton.txt";


		// Read Video
		int numFrame, numJoint, virNumFrame;
		double step;
		ifstream videoSample;
		videoSample.open(sampleName, ios::in);

		videoSample >> numFrame >> numJoint;

		step = (double(numFrame) - RANGE_ACTION_SNIPPET) / (NUM_STAGES-1);
		virNumFrame = numFrame;


		vector<double*> jointFrames;
		vector<double*> fstOdFrames;

		jointFrames.resize(virNumFrame);
		fstOdFrames.resize(virNumFrame);

		for (int frmIdx = 0; frmIdx < numFrame; ++frmIdx)
		{
			double xOri, yOri, zOri;
			jointFrames[frmIdx] = new double[numJoint*3];
			
			for (int jointIdx = 0; jointIdx < numJoint; ++jointIdx)
			{
				double x, y, z, flag;

				videoSample >> x >> y >> z >> flag;


				jointFrames[frmIdx][jointIdx*3 + 0] = x;
				jointFrames[frmIdx][jointIdx*3 + 1] = y;
				jointFrames[frmIdx][jointIdx*3 + 2] = z;

				if (jointIdx == HIP_INDEX)
				{
					xOri = x;
					yOri = y;
					zOri = z;
				}
			}


			// Centralization
			for (int jointIdx = 0; jointIdx < numJoint; ++jointIdx)
			{
				jointFrames[frmIdx][jointIdx*3 + 0] -= xOri;
				jointFrames[frmIdx][jointIdx*3 + 1] -= yOri;
				jointFrames[frmIdx][jointIdx*3 + 2] -= zOri;
			}

		}

		for (int frmIdx = numFrame; frmIdx < virNumFrame; frmIdx++)
		{
			jointFrames[frmIdx] = new double[numJoint*3];
			for (int jointIdx = 0; jointIdx < numJoint; ++jointIdx)
			{
				jointFrames[frmIdx][jointIdx*3 + 0] = jointFrames[numFrame-1][jointIdx*3 + 0];
				jointFrames[frmIdx][jointIdx*3 + 1] = jointFrames[numFrame-1][jointIdx*3 + 1];
				jointFrames[frmIdx][jointIdx*3 + 2] = jointFrames[numFrame-1][jointIdx*3 + 2];
			}

		}


		// First Order
		for (int frmIdx = 0; frmIdx < numFrame-1; ++frmIdx)
		{
			fstOdFrames[frmIdx] = new double[numJoint*3];
			for (int jointIdx = 0; jointIdx < numJoint; ++jointIdx)
			{
				fstOdFrames[frmIdx][jointIdx*3 + 0] = jointFrames[frmIdx+1][jointIdx*3 + 0] - jointFrames[frmIdx][jointIdx*3 + 0];
				fstOdFrames[frmIdx][jointIdx*3 + 1] = jointFrames[frmIdx+1][jointIdx*3 + 1] - jointFrames[frmIdx][jointIdx*3 + 1];
				fstOdFrames[frmIdx][jointIdx*3 + 2] = jointFrames[frmIdx+1][jointIdx*3 + 2] - jointFrames[frmIdx][jointIdx*3 + 2];
			}

		}

		for (int frmIdx = numFrame-1; frmIdx < virNumFrame; frmIdx++)
		{
			fstOdFrames[frmIdx] = new double[numJoint*3];
			for (int jointIdx = 0; jointIdx < numJoint; ++jointIdx)
			{
				fstOdFrames[frmIdx][jointIdx*3 + 0] = 0;
				fstOdFrames[frmIdx][jointIdx*3 + 1] = 0;
				fstOdFrames[frmIdx][jointIdx*3 + 2] = 0;
			}

		}



		// Obtain Action Snippet
		int countFrm = 0;
		for (double frmIdx = 0; frmIdx < virNumFrame - RANGE_ACTION_SNIPPET + 1; frmIdx = frmIdx + step)
		{
			int startFrmIdx = floor(frmIdx);

			double *actionSnip = new double[DIM_FEATURE];

			int counter = 0;

			// ================ Loading in Pose Information ================ //
			for (int rangeIdx = 0; rangeIdx < RANGE_ACTION_SNIPPET; rangeIdx++)
			{
				for (int jointIdx = 0; jointIdx < numJoint; jointIdx++)
				{
					actionSnip[counter++] = jointFrames[startFrmIdx + rangeIdx][jointIdx*3 + 0];
					actionSnip[counter++] = jointFrames[startFrmIdx + rangeIdx][jointIdx*3 + 1];
					actionSnip[counter++] = jointFrames[startFrmIdx + rangeIdx][jointIdx*3 + 2];				
				}
				
			}

			// Normalization
			double norm = 0;
			for (int dimIdx = 0; dimIdx < DIM_FEATURE/2; dimIdx++)
			{
				norm += actionSnip[dimIdx]*actionSnip[dimIdx];
			}

			norm = sqrt(norm);

			for (int dimIdx = 0; dimIdx < DIM_FEATURE/2; dimIdx++)
			{
				if (norm)
				{
					actionSnip[dimIdx] /= norm;
				}
			}


			// ================ Load in Fisrt Order Information ================ //
			for (int rangeIdx = 0; rangeIdx < RANGE_ACTION_SNIPPET; rangeIdx++)
			{
				for (int jointIdx = 0; jointIdx < numJoint; jointIdx++)
				{
					actionSnip[counter++] = fstOdFrames[startFrmIdx + rangeIdx][jointIdx*3 + 0];
					actionSnip[counter++] = fstOdFrames[startFrmIdx + rangeIdx][jointIdx*3 + 1];
					actionSnip[counter++] = fstOdFrames[startFrmIdx + rangeIdx][jointIdx*3 + 2];				
				}
				
			}

			// Normalization
			norm = 0;
			for (int dimIdx = DIM_FEATURE/2; dimIdx < DIM_FEATURE; dimIdx++)
			{
				norm += actionSnip[dimIdx]*actionSnip[dimIdx];
			}

			norm = sqrt(norm);

			for (int dimIdx = DIM_FEATURE/2; dimIdx < DIM_FEATURE; dimIdx++)
			{
				if (norm)
				{
					actionSnip[dimIdx] /= norm;
				}
			}

			newSam.actSnip.push_back(actionSnip);
			countFrm++;
			if (countFrm >= NUM_STAGES)
			{
				break;
			}
		}

		newSam.frm = countFrm;
		samples[fileIdx] = newSam;


		// Obtain Action Snippet
		countFrm = 0;
		for (int frmIdx = 0; frmIdx < virNumFrame - RANGE_ACTION_SNIPPET + 1; frmIdx++)
		{
			double *actionSnip = new double[DIM_FEATURE];
			int counter = 0;


			// ================ Loading in Pose Information ================ //
			for (int rangeIdx = 0; rangeIdx < RANGE_ACTION_SNIPPET; rangeIdx++)
			{
				for (int jointIdx = 0; jointIdx < numJoint; jointIdx++)
				{
					actionSnip[counter++] = jointFrames[frmIdx + rangeIdx][jointIdx*3 + 0];
					actionSnip[counter++] = jointFrames[frmIdx + rangeIdx][jointIdx*3 + 1];
					actionSnip[counter++] = jointFrames[frmIdx + rangeIdx][jointIdx*3 + 2];				
				}
				
			}

			// Normalization
			double norm = 0;
			for (int dimIdx = 0; dimIdx < DIM_FEATURE/2; dimIdx++)
			{
				norm += actionSnip[dimIdx]*actionSnip[dimIdx];
			}

			norm = sqrt(norm);

			for (int dimIdx = 0; dimIdx < DIM_FEATURE/2; dimIdx++)
			{
				if (norm)
				{
					actionSnip[dimIdx] /= norm;
				}
			}


			// ================ Load in Fisrt Order Information ================ //
			for (int rangeIdx = 0; rangeIdx < RANGE_ACTION_SNIPPET; rangeIdx++)
			{
				for (int jointIdx = 0; jointIdx < numJoint; jointIdx++)
				{
					actionSnip[counter++] = fstOdFrames[frmIdx + rangeIdx][jointIdx*3 + 0];
					actionSnip[counter++] = fstOdFrames[frmIdx + rangeIdx][jointIdx*3 + 1];
					actionSnip[counter++] = fstOdFrames[frmIdx + rangeIdx][jointIdx*3 + 2];				
				}
				
			}

			// Normalization
			norm = 0;
			for (int dimIdx = DIM_FEATURE/2; dimIdx < DIM_FEATURE; dimIdx++)
			{
				norm += actionSnip[dimIdx]*actionSnip[dimIdx];
			}

			norm = sqrt(norm);

			for (int dimIdx = DIM_FEATURE/2; dimIdx < DIM_FEATURE; dimIdx++)
			{
				if (norm)
				{
					actionSnip[dimIdx] /= norm;
				}
			}



			newSamOri.actSnip.push_back(actionSnip);
			countFrm++;
		}


		newSamOri.frm = countFrm;
		samplesOri[fileIdx] = newSamOri;

		for (int frmIdx = 0; frmIdx < virNumFrame; frmIdx++)
		{
			delete []jointFrames[frmIdx];
		}

		for (int frmIdx = 0; frmIdx < virNumFrame; frmIdx++)
		{
			delete []fstOdFrames[frmIdx];
		}
	}

	printf("%s\n", " DONE!");
}



void splitInit(vector< int > &splitList)
{
	splitList.resize(NUM_SUBJECT);
	int init[] = {1,2,3,4,5,6,7};

	for (int i = 0; i < NUM_SUBJECT; i++)
	{
		int flag = 0;
		for (int initIdx = 0; initIdx < 7; initIdx++)
		{
			if (i==init[initIdx]-1)
			{
				flag = 1;
			}
		}

		splitList[i] = flag;
	}

}

int cls2Idx(int clsIdx)
{
	int actionSet[NUM_CLASS] = ACTION_SET;

	for (int setIdx = 0; setIdx < NUM_CLASS; setIdx++)
	{
		if (actionSet[setIdx] == clsIdx+1)
		{
			return setIdx;
		}
	}

	return -1;
}

void splitOriSamples(vector< sample >       samples, 
					vector< ANNpointArray > &trainSamples, 
					vector< int >           &countTrain, 
					vector< int >    		&splitIndice,
					int 					ignoreID,
					int 					ignoreSBJ)
{
	countTrain.resize(NUM_CLASS);

	for (int cntIdx = 0; cntIdx < NUM_CLASS; ++cntIdx)
	{
		countTrain[cntIdx] = 0;
	}

	trainSamples.resize(NUM_CLASS);

	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		trainSamples[clsIdx] = annAllocPts(NUM_MAX_FEATURE, DIM_FEATURE);
	}

	// Split All Samples to Training and Testing Sample Group (in ANN data structure)
	for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
	{
		int sbjIdx = samples[sampleIdx].sbj;
		int clsIdx = samples[sampleIdx].cls;
		int numFrm = samples[sampleIdx].frm;
		int samIdx = samples[sampleIdx].ID;


		if (samIdx == ignoreID)
		{
			continue;
		}


		if (splitIndice[sbjIdx])
		{
			for (int frmIdx = 0; frmIdx < numFrm; frmIdx++)
			{
				for (int dimIdx = 0; dimIdx < DIM_FEATURE; dimIdx++)
				{
					trainSamples[ cls2Idx(clsIdx) ][ countTrain[cls2Idx(clsIdx)] ][ dimIdx ] = samples[ sampleIdx ].actSnip[ frmIdx ][ dimIdx ];
				}

				countTrain[cls2Idx(clsIdx)]++;
			}

		}
	}
}

void splitStaSamples(vector< sample > samples, 
				  	 vector< sample > &trainSamples, 
				  	 vector< sample > &testSamples,
				  	 vector< int >    &splitIndice,
				  	 vector< int >	  &countTest,
				  	 vector< int >    &countTrainS)
{
	countTest.resize(NUM_CLASS);
	countTrainS.resize(NUM_CLASS);

	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		countTest[clsIdx] = 0;
		countTrainS[clsIdx] = 0;
	}

	for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
	{
		int sbjIdx = samples[sampleIdx].sbj;
		int clsIdx = samples[sampleIdx].cls;	
		int numFrm = samples[sampleIdx].frm;


		if (splitIndice[sbjIdx])
		{
			trainSamples.push_back(samples[sampleIdx]);
			countTrainS[cls2Idx(clsIdx)]++;
		}
		else
		{
			testSamples.push_back(samples[sampleIdx]);
			countTest[cls2Idx(clsIdx)]++;
		}
	}
}

void training(vector< ANNkd_tree* >    &classTree,
			  vector< ANNpointArray >  &trainSamples,
			  vector< int > 		   &countTrain )
{
	classTree.resize(NUM_CLASS);

	for (int clsIdx = 0; clsIdx < NUM_CLASS; clsIdx++)
	{
		classTree[clsIdx] = new ANNkd_tree(trainSamples[clsIdx], countTrain[clsIdx], DIM_FEATURE);
	}
}


STM distFeat2File(vector< ANNkd_tree* > classTree, sample &sampleActSnips, vector< ANNpointArray > &trainSamples, int &prdC, int train)
{
	int numFrm = sampleActSnips.frm;

	STM STMatrix(NUM_CLASS);

	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		STMatrix[clsIdx].resize(numFrm);
	}

	vector< double > distsClass(NUM_CLASS,0);

	for (int frmIdx = 0; frmIdx < numFrm; frmIdx++)
	{
		ANNpoint     queryPt = annAllocPt(DIM_FEATURE);		// query point
		ANNidxArray	 nnIdx;					      // near neighbor indices
		ANNdistArray dists;					    // near neighbor distances

		nnIdx = new ANNidx[NUM_NEARNEIGH];
		dists = new ANNdist[NUM_NEARNEIGH];

		for (int dimIdx = 0; dimIdx < DIM_FEATURE; ++dimIdx)
		{
			queryPt[dimIdx] = sampleActSnips.actSnip[frmIdx][dimIdx];
		}

		for (int treeIdx = 0; treeIdx < NUM_CLASS; ++treeIdx)
		{
			vector< double > stageFeat(DIM_FEATURE);
			classTree[treeIdx]->annkSearch(queryPt, NUM_NEARNEIGH, nnIdx, dists, EPS);
			double sumDist = 0;

			distsClass[treeIdx] += dists[0];

			for (int dimIdx = 0; dimIdx < DIM_FEATURE; ++dimIdx)
			{
				double dist = queryPt[dimIdx] - trainSamples[treeIdx][nnIdx[0]][dimIdx];
				stageFeat[dimIdx] = dist*dist;
				sumDist += stageFeat[dimIdx];
			}

			STMatrix[treeIdx][frmIdx] = stageFeat;
		}


		delete [] nnIdx;
		delete [] dists;
		annDeallocPt(queryPt);
	}

	prdC = min_element( distsClass.begin(), distsClass.end() ) - distsClass.begin();

	ofstream listTXT;
	string slash = "/";

	if (train)
	{
		listTXT.open(OUT_DIR + slash + "train.txt", ios::app);
	}
	else
	{
		listTXT.open(OUT_DIR + slash + "test.txt", ios::app);
	}

	listTXT << sampleActSnips.name << endl;
	listTXT.close();

	// Output to Files
	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		ofstream STMFile;
		stringstream sClsIdx;  //create a stringstream
		sClsIdx << clsIdx;
		if (train)
		{	
			STMFile.open(OUT_DIR + slash + "all" + slash + sampleActSnips.name + '_' + sClsIdx.str(), ios::out);
		}
		else
		{
			STMFile.open(OUT_DIR + slash + "all" + slash + sampleActSnips.name + '_' + sClsIdx.str(), ios::out);
		}
		
		for (int dimIdx = 0; dimIdx < DIM_FEATURE; ++dimIdx)
		{
			for (int stageIdx = 0; stageIdx < numFrm; ++stageIdx)
			{
				STMFile << STMatrix[clsIdx][stageIdx][dimIdx] << ' ';
			}
			STMFile << endl;
		}
		
		STMFile.close();
	}

	return STMatrix;
}



int main()
{
	vector< int >  splitIndice;   // 1 indicates trainning samples
	vector< sample > sampleStages;      // length : number of samples
	vector< sample > 	sampleOri;      // length : number of samples

	
	// Load in Training and Testing Samples
	loadOriFeature(sampleStages, sampleOri);

	// Initialization for split index
	splitInit(splitIndice);

	/* ==================== Testing Splits ==================== */

	vector< int > countTest;   // Count Video Sample
	vector< int > countTrainS;
	vector< int > countTrain;  // Count Action Snippet Sample

	vector< ANNkd_tree* >     classTree;
	vector< ANNpointArray >   trainSamplesSnip;

	vector< sample > 	testSampleStages;      // length : number of samples
	vector< sample >   trainSampleStages;      // length : number of samples

	// Split Stage-Samples
	splitStaSamples(sampleStages, trainSampleStages, testSampleStages, splitIndice, countTest, countTrainS);

	cout << endl << " #####  NBNN Accuracy: ";

	/* ================== Generating Training Samples ================== */
	vector< double > corCountTrain(NUM_CLASS,0);

	for (int trainIdx = 0; trainIdx < trainSampleStages.size(); ++trainIdx)
	{

		int ID  = trainSampleStages[trainIdx].ID;
		int CLS = trainSampleStages[trainIdx].cls;
		int SBJ = trainSampleStages[trainIdx].sbj;

		int numFrm = trainSampleStages[trainIdx].frm;


		// Split Original Samples
		splitOriSamples(sampleOri, trainSamplesSnip, countTrain, splitIndice, ID, -1);

		// Trees Construction
		training(classTree, trainSamplesSnip, countTrain);
		
		int prdC;

		STM trainSTM = distFeat2File(classTree, trainSampleStages[trainIdx], trainSamplesSnip, prdC, 1);


		if (prdC == cls2Idx(CLS))
		{
			corCountTrain[cls2Idx(CLS)]++;
		}

		// Release Memory
		for (int clsIdx = 0; clsIdx < NUM_CLASS; clsIdx++)
		{
			annDeallocPts(trainSamplesSnip[clsIdx]);
			delete classTree[clsIdx];
		}

		countTrain.clear();
		classTree.clear();
		trainSamplesSnip.clear();
		annClose();


	}


	double corrSumTrain = 0;
	double testSumTrain = 0;

	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		corrSumTrain += corCountTrain[clsIdx];
		testSumTrain += countTrainS[clsIdx];
	}
	cout.precision(3);
	cout << " Train:" << corrSumTrain/testSumTrain << " ";

	
	/* ================== Generating Testing Samples ================== */
	// Split Original Samples
	splitOriSamples(sampleOri, trainSamplesSnip, countTrain, splitIndice, -1, -1);

	// Trees Construction
	training(classTree, trainSamplesSnip, countTrain);

	vector< double > corCount(NUM_CLASS,0);
	for (int testIdx = 0; testIdx < testSampleStages.size(); testIdx++)
	{
		int clsLbl  = testSampleStages[testIdx].cls;
		int numFrm  = testSampleStages[testIdx].frm;
		int prdC = 0;

		STM testSTM = distFeat2File(classTree, testSampleStages[testIdx], trainSamplesSnip, prdC, 0);

		if (prdC == cls2Idx(clsLbl))
		{
			corCount[cls2Idx(clsLbl)]++;
		}
	}


	double corrSum = 0;
	double testSum = 0;

	for (int clsIdx = 0; clsIdx < NUM_CLASS; ++clsIdx)
	{
		corrSum += corCount[clsIdx];
		testSum += countTest[clsIdx];
	}
	cout.precision(3);
	cout << " Test:" << corrSum/testSum << "  #####" << endl << endl;



	// Release Memory
	for (int clsIdx = 0; clsIdx < NUM_CLASS; clsIdx++)
	{
		annDeallocPts(trainSamplesSnip[clsIdx]);
		delete classTree[clsIdx];
	}

	countTest.clear();
	countTrain.clear();
	classTree.clear();
	trainSamplesSnip.clear();
	annClose();


	

	return 0;
}