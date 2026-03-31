// TCCAFramework.h
#ifndef TCCA_FRAMEWORK_H
#define TCCA_FRAMEWORK_H

#include "NetworkTopology.h"
#include "QoSMetrics.h"
#include <vector>
#include <map>
#include <random>

struct NodeState {
    vector<double> hiddenState;
    double faultProbability;
    double trustScore;
    double uncertainty;
    
    NodeState(int hiddenDim = 64) 
        : hiddenState(hiddenDim, 0.0), faultProbability(0.0), 
          trustScore(0.0), uncertainty(0.0) {}
};

class TCCAFramework {
private:
    NetworkTopology& topology;
    const vector<vector<QoSObservation>>& observations;
    
    int hiddenDim;
    int numTimeSteps;
    double alpha;  // Local fault weight
    double beta;   // Upstream influence weight
    double lambda; // Regularization parameter
    
    map<int, NodeState> nodeStates;
    map<pair<int,int>, double> attentionWeights;
    
    mt19937 rng;
    
    // Topology-constrained causal inference
    double computeUpstreamInfluence(int nodeId, int timeStep);
    double computeAttentionWeight(int fromNode, int toNode, int timeStep);
    
    // Reliability-oriented trust metric
    double computeStructuralConsistency(int nodeId, int timeStep);
    double computeObservationalSupport(int nodeId, int timeStep);
    double computeTemporalStability(int nodeId, int timeStep);
    double computeUncertainty(int nodeId, int timeStep);
    
    // Temporal graph learning
    void updateHiddenState(int nodeId, int timeStep);
    vector<double> gruUpdate(const vector<double>& prevState,
                             const vector<double>& input,
                             const vector<double>& aggregatedInput);
    
    bool useTopologyConstraint;
    bool useTrustMetric;
    bool useTemporalModel;
    
public:
    TCCAFramework(NetworkTopology& topo, 
                  const vector<vector<QoSObservation>>& obs,
                  bool topology = true, bool trust = true, bool temporal = true);
    
    map<int, double> localizeFaults();
    
    void trainModel(int epochs = 50, double learningRate = 0.001);
    
    void exportResults(const string& filename, 
                       const map<int, double>& faultScores);
};

#endif
