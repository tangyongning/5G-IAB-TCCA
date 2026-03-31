// QoSMetrics.h
#ifndef QOS_METRICS_H
#define QOS_METRICS_H

#include "NetworkTopology.h"
#include "FaultModel.h"
#include <vector>
#include <map>
#include <random>

struct QoSObservation {
    double latency;
    double packetLoss;
    double throughput;
    double availability;
    int timestamp;
    int nodeId;
    
    QoSObservation(int node = 0, int time = 0, 
                   double lat = 10.0, double pl = 0.1, 
                   double tp = 100.0, double av = 99.9)
        : nodeId(node), timestamp(time), latency(lat), 
          packetLoss(pl), throughput(tp), availability(av) {}
};

struct BaselineQoS {
    double latency;
    double packetLoss;
    double throughput;
    double availability;
    
    BaselineQoS(double lat = 10.0, double pl = 0.1, 
                double tp = 100.0, double av = 99.9)
        : latency(lat), packetLoss(pl), 
          throughput(tp), availability(av) {}
};

class QoSMetrics {
private:
    NetworkTopology& topology;
    FaultModel& faultModel;
    double noiseLevel;
    mt19937 rng;
    
    map<int, BaselineQoS> baselineMetrics;
    
    void initializeBaselineMetrics();
    QoSObservation generateObservation(int nodeId, int timestamp,
                                        const QoSDegradation& degradation);
    
public:
    QoSMetrics(NetworkTopology& topo, FaultModel& fault, double noise = 0.05);
    
    vector<vector<QoSObservation>> generateTimeSeries(int numTimeSteps);
    BaselineQoS getBaselineQoSDegradation();
    BaselineQoS calculatePostMitigationQoS(const map<int, double>& faultScores);
    
    void addNoise(QoSObservation& obs);
    void exportToCSV(const string& filename, 
                     const vector<vector<QoSObservation>>& observations);
};

#endif
