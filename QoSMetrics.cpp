// QoSMetrics.cpp
#include "QoSMetrics.h"
#include <fstream>
#include <iostream>
#include <cmath>

QoSMetrics::QoSMetrics(NetworkTopology& topo, FaultModel& fault, double noise)
    : topology(topo), faultModel(fault), noiseLevel(noise) {
    rng.seed(chrono::steady_clock::now().time_since_epoch().count());
    initializeBaselineMetrics();
}

void QoSMetrics::initializeBaselineMetrics() {
    const auto& nodes = topology.getNodes();
    
    for (const auto& node : nodes) {
        double baseLatency, basePL, baseTP, baseAvail;
        
        if (node.type == "core" || node.type == "iab_donor") {
            baseLatency = 5.0;
            basePL = 0.01;
            baseTP = 1000.0;
            baseAvail = 99.99;
        } else if (node.type == "edge") {
            baseLatency = 10.0;
            basePL = 0.05;
            baseTP = 500.0;
            baseAvail = 99.9;
        } else if (node.type == "base_station" || node.type == "iab_node") {
            baseLatency = 20.0;
            basePL = 0.1;
            baseTP = 200.0;
            baseAvail = 99.5;
        } else {  // UE
            baseLatency = 50.0;
            basePL = 0.5;
            baseTP = 100.0;
            baseAvail = 99.0;
        }
        
        baselineMetrics[node.id] = BaselineQoS(baseLatency, basePL, baseTP, baseAvail);
    }
}

vector<vector<QoSObservation>> QoSMetrics::generateTimeSeries(int numTimeSteps) {
    vector<vector<QoSObservation>> observations;
    const auto& nodes = topology.getNodes();
    
    for (int t = 0; t < numTimeSteps; t++) {
        vector<QoSObservation> timeStepObs;
        auto degradation = faultModel.getCurrentDegradation(t);
        
        for (const auto& node : nodes) {
            QoSDegradation deg;
            if (degradation.find(node.id) != degradation.end()) {
                deg = degradation[node.id];
            }
            
            auto obs = generateObservation(node.id, t, deg);
            addNoise(obs);
            timeStepObs.push_back(obs);
        }
        
        observations.push_back(timeStepObs);
    }
    
    cout << "Generated time series with " << numTimeSteps 
         << " time steps and " << nodes.size() << " nodes" << endl;
    
    return observations;
}

QoSObservation QoSMetrics::generateObservation(int nodeId, int timestamp,
                                                const QoSDegradation& degradation) {
    auto baseline = baselineMetrics[nodeId];
    
    QoSObservation obs;
    obs.nodeId = nodeId;
    obs.timestamp = timestamp;
    
    // Apply degradation
    obs.latency = baseline.latency + degradation.latencyIncrease;
    obs.packetLoss = min(100.0, baseline.packetLoss + degradation.packetLossIncrease);
    obs.throughput = max(0.0, baseline.throughput - degradation.throughputDecrease);
    obs.availability = max(0.0, baseline.availability - degradation.availabilityDecrease);
    
    return obs;
}

void QoSMetrics::addNoise(QoSObservation& obs) {
    normal_distribution<double> noise(0.0, noiseLevel);
    
    obs.latency *= (1.0 + noise(rng));
    obs.packetLoss *= (1.0 + noise(rng));
    obs.throughput *= (1.0 + noise(rng));
    obs.availability *= (1.0 + noise(rng));
    
    // Ensure non-negative
    obs.latency = max(0.0, obs.latency);
    obs.packetLoss = max(0.0, obs.packetLoss);
    obs.throughput = max(0.0, obs.throughput);
    obs.availability = max(0.0, obs.availability);
}

BaselineQoS QoSMetrics::getBaselineQoSDegradation() {
    double totalLat = 0, totalPL = 0, totalTP = 0, totalAvail = 0;
    int count = 0;
    
    for (const auto& [nodeId, baseline] : baselineMetrics) {
        totalLat += baseline.latency;
        totalPL += baseline.packetLoss;
        totalTP += baseline.throughput;
        totalAvail += baseline.availability;
        count++;
    }
    
    return BaselineQoS(totalLat/count, totalPL/count, totalTP/count, totalAvail/count);
}

BaselineQoS QoSMetrics::calculatePostMitigationQoS(const map<int, double>& faultScores) {
    // Simulate QoS after mitigating top-k faults
    double totalLat = 0, totalPL = 0, totalTP = 0, totalAvail = 0;
    int count = 0;
    
    for (const auto& [nodeId, baseline] : baselineMetrics) {
        double mitigationFactor = 1.0;
        
        // If node is identified as faulty, apply mitigation
        if (faultScores.find(nodeId) != faultScores.end() && 
            faultScores.at(nodeId) > 0.7) {
            mitigationFactor = 0.3;  // 70% improvement after mitigation
        }
        
        totalLat += baseline.latency * mitigationFactor;
        totalPL += baseline.packetLoss * mitigationFactor;
        totalTP += baseline.throughput * (2.0 - mitigationFactor);
        totalAvail += baseline.availability * (2.0 - mitigationFactor);
        count++;
    }
    
    return BaselineQoS(totalLat/count, totalPL/count, totalTP/count, totalAvail/count);
}

void QoSMetrics::exportToCSV(const string& filename,
                              const vector<vector<QoSObservation>>& observations) {
    ofstream file(filename);
    file << "timestamp,node_id,latency,packet_loss,throughput,availability\n";
    
    for (const auto& timeStep : observations) {
        for (const auto& obs : timeStep) {
            file << obs.timestamp << ","
                 << obs.nodeId << ","
                 << obs.latency << ","
                 << obs.packetLoss << ","
                 << obs.throughput << ","
                 << obs.availability << "\n";
        }
    }
    
    file.close();
    cout << "QoS observations exported to " << filename << endl;
}
