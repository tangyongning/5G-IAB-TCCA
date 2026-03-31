// FaultModel.cpp
#include "FaultModel.h"
#include <iostream>
#include <algorithm>

FaultModel::FaultModel(NetworkTopology& topo, int numFaults) 
    : topology(topo) {
    rng.seed(chrono::steady_clock::now().time_since_epoch().count());
}

void FaultModel::injectRandomFaults() {
    faults.clear();
    groundTruth.clear();
    
    uniform_int_distribution<int> nodeDist(0, topology.getNodes().size() - 1);
    uniform_int_distribution<int> typeDist(0, 3);
    uniform_real_distribution<double> severityDist(0.3, 1.0);
    uniform_int_distribution<int> durationDist(10, 50);
    
    const auto& nodes = topology.getNodes();
    
    for (int i = 0; i < (int)nodes.size() / 20; i++) {
        int nodeId = nodeDist(rng);
        int faultType = typeDist(rng);
        double severity = severityDist(rng);
        int startTime = 0;
        int duration = durationDist(rng);
        
        faults.emplace_back(nodeId, faultType, severity, startTime, duration);
        groundTruth[nodeId] = true;
    }
    
    cout << "Injected " << faults.size() << " random faults" << endl;
}

void FaultModel::injectCascadingFaults() {
    faults.clear();
    groundTruth.clear();
    
    // Start with a core node failure
    const auto& nodes = topology.getNodes();
    int coreNode = -1;
    
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].type == "core" || nodes[i].type == "iab_donor") {
            coreNode = i;
            break;
        }
    }
    
    if (coreNode != -1) {
        faults.emplace_back(coreNode, 0, 1.0, 0, 100);
        groundTruth[coreNode] = true;
        
        // Propagate to children
        auto children = topology.getChildNodes(coreNode);
        for (int child : children) {
            if (uniform_real_distribution<double>(0, 1)(rng) < 0.7) {
                faults.emplace_back(child, 1, 0.8, 2, 80);
                groundTruth[child] = true;
            }
        }
    }
    
    cout << "Injected cascading fault scenario" << endl;
}

void FaultModel::injectIABSpecificFaults() {
    faults.clear();
    groundTruth.clear();
    
    const auto& nodes = topology.getNodes();
    
    // IAB-specific faults: beam failure, backhaul blockage
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].type == "iab_node") {
            if (uniform_real_distribution<double>(0, 1)(rng) < 0.15) {
                // Beam failure
                faults.emplace_back(i, 3, 0.9, 0, 30);
                groundTruth[i] = true;
            }
        }
    }
    
    cout << "Injected " << faults.size() << " IAB-specific faults" << endl;
}

QoSDegradation FaultModel::calculateFaultImpact(const Fault& fault) {
    QoSDegradation degradation;
    
    switch (fault.faultType) {
        case 0:  // Node failure
            degradation.latencyIncrease = 1000.0 * fault.severity;
            degradation.packetLossIncrease = 90.0 * fault.severity;
            degradation.throughputDecrease = 95.0 * fault.severity;
            degradation.availabilityDecrease = 100.0 * fault.severity;
            break;
            
        case 1:  // Congestion
            degradation.latencyIncrease = 200.0 * fault.severity;
            degradation.packetLossIncrease = 30.0 * fault.severity;
            degradation.throughputDecrease = 50.0 * fault.severity;
            degradation.availabilityDecrease = 10.0 * fault.severity;
            break;
            
        case 2:  // Link degradation
            degradation.latencyIncrease = 150.0 * fault.severity;
            degradation.packetLossIncrease = 40.0 * fault.severity;
            degradation.throughputDecrease = 60.0 * fault.severity;
            degradation.availabilityDecrease = 20.0 * fault.severity;
            break;
            
        case 3:  // Beam failure (IAB-specific)
            degradation.latencyIncrease = 300.0 * fault.severity;
            degradation.packetLossIncrease = 50.0 * fault.severity;
            degradation.throughputDecrease = 70.0 * fault.severity;
            degradation.availabilityDecrease = 40.0 * fault.severity;
            break;
    }
    
    return degradation;
}

void FaultModel::propagateFault(const Fault& fault, 
                                 map<int, QoSDegradation>& degradation) {
    auto localImpact = calculateFaultImpact(fault);
    degradation[fault.nodeId] = localImpact;
    
    // Propagate to children with attenuation
    vector<int> queue = {fault.nodeId};
    map<int, bool> visited;
    visited[fault.nodeId] = true;
    
    double attenuation = 0.7;
    int hop = 0;
    
    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());
        
        auto children = topology.getChildNodes(current);
        for (int child : children) {
            if (!visited[child]) {
                visited[child] = true;
                queue.push_back(child);
                
                // Attenuated impact
                double factor = pow(attenuation, hop + 1);
                degradation[child].latencyIncrease += 
                    localImpact.latencyIncrease * factor;
                degradation[child].packetLossIncrease += 
                    localImpact.packetLossIncrease * factor * 0.8;
                degradation[child].throughputDecrease += 
                    localImpact.throughputDecrease * factor * 0.9;
                degradation[child].availabilityDecrease += 
                    localImpact.availabilityDecrease * factor * 0.5;
            }
        }
        hop++;
    }
}

void FaultModel::updateFaults(int currentTime) {
    for (auto& fault : faults) {
        if (currentTime < fault.startTime) {
            fault.active = false;
        } else if (currentTime >= fault.startTime + fault.duration) {
            fault.active = false;
        } else {
            fault.active = true;
        }
    }
}

map<int, QoSDegradation> FaultModel::getCurrentDegradation(int currentTime) {
    updateFaults(currentTime);
    map<int, QoSDegradation> degradation;
    
    for (const auto& fault : faults) {
        if (fault.active) {
            propagateFault(fault, degradation);
        }
    }
    
    return degradation;
}

void FaultModel::printFaults() const {
    cout << "\n=== Active Faults ===" << endl;
    const char* typeNames[] = {"Node Failure", "Congestion", 
                                "Link Degradation", "Beam Failure"};
    
    for (const auto& fault : faults) {
        if (fault.active) {
            cout << "Node " << fault.nodeId << ": " 
                 << typeNames[fault.faultType]
                 << " (severity: " << fault.severity 
                 << ", duration: " << fault.duration << ")" << endl;
        }
    }
}
