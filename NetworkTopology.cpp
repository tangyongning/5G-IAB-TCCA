// NetworkTopology.cpp
#include "NetworkTopology.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include <cmath>

NetworkTopology::NetworkTopology(int n) : numNodes(n) {
    rng.seed(chrono::steady_clock::now().time_since_epoch().count());
}

void NetworkTopology::generateHierarchicalTopology() {
    nodes.clear();
    edges.clear();
    adjacencyMatrix.clear();
    
    // Create hierarchical structure: Core -> Edge -> Base Stations -> UEs
    int coreNodes = max(1, numNodes / 50);
    int edgeNodes = max(1, numNodes / 10);
    int bsNodes = max(1, numNodes / 3);
    int ueNodes = numNodes - coreNodes - edgeNodes - bsNodes;
    
    int nodeId = 0;
    
    // Core layer
    for (int i = 0; i < coreNodes; i++) {
        nodes.emplace_back(nodeId++, "core", 10000.0);
    }
    
    // Edge layer
    for (int i = 0; i < edgeNodes; i++) {
        nodes.emplace_back(nodeId++, "edge", 5000.0);
        // Connect to core
        int parentCore = i % coreNodes;
        edges.emplace_back(parentCore, nodeId - 1, 1000.0, 2.0);
        adjacencyMatrix[{parentCore, nodeId - 1}] = 1.0;
        nodes[parentCore].childNodes.push_back(nodeId - 1);
        nodes[nodeId - 1].parentNodes.push_back(parentCore);
    }
    
    // Base station layer
    for (int i = 0; i < bsNodes; i++) {
        nodes.emplace_back(nodeId++, "base_station", 1000.0);
        // Connect to edge
        int parentEdge = i % edgeNodes;
        edges.emplace_back(edgeNodes + parentEdge, nodeId - 1, 500.0, 5.0);
        adjacencyMatrix[{edgeNodes + parentEdge, nodeId - 1}] = 1.0;
        nodes[edgeNodes + parentEdge].childNodes.push_back(nodeId - 1);
        nodes[nodeId - 1].parentNodes.push_back(edgeNodes + parentEdge);
    }
    
    // UE layer
    for (int i = 0; i < ueNodes; i++) {
        nodes.emplace_back(nodeId++, "ue", 100.0);
        // Connect to base station
        int parentBS = coreNodes + edgeNodes + (i % bsNodes);
        edges.emplace_back(parentBS, nodeId - 1, 100.0, 10.0, 0.05);
        adjacencyMatrix[{parentBS, nodeId - 1}] = 1.0;
        nodes[parentBS].childNodes.push_back(nodeId - 1);
        nodes[nodeId - 1].parentNodes.push_back(parentBS);
    }
    
    // Add some cross-layer connections for realism
    uniform_real_distribution<double> prob(0.0, 1.0);
    for (int i = 0; i < edgeNodes; i++) {
        for (int j = 0; j < coreNodes; j++) {
            if (i != j && prob(rng) < 0.3) {
                int edgeId = coreNodes + i;
                if (!hasEdge(j, edgeId)) {
                    edges.emplace_back(j, edgeId, 800.0, 3.0);
                    adjacencyMatrix[{j, edgeId}] = 0.8;
                    nodes[j].childNodes.push_back(edgeId);
                    nodes[edgeId].parentNodes.push_back(j);
                }
            }
        }
    }
    
    cout << "Generated hierarchical topology with " << numNodes << " nodes" << endl;
    cout << "  Core: " << coreNodes << ", Edge: " << edgeNodes 
         << ", BS: " << bsNodes << ", UE: " << ueNodes << endl;
}

void NetworkTopology::generateIABTopology() {
    nodes.clear();
    edges.clear();
    adjacencyMatrix.clear();
    
    // IAB-specific topology: Donor -> IAB Nodes (multi-hop) -> UEs
    int numDonors = max(1, numNodes / 100);
    int numIABNodes = max(1, numNodes / 5);
    int numUEs = numNodes - numDonors - numIABNodes;
    
    int nodeId = 0;
    
    // IAB Donors
    for (int i = 0; i < numDonors; i++) {
        nodes.emplace_back(nodeId++, "iab_donor", 10000.0);
    }
    
    // IAB Nodes (multi-hop)
    vector<int> iabNodeIds;
    for (int i = 0; i < numIABNodes; i++) {
        nodes.emplace_back(nodeId++, "iab_node", 2000.0);
        iabNodeIds.push_back(nodeId - 1);
        
        // Connect to donor or parent IAB node
        if (i < numDonors) {
            // Direct connection to donor
            edges.emplace_back(i, nodeId - 1, 1000.0, 2.0, 0.02);
            adjacencyMatrix[{i, nodeId - 1}] = 1.0;
            nodes[i].childNodes.push_back(nodeId - 1);
            nodes[nodeId - 1].parentNodes.push_back(i);
        } else {
            // Multi-hop: connect to existing IAB node
            int parentIdx = uniform_int_distribution<int>(0, iabNodeIds.size() - 2)(rng);
            int parentId = iabNodeIds[parentIdx];
            edges.emplace_back(parentId, nodeId - 1, 500.0, 5.0, 0.03);
            adjacencyMatrix[{parentId, nodeId - 1}] = 1.0;
            nodes[parentId].childNodes.push_back(nodeId - 1);
            nodes[nodeId - 1].parentNodes.push_back(parentId);
        }
    }
    
    // UEs
    for (int i = 0; i < numUEs; i++) {
        nodes.emplace_back(nodeId++, "ue", 100.0);
        
        // Connect to IAB node or donor
        int parentIdx = uniform_int_distribution<int>(0, numDonors + numIABNodes - 1)(rng);
        int parentId = (parentIdx < numDonors) ? parentIdx : 
                       iabNodeIds[parentIdx - numDonors];
        
        double latency = (parentId < numDonors) ? 10.0 : 15.0;
        edges.emplace_back(parentId, nodeId - 1, 100.0, latency, 0.05);
        adjacencyMatrix[{parentId, nodeId - 1}] = 1.0;
        nodes[parentId].childNodes.push_back(nodeId - 1);
        nodes[nodeId - 1].parentNodes.push_back(parentId);
    }
    
    cout << "Generated IAB topology with " << numNodes << " nodes" << endl;
    cout << "  Donors: " << numDonors << ", IAB Nodes: " << numIABNodes 
         << ", UEs: " << numUEs << endl;
}

void NetworkTopology::generateRandomTopology(double connectionProb) {
    nodes.clear();
    edges.clear();
    adjacencyMatrix.clear();
    
    // Create nodes
    for (int i = 0; i < numNodes; i++) {
        string type = (i < numNodes / 10) ? "core" : 
                     (i < numNodes / 3) ? "edge" : "base_station";
        double cap = (type == "core") ? 10000.0 : 
                    (type == "edge") ? 5000.0 : 1000.0;
        nodes.emplace_back(i, type, cap);
    }
    
    // Create random edges
    uniform_real_distribution<double> prob(0.0, 1.0);
    uniform_real_distribution<double> capacity(100.0, 1000.0);
    uniform_real_distribution<double> latency(1.0, 20.0);
    
    for (int i = 0; i < numNodes; i++) {
        for (int j = i + 1; j < numNodes; j++) {
            if (prob(rng) < connectionProb) {
                // Ensure directionality (lower ID -> higher ID)
                edges.emplace_back(i, j, capacity(rng), latency(rng));
                adjacencyMatrix[{i, j}] = 1.0;
                nodes[i].childNodes.push_back(j);
                nodes[j].parentNodes.push_back(i);
            }
        }
    }
    
    // Ensure connectivity using BFS
    vector<bool> visited(numNodes, false);
    queue<int> q;
    q.push(0);
    visited[0] = true;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v : nodes[u].childNodes) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    
    // Connect unvisited nodes
    for (int i = 0; i < numNodes; i++) {
        if (!visited[i]) {
            int parent = uniform_int_distribution<int>(0, i - 1)(rng);
            edges.emplace_back(parent, i, 500.0, 10.0);
            adjacencyMatrix[{parent, i}] = 1.0;
            nodes[parent].childNodes.push_back(i);
            nodes[i].parentNodes.push_back(parent);
            visited[i] = true;
        }
    }
    
    cout << "Generated random topology with " << numNodes << " nodes and " 
         << edges.size() << " edges" << endl;
}

vector<int> NetworkTopology::getParentNodes(int nodeId) const {
    if (nodeId >= 0 && nodeId < numNodes) {
        return nodes[nodeId].parentNodes;
    }
    return {};
}

vector<int> NetworkTopology::getChildNodes(int nodeId) const {
    if (nodeId >= 0 && nodeId < numNodes) {
        return nodes[nodeId].childNodes;
    }
    return {};
}

bool NetworkTopology::hasEdge(int from, int to) const {
    return adjacencyMatrix.find({from, to}) != adjacencyMatrix.end();
}

double NetworkTopology::getEdgeWeight(int from, int to) const {
    auto it = adjacencyMatrix.find({from, to});
    if (it != adjacencyMatrix.end()) {
        return it->second;
    }
    return 0.0;
}

void NetworkTopology::exportToGraphML(const string& filename) {
    ofstream file(filename);
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n";
    file << "  <graph id=\"G\" edgedefault=\"directed\">\n";
    
    // Nodes
    for (const auto& node : nodes) {
        file << "    <node id=\"" << node.id << "\">\n";
        file << "      <data key=\"type\">" << node.type << "</data>\n";
        file << "      <data key=\"capacity\">" << node.processingCapacity << "</data>\n";
        file << "    </node>\n";
    }
    
    // Edges
    for (const auto& edge : edges) {
        file << "    <edge source=\"" << edge.from 
             << "\" target=\"" << edge.to << "\">\n";
        file << "      <data key=\"capacity\">" << edge.capacity << "</data>\n";
        file << "      <data key=\"latency\">" << edge.latency << "</data>\n";
        file << "    </edge>\n";
    }
    
    file << "  </graph>\n</graphml>\n";
    file.close();
    
    cout << "Topology exported to " << filename << endl;
}

void NetworkTopology::printTopology() const {
    cout << "\n=== Network Topology ===" << endl;
    cout << "Nodes: " << numNodes << ", Edges: " << edges.size() << endl;
    
    cout << "\nNode Types:" << endl;
    map<string, int> typeCount;
    for (const auto& node : nodes) {
        typeCount[node.type]++;
    }
    for (const auto& [type, count] : typeCount) {
        cout << "  " << type << ": " << count << endl;
    }
    
    cout << "\nSample Edges (first 10):" << endl;
    int count = 0;
    for (const auto& edge : edges) {
        if (count++ >= 10) break;
        cout << "  " << edge.from << " -> " << edge.to 
             << " (cap: " << edge.capacity 
             << ", lat: " << edge.latency << ")" << endl;
    }
}
