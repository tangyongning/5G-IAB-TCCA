# oai_e2_adapter.py
import zmq
import json
import time
from datetime import datetime

class OAIE2Adapter:
    """Adapt OAI logs to E2-like telemetry stream for TCCA"""
    
    def __init__(self, oai_log_path, tcca_endpoint="tcp://localhost:5555"):
        self.oai_log_path = oai_log_path
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(tcca_endpoint)
        
    def parse_oai_kpi(self, log_line):
        """Parse OAI log line to standardized KPI dict"""
        # Example OAI log: 
        # "[MAC] UE 1: UL throughput=45.2 Mbps, HARQ NACK=3/10, RLC buffer=1250 bytes"
        kpi = {}
        if "throughput" in log_line:
            kpi["throughput_mbps"] = float(log_line.split("throughput=")[1].split()[0])
        if "HARQ NACK" in log_line:
            parts = log_line.split("HARQ NACK=")[1].split("/")[0]
            kpi["harq_nack_rate"] = float(parts) / 10.0  # Normalize
        if "RLC buffer" in log_line:
            kpi["rlc_buffer_bytes"] = int(log_line.split("RLC buffer=")[1].split()[0])
        return kpi
    
    def stream_telemetry(self, interval_ms=100):
        """Continuously parse OAI logs and publish to TCCA"""
        with open(self.oai_log_path, "r") as f:
            f.seek(0, 2)  # Start at end
            
            while True:
                line = f.readline()
                if line:
                    kpi = self.parse_oai_kpi(line.strip())
                    if kpi:
                        message = {
                            "timestamp": datetime.now().timestamp(),
                            "node_id": self._extract_node_id(line),
                            "kpi": kpi
                        }
                        self.socket.send_json(message)
                else:
                    time.sleep(interval_ms / 1000.0)
    
    def _extract_node_id(self, log_line):
        # Extract node identifier from OAI log format
        if "Donor" in log_line:
            return "donor_0"
        elif "IAB" in log_line:
            return log_line.split("IAB-")[1].split(":")[0]
        return "unknown"
