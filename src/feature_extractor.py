import numpy as np
from src.wf_config import CONFIG

def extract_general_features(packet_sequence):
    """Extract general features from packet sequence"""
    total_packets = len(packet_sequence)
    total_time = packet_sequence['timestamp'].iloc[-1] - packet_sequence['timestamp'].iloc[0]
    outgoing = (packet_sequence['direction'] == 1).sum()
    incoming = total_packets - outgoing
    return [total_packets, total_time, outgoing, incoming]

def extract_burst_features(packet_sequence):
    """Extract burst-related features"""
    direction_changes = (packet_sequence['direction'].diff() != 0).cumsum()
    burst_sizes = packet_sequence.groupby(direction_changes)['direction'].apply(
        lambda x: (x == 1).sum() if x.iloc[0] == 1 else -(x == -1).sum())
    
    outgoing_bursts = [b for b in burst_sizes if b > 0]
    incoming_bursts = [abs(b) for b in burst_sizes if b < 0]
    
    features = []
    features.append(len(outgoing_bursts))  # Number of outgoing bursts
    features.append(np.mean(outgoing_bursts) if outgoing_bursts else 0)  # Mean outgoing burst size
    features.append(max(outgoing_bursts) if outgoing_bursts else 0)  # Max outgoing burst size
    features.append(np.mean(incoming_bursts) if incoming_bursts else 0)  # Mean incoming burst size
    
    return features

def extract_initial_packets(packet_sequence, n=20):
    """Extract features from first n packets"""
    first_n = packet_sequence['direction'].head(n)
    features = first_n.tolist()
    if len(features) < n:
        features.extend([0] * (n - len(features)))  # Pad with zeros
    return features

def extract_all_features(packet_sequence):
    """Extract all features for a packet sequence"""
    features = []
    
    # General features
    features.extend(extract_general_features(packet_sequence))
    
    # Burst features
    features.extend(extract_burst_features(packet_sequence))
    
    # Initial packets
    features.extend(extract_initial_packets(packet_sequence))
    
    # Add more feature types as needed...
    
    return np.array(features)
