# Configuration parameters matching the paper
CONFIG = {
    "DATA_PATH": "./data",          # Path to dataset
    "MONITORED_SITES": 100,         # Number of monitored websites
    "INSTANCES_PER_SITE": 90,       # Instances per monitored site
    "NON_MONITORED_SITES": 9000,    # Number of non-monitored sites
    "FEATURE_SET_SIZE": 4000,       # Approximate number of features
    "K_NEIGHBORS": 5,               # k for k-NN
    "K_RECO": 5,                    # k for weight recommendation
    "N_ROUNDS": 10,               # Training rounds
    "TEST_SIZE": 0.3,               # Test set size
    "RANDOM_SEED": 42
}