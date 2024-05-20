# Real-Time Data Filtering from the LHC using Geometric Deep Learning

## Project Overview

This project aims to develop a cutting-edge model that can filter data from the Large Hadron Collider (LHC) in real-time using geometric deep learning techniques. The model will leverage the latest advancements in machine learning, particularly graph neural networks (GNNs), to efficiently process and analyze complex collision data, enabling high-precision identification of events of interest.

## Background and Motivation

High-energy physics experiments at the LHC generate enormous amounts of data, with millions of particle collisions occurring every second. Filtering this data in real-time to identify significant events, such as Higgs boson decays, is a critical task that poses substantial challenges. Traditional methods struggle to keep up with the data volume and complexity, necessitating the development of more advanced, scalable solutions.

Geometric deep learning, specifically graph neural networks, offers a promising approach to address these challenges. By modeling the relationships between particles in a collision event as a graph, GNNs can effectively capture the intricate dependencies and interactions within the data, leading to more accurate and efficient filtering.

## Project Goals

1. **Develop a Real-Time Filtering Model**: Create a robust GNN-based model capable of processing LHC data in real-time to filter out irrelevant events and identify those of significant interest.
2. **Achieve High Precision and Recall**: Ensure the model achieves high precision and recall rates, minimizing false positives and false negatives.
3. **Scalability and Efficiency**: Optimize the model for scalability and efficiency, ensuring it can handle the massive data throughput from the LHC.
4. **Integration with LHC Data Pipelines**: Ensure the model can be seamlessly integrated into existing LHC data processing pipelines for real-time application.

## Dataset

### Dataset Components

- **Simulated Events**: Fully simulated events of proton-proton collisions at the LHC, provided by the CMS Collaboration or similar sources. Includes both signal events (e.g., Higgs boson decays) and background events (e.g., ordinary jets from quark and gluon interactions).
- **Particle Features**: Detailed information about the charged particles within each jet, including:
  - Transverse momentum (pT)
  - Energy (E)
  - Pseudorapidity (η)
  - Azimuthal angle (φ)
  - Impact parameters (transverse d0, longitudinal z0)
  - Track quality indicators
  - Covariance matrix entries for the track parameters
  - Particle identification (PID) flags if available
- **Secondary Vertices (SVs)**: Information about reconstructed secondary vertices associated with each jet, including:
  - Transverse displacement (dxy)
  - Longitudinal displacement (dz)
  - Number of associated tracks
  - Invariant mass
  - Energy
  - Cosine of the angle between the vertex flight direction and the jet axis (cosθ)
- **Jet Features**: Overall properties of the jets, such as:
  - Jet transverse momentum (pT)
  - Jet mass (including soft-drop mass)
  - Jet area
  - Number of constituent particles
  - Subjet information (from jet clustering algorithms)
  - N-subjettiness variables (τN)
  - Energy correlation functions
- **Event Metadata**: Information about the entire event, such as:
  - Number of primary vertices
  - Total event energy
  - Global event variables (e.g., missing transverse energy)
- **Labels**: Ground truth labels indicating whether a jet is from a Higgs boson decay or a background process (QCD jet).

### Dataset Sources

- **CERN Open Data Portal**: Provides access to simulated collision data from CMS, including events with Higgs boson decays and QCD background processes.
- **MC Generators**: Monte Carlo generators like MADGRAPH5_aMC@NLO and PYTHIA can be used to generate signal and background events, followed by detailed detector simulations with GEANT4 to mimic the CMS detector response.

## Methodology

### Graph Neural Networks (GNNs)

#### Interaction Networks (INs)
Interaction Networks are designed to model interactions within a system of objects. They treat particles and secondary vertices within a jet as nodes in a graph, with interactions represented as edges. The model will utilize:
- **Node Features**: Properties of particles and secondary vertices.
- **Edge Features**: Interactions between particles and vertices.

#### Graph Convolutional Networks (GCNs)
GCNs apply convolution operations to nodes in a graph, aggregating information from neighboring nodes to learn node representations. This approach captures local interactions within a jet.

#### Long Short-Term Memory Networks (LSTMs)
LSTMs will be used to model the sequence of particles in a jet, treating the sequence as a time series to capture dependencies along the jet's particle sequence.

#### Convolutional Neural Networks (CNNs)
CNNs will be adapted to represent jets as 2D histograms, applying convolutional filters to extract spatial features.

### Model Development and Training

- **Data Collection and Preparation**: Acquire and preprocess high-quality simulated datasets.
- **Model Design and Implementation**: Develop the interaction network and integrate GNN, LSTM, and CNN techniques.
- **Training and Validation**: Train the model using advanced optimization techniques, validate on separate subsets, and tune hyperparameters.
- **Performance Evaluation**: Evaluate the model using metrics such as accuracy, AUC, and background rejection rate.
- **Mass Decorrelation**: Implement techniques to decorrelate the model's output from jet mass.
- **Robustness and Scalability**: Ensure robustness against varying conditions and scale the model for large datasets.

## Expected Outcomes

- **Improved Jet Tagging Performance**: Achieve an AUC greater than 0.99.
- **High Background Rejection Rate**: Achieve a background rejection factor (1/FPR) of over 1000 at a TPR of 30%, and over 500 at a TPR of 50%.
- **Robustness to Pileup**: Maintain high performance with minimal degradation as the number of reconstructed primary vertices increases.
- **Mass Decorrelation**: Achieve a high decorrelation metric (e.g., 1/DJS > 1000).

## Deliverables

- **Trained Interaction Network Model**: Optimized for real-time LHC data filtering.
- **Source Code and Documentation**: Including data preprocessing, training, validation, and testing scripts.
- **Performance Evaluation Report**: Detailed analysis of the model's performance.
- **Mass Decorrelation Techniques**: Documented methods and their effectiveness.
- **Data Handling and Preparation Pipeline**: Modular and reusable for future projects.
- **Integration Guidelines**: Instructions for integrating the model into existing workflows.
- **Published Research Paper**: Detailing the methodology, results, and implications.
- **User-Friendly Interface**: For running the model on new datasets.
- **Performance Monitoring and Visualization Tools**: For real-time metrics visualization and model interpretability.

## Team Roles

1. **Data Engineer**
2. **Model Architect**
3. **Training and Validation Engineer**
4. **Performance Evaluation Specialist**
5. **Integration and Deployment Engineer**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We would like to thank the CMS Collaboration for providing the datasets and the CERN Open Data Portal for making the data accessible.

