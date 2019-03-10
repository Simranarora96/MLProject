# MLProject
Amalgamation of IoT and machine learning algorithms to Design, Develop and Implement a generic smart Iot Network
comprising of various smart gateways and a Smart Cloud.
Different categories of gateways create a generic system for various IoT device subscriptions, each gateway has the following
functionality:
1. Filtration and Anomaly Detection: Use of Decision tree algorithms such as C4.5 and C5.0 alongwith Jaccard coefficient
to filter out data.
2. Predictive Analysis/ Classification: Data is shrunk in volume as some predictive analysis and classification is performed
using Random forest algorithm coupled with time series analysis to draw results which can be further used by the central
cloud to draw useful insights.

The reduced smart data received from these distributed gateways is coupled at the central smartcloud to draw relevant insights.
The cloud is also enabled with the following feature making it novel and smart:
1. Autonomous IoT device assignment to gateway: Based upon various properties(cpu utilization, network type, data
generation rate, capacitance, etc.) of the device the cloud uses random forest algorithm to classify the device to each
gateway thus making the selection autonomous, independent, relevant and scalable for millions of devices.
