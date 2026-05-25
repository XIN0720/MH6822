Name: TONGXIN
Matriculation ID: G2505218J
Email: XIN021@e.ntu.edu.sg

# JurisFair BNPL Credit Monitor

## Assignment Information
This repository contains the Task 3 GitHub submission materials for Assignment 1, Session 1: Designing a Jurisdiction-Aware RegTech Tool.
The project follows Task 3 Option C: Analytical Design with Quantitative Component.
The selected regulated entity is Klarna Bank AB / Klarna Group.
The selected regulatory domain is credit scoring, algorithmic fairness, fair lending and AI governance.
The selected jurisdictions are the United States and the European Union.

## Project Summary
JurisFair BNPL Credit Monitor is a jurisdiction-aware compliance monitoring prototype for Klarna-style BNPL and consumer credit decisions.
The tool uses synthetic BNPL application data to train an interpretable credit risk model and monitor risk-adjusted approval disparity across audit groups.
The same quantitative result is interpreted differently under US and EU rule configurations.
In the US configuration, the tool focuses on fair lending monitoring, ECOA / Regulation B adverse-action reason specificity, AI credit decision explainability and fair-access review.
In the EU configuration, the tool focuses on EU AI Act high-risk AI governance, bias monitoring, data governance, traceability, technical documentation, post-market monitoring and human oversight.

## What the Tool Does
The tool generates synthetic BNPL credit application data.
The tool trains a logistic regression model to estimate predicted probability of 90-day default.
The tool applies a residual policy layer to simulate additional business decision rules.
The tool generates approval outcomes and denial reasons.
The tool calculates Risk-Adjusted Approval Disparity, AIR, false-negative-rate gap, PSI and subgroup PSI.
The tool applies jurisdiction-specific thresholds and trigger logic using a YAML rule configuration file.
The tool produces baseline results, sensitivity analysis results, jurisdiction flags and figures for management review.

## What the Tool Does Not Do
The tool does not approve or reject real consumers.
The tool does not use real Klarna customer data.
The tool does not represent Klarna's actual credit model, approval policy, compliance status or regulatory risk.
The tool does not provide final legal advice.
The tool does not automatically send adverse-action notices.
The tool does not prove that a model is fair or unlawful.
The tool is a teaching prototype and second-line compliance monitoring demonstration.

## Data Collaboration Statement

The synthetic dataset used in this project was generated collaboratively by Tong Xin and Lei Min for the data generation portion of the assignment.

Collaborator:
Lei Min  
Matriculation ID: G2506259A  
Email: MIN016@e.ntu.edu.sg

The collaboration was limited to the synthetic data generation component of the project. No real customer data, real Klarna data, or confidential consumer information was used. The dataset was created only for demonstrating the jurisdiction-aware RegTech tool logic, including credit-risk modelling, risk-adjusted approval disparity monitoring, jurisdiction-specific threshold testing and sensitivity analysis.

All remaining analytical design, written interpretation, tool framing, jurisdictional explanation and final submission materials are prepared for the individual project submission.
