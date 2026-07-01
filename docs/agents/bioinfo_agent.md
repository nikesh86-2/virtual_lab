# Bioinformatics Agent Documentation

## Overview

The bioinformatics agent analyzes evolutionary conservation patterns in viral RNA sequences.

## Responsibilities

- Identify organism via NCBI Taxon ID
- Retrieve genomes from NCBI databases
- Perform multiple sequence alignment (MSA) using MAFFT
- Compute conservation metrics from aligned sequences

## Outputs

- Conservation percentage across aligned sequences
- Conserved regions identified in the alignment
- MSA data in standard format

## Role in Pipeline

The bioinformatics agent provides evolutionary context that informs structural analysis:
- Evolution → constraint → filtering
- Conservation data feeds into the structural agent for conservation-aware structure prediction

## Tools Used

- NCBI Entrez API for genome retrieval
- MAFFT for multiple sequence alignment
- Custom conservation analysis scripts

---
*Last updated: 2026-07-01*