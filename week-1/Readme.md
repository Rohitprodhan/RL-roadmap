# Week 1: Introduction to Markov Decision Processes

## Overview
Reinforcement Learning is a computational approach to solving Markov Decision Processes (MDPs). This week focuses on understanding MDPs before diving into RL algorithms.

## Learning Objectives
- Understand the fundamentals of Markov Decision Processes
- Learn key components of an MDP
- Formulate problems as MDPs
- Explore the Student MDP example

## Topics Covered

### 1. What is a Markov Decision Process?
An MDP is a mathematical framework for modeling sequential decision-making problems where:
- The future depends only on the present state (Markov property)
- Decisions are made by an agent in an environment
- Each action leads to rewards and state transitions

### 2. MDP Components
- **State Space (S)**: Set of all possible states
- **Action Space (A)**: Set of all available actions
- **Transition Function (P)**: Probability of moving from one state to another
- **Reward Function (R)**: Immediate reward for taking an action in a state
- **Discount Factor (γ)**: Weight for future rewards

### 3. Problem Formulation: Student MDP
A practical example from David Silver's lectures demonstrating how to model a real-world problem as an MDP.

## Next Week
Week 2 will cover algorithms to solve MDPs, including dynamic programming approaches and value iteration.

## Resources
- [David Silver's RL Course (Lecture 2)](https://youtu.be/lfHX2hHRMVQ?si=Sxbv2A8mYMYBbVfL)
- [Reinforcement Learning: An Introduction, by Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
