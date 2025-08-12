## Heartrate Analysis
Mira Welner

August 2025

This project takes in Dr. Dey's heartrate data and analyses it with both a [Wasserstein GAN with Gradient Penalty (WGAN-GP)](https://arxiv.org/abs/1704.00028), and a [neural prophet network](https://neuralprophet.com/).


The sinusiod_test folder creates and tests fake data based on sinusoids, as a basic test to see if the GAN works at all

The create_analysis_plots.py file runs various tests on the heartrate data itself, and stores the corresponding plots in the figures folder.
It does not involve the GAN or the FFNN or any actual model.
