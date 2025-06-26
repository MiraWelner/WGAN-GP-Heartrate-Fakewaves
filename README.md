# Wasserstein GAN trained on Dr. Cong's ECG Data Analysis

This codebase takes Dr Cong's data, proccesses it via the `Raw_Data_Proccessing` folder, and then trains a Wasserstein GAN in the WGAN-GP folder. The models are stored in the `models` folder and the figures are stored in the `figures` folder.

This is a proof-of-concept for the WGAN-GP portion of the digital patient project.

## Setting up environment
This project uses Python 3.12, since that is the highest version that the graphing library, [DGL](https://www.dgl.ai/), will accept. DGL isn't used on the GAN but it is used on other parts of the project and I thought a uniform Python version would be easiest.

You *can* set up the environment by installing the contents of `requirements.txt` however I would instead reccomend using the `.venv` folder which is also on the GitHub (or the sharepoint if that is where you downloaded the code). You can activate the `.venv` environment by running `source .venv/bin/activate`, or `source .venv/bin/activate.fish` if you use fish shell or `.venv\Scripts\activate.bat` on Windows.

## Proccessing data
The raw data in the form given me by Dr. Cong is in the `Raw_Data_Proccessing\raw_ecg_ppg` folder. To proccess it, which includes denoising, normalizing it between -1 and 1, and removing frequency nonise, simply run `proccess_raw_data.py`. It doesn't need any arguments. This will create the `proccessed_data.csv` file, which is included on sharepoint but could not be loaded onto GitHub due to its size.

The `proccessed_data.csv` file, when created from the `proccess_raw_data.py` file as it is currently set up or as it is on sharepoint, has the following channels:

`ecg0_channel0, ecg0_channel1, ecg0_channel2, ecg1_channel0, ecg1_channel1,ecg1_channel2, ecg2_channel0, ecg2_channel1, ecg2_channel2, ppg0, ppg1, ppg2
`

`ecg0*` and `ppg0` correspond to patient 2. This is somewhat odd but it is because there is no ppg data corresponding to patient 1, and the patient number are 1 indexed 1 to 4 while I'm naming them 0 index. So `ecg1*` and `ppg1` correspond to patient 3 and `ecg2*` and `pp2` correspond to patient 4.

The ecg signals all have 3 channels corresponding to 3 ECG sensors placed on the body.

## Training WGAN-GP
The generator and the critic of the Wasserstein GAN with Gradient Penalty are in the `WGAN-GP/wgan.py` file. They are based on [this publication](https://arxiv.org/abs/1704.00028).

To train the GAN, just run `train_GAN.py`. It doesn't need any arguments. It will create all the images in the `figures` folder (other than `generate_signals`  and the models in the `models` folder.

The `generate_GAN_data.py` file generates the `generate_signals` figure.

The `train_GAN.py` file can easily be modified by changing the following constants at the top of the file:

`seconds = 7` This is how many seconds of ECG/PPG data the WGAN-GP produces<br>

`hz=500` You shouldn't change this unless you get new data from Dr. Cong or somebody else. The ECG data used in this project was 500hz and the PPG was 125 and interpolated to be 500hz.

`batch_size=248` The batch size - make this higher if your computer is beefy.

`latent_dim=100` the Gaussian noise which is put into the GAN is `latent_dim` units long

`epochs = 500` This is just the epochs


`itt = 100` in the `output_comparison` figure, the WGAN-GP is run `itt` times and the mean and 95% confidence are reported.

`signals_tested = ['ecg0_channel0']` This means that the WGAN-GP will only produce data similar to `ecg0_channel0`. Any of the column names in `proccessed_data.csv` can be added.
