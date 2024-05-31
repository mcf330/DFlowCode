This is a implementation of our work: DFlow: Combining Denoising AutoEncoder and Normalizing Flow for High Fidelity Waveform Generation

## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config ljconfig.json
```

## Inference Example
# TTS LJ-Speech
python3 inference.py --checkpoint_file=path-to-checkpoint