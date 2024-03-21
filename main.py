

import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
# from 
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import SepformerSeparation as separator

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("main.wav", "main.wav")
print(score,prediction)
