This is the repository for the script that

* Generates obfuscated images from CFP and JEFF databases

* Generates vector representations of obfuscated images using pretraining pth files
that varies according to tau value
	Use fuzzyarcface_resnet101_customized4 tau 0.1 100 epochs.pth for the best face verification precision

* Compare obfuscated images for face verification of the same class

* Does this for the entire CFP and JEFF dataset


INFERENCING DATASETS USED

CFP DATASET, CELEBRITIES DATASET FACES FRONTAL AND PROFILE http://www.cfpw.io/

JEFF Japanese Female Facial expression FACE RECOGITION JAPAN https://zenodo.org/records/3451524


NEXT TO BE USED

MIT CIBCL FACE RECOGNITION, SEVERAL CLASSES http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html

CROSS AGE CELEBRITY DATASET https://www.v7labs.com/open-datasets/cacd