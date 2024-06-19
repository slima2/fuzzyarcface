# FuzzyArcface
Fuzzy ArcFace: Enhancing Face Recognition with Membership-Function-Integrated Angular Margin Loss


Paper and source code for FuzzyArcface


Face recognition systems have significantly benefited
from the introduction of the ArcFace loss function, which
enhances the discrimination capability by focusing on angular
relationships within facial feature vectors. However, ArcFace
struggles with images positioned at or near class boundaries,
such as those affected by occlusions, lighting, or complex facial
expressions. This paper presents FuzzyArcFace, an enhanced loss
function that integrates a fuzzy membership function into the
traditional ArcFace framework. The fuzzy membership function
dynamically adjusts the angular margin based on the certainty
of class membership. This adjustment provides a flexible and
nuanced mechanism for classifying boundary cases more effec-
tively. Extensive experiments on the labeled faces in the wild
(LFW), japanese female facial expression (JEFF), and celebrities
in frontal-profile (CFP) datasets demonstrate that FuzzyArc-
Face consistently outperforms ArcFace, especially in challenging
scenarios. Notably, FuzzyArcFace achieves the highest accuracy
on JEFF and CFP datasets by allowing more fluctuation in
margin to accommodate nuanced images. These results highlight
FuzzyArcFace’s potential to enhance face recognition accuracy
and robustness across diverse scenarios, making it a promising
tool for applications requiring high precision in face verification
