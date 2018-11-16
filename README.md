# LUPI-SVM
LUPI-SVM: SVM with Learning Using Privileged Information (LUPI) Framework

See example.py for basic use and implementation.

# What is Learning Using Privileged Information (LUPI)?
It is a learning paradigm where, at the training stage, Teacher provides additional information x' called previleged information about
training example x. The crucial point in this paradigm is that the privileged information is available only at the training stage (when Teacher interacts with Student) and is not available at the test stage (when Student operates without supervision of Teacher) [1].

Here, we have implemented in python a stochastic sub-gradient optimization (SSGO) algorithm for LUPI, inspired by the Pegasos solver for conventional binary SVMs [2]. The detail of this implemenation is given in [3].

# Citation details
If you use this code please cite: Wajid Arshad Abbasi, Amina Asif, Asa Ben-Hur and Fayyaz-ul-Amir Afsar Minhas (2018), "Learning protein binding affinity using privileged information", BMC Bioinformatics, 19, 425. 

# References

[1]. Vapnik,V. and Izmailov,R. (2015) Learning Using Privileged Information: Simi-larity Control and Knowledge Transfer. J. Mach. Learn. Res., 16, 2023–2049.

[2]. Shalev-Shwartz,S. et al. (2011) Pegasos: primal estimated sub-gradient solver for SVM. Math. Program., 127, 3–30.

[3]. Wajid Arshad Abbasi, Amina Asif, Asa Ben-Hur and Fayyaz-ul-Amir Afsar Minhas (2018), "Learning protein binding affinity using privileged information", BMC Bioinformatics, 19, 425.

