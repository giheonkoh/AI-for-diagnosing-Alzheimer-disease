#
*** This is a set of sample codes to show a general schema of AI system *** 
*** SHOULD BE STARTED IN src FOLDER !!!
*** FULL PATH statement is required !!! 

> cd ~/Desktop/src

--------------------------------------------------------------------------
* CHECKING Dependency and SET UP
 [python3 - pip - [TensorFlow, Pandas, Numpy, scikit-learn, matplotlib, pickle]]
 
 > ./setup.sh

--------------------------------------------------------------------------
* MERGE tables and EXPORT csv file
 ./getData.sh [uploadPATH] [inputdir]

 > ./getData.sh /User/user/srv/save sub_0001

--------------------------------------------------------------------------
* TRAINING
 ./train.sh [sample csv file] [mri device(sm, pl, ge)]

 > ./train.sh training_sample.csv sm

--------------------------------------------------------------------------
* TESTING
 ./test.sh [sample csv file] [mri device(sm, pl, ge)]

 > ./test.sh sub_0001_sample.csv sm

--------------------------------------------------------------------------
* Generating Report
 ./report.sh [inputdir] [inpudir_data] [mri device(sm, pl, ge)]

 > ./report.sh sub_0001 sub_0001_sample.csv pl

--------------------------------------------------------------------------




#




