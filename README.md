Sparkling Vector Machines
===========================

These codes implements Sparkling Vector Machines (SkVM) using MATLAB and Apache Spark with Python API. The codes are tested on MATLAB R2014b and Apache Spark version 1.4.1, Python 2.7. Please make sure that you have installed *python-numpy* and you can run *Spark on local* to run the demo.

Run a demo of SkVM for classification
-------------------------------------
	MATLAB: run_skvm.m
	Apache Spark: spark-submit --master local[*] skvm_spark.py

Run a demo of SkVM for label-drift classification
-------------------------------------------------
	MATLAB: run_skvm_labeldrift.m
	Apache Spark: spark-submit --master local[*] skvm_spark_labeldrift.py

Run a demo of SkVM for regression
---------------------------------
	MATLAB: run_skvm_regression_airline.m
	Apache Spark: spark-submit --master local[*] skvm_spark_regression.py

Citation
--------
	Tu Dinh Nguyen, Vu Nguyen, Trung Le, Dinh Phung. "Distributed Data Augmented Support Vector Machine on Spark". Accepted for the 23rd International Conference on Pattern Recognition (ICPR), 2016.
