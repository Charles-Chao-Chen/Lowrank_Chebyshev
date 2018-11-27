
all:
	g++ -Wall -g lowrank.cpp -isystem $(EIGEN_DIR) -o lowrank

