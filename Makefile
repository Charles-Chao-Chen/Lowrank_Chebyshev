
all:
	g++ -Wall -O3 lowrank.cpp -isystem $(EIGEN_DIR) -o lowrank

