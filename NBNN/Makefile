CXX      = g++
CXXFLAGS = -Iinclude/ANN
LDFLAGS  = -g2
LIBS     = -Llib/ANN -lANN

all: nbnn

nbnn: nbnn.o 
	$(CXX) -o nbnn $(LIBS) $(LDFLAGS) nbnn.o

nbnn.o: nbnn.cpp
	$(CXX) -c $(CXXFLAGS) nbnn.cpp

clean:
	rm -f *.o
	rm -f nbnn