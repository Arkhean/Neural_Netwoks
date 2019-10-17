all : network

network: build/matrix.o build/nn.o build/test.o build/random.o build/activation.o
	mkdir -p build
	g++ -g -O0 -o test $^
	./test

rnn: build/matrix.o build/rnn.o build/test_rnn.o build/random.o build/activation.o
	mkdir -p build
	g++ -g -O0 -o test $^
	./test

# ==============================================================================

networks.so: networks/nn.cpp utils/matrix.cpp utils/random.cpp utils/activation.cpp
	g++ -o lib$@ $^ -fPIC -shared

	# g++ test.cpp -L. -lnetworks 
	# export LD_LIBRARY_PATH=/your/path:$LD_LIBRARY_PATH


# ==============================================================================

build/matrix.o : include/matrix.hpp utils/matrix.cpp
	g++ -g -O0 -c utils/matrix.cpp -o $@
build/random.o : include/random.hpp utils/random.cpp
	g++ -g -O0 -c utils/random.cpp -o $@
build/activation.o : include/activation.hpp utils/activation.cpp
	g++ -g -O0 -c utils/activation.cpp -o $@
build/nn.o : networks/nn.cpp include/nn.hpp
	g++ -g -O0 -c networks/nn.cpp -o $@
build/rnn.o : networks/rnn.cpp include/rnn.hpp
	g++ -g -O0 -c networks/rnn.cpp -o $@
build/test.o : test.cpp
	g++ -g -O0 -c test.cpp -o $@
build/test_rnn.o : test_rnn.cpp
	g++ -g -O0 -c test_rnn.cpp -o $@

# ==============================================================================

clean:
	rm build/*
