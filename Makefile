
all:
	cd libz80;make
	cd native;make

clean:
	rm -f *.so
	cd libz80;make clean

