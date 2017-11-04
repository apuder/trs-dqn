
all:
	mkdir -p lib
	cd libz80;make
	cp libz80/libz80.so lib/
	cd native;make

clean:
	rm -rf lib
	cd libz80;make clean

