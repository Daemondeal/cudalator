.PHONY: all
all:
	cd ./surelog-sys/external/Surelog/ && $(MAKE)
	cd ./surelog-sys/external/Surelog/ && PREFIX=../../build/surelog $(MAKE) install

