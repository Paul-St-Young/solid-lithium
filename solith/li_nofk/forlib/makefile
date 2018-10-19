export FC=gfortran
export CC=gcc

all:
	for name in det unfold; do \
		f2py -c -m $$name $$name.f90; \
	done

clean:
	for name in det unfold; do \
		rm $$name.so; \
	done
