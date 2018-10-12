NAME=unfold
export FC=gfortran
export CC=gcc

all:
	f2py -c -m $(NAME) $(NAME).f90

clean:
	rm $(NAME).so
