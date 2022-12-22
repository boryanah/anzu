FROM ubuntu:latest
WORKDIR /opt

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        python3-dev          \
        python3-pip          \
	libgsl-dev	     \
	git		     \
#	libhdf5-serial-dev   \
	pkg-config           \
        wget              && \
    apt-get clean all

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich
ARG FFLAGS="-w -fallow-argument-mismatch -O2"

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure --prefix=/usr/ --enable-shared=yes                          && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN \
    wget http://www.fftw.org/fftw-3.3.10.tar.gz				    && \
    tar xvzf fftw-3.3.10.tar.gz                                             && \
    cd fftw-3.3.10                                                          && \
    ./configure --prefix=/usr/ --enable-shared=yes --enable-threads && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    ./configure --prefix=/usr/ --enable-shared=yes --enable-threads --enable-float  && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf fftw-3.3.10

RUN \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz && \
    tar xvzf hdf5-1.12.1.tar.gz				       && \
    cd hdf5-1.12.1					       && \
    ./configure --prefix=/usr/ --enable-shared=yes --enable-parallel && \
    make -j 4		       			   	       && \
    make install 					       && \
    make clean 						       && \
    cd ..						       && \
    rm -rf hdf5-1.12.1

RUN /sbin/ldconfig
RUN pip install cython numpy scipy
RUN pip install nose jupyter ipython
RUN python3 -m pip install camb cython chaospy
RUN MPICC=/usr/bin/mpicc python3 -m pip install mpi4py #==3.0.b3 --no-cache-dir --no-binary=mpi4py
RUN FFTW_DIR=/usr python3 -m pip install mpi4py-fft --no-cache-dir
RUN git clone https://github.com/j-dr/h5py.git && cd h5py && HDF5_DIR=/usr/ CC=mpicc HDF5_MPI="ON" python3 setup.py build && HDF5_DIR=/usr/ CC=mpicc HDF5_MPI="ON" python3 setup.py install #install --no-cache-dir --no-binary=h5py h5py
RUN python3 -m pip install sympy numexpr
RUN echo "hello"
RUN file="$(which mpicc)" && echo $file

RUN python3 -m pip install git+https://github.com/cosmodesi/pypower
RUN pip install nbodykit
RUN pip install pyfftw
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN git clone https://github.com/lesgourg/class_public.git class && cd class && make
RUN python3 -m pip install -v git+https://github.com/sfschen/velocileptors
RUN python3 -m pip install matplotlib
ENV CSBUST=19
ENV ANOTHAONE=1
RUN python3 -m pip install numba
ENV CSBUST=20
RUN git clone https://github.com/j-dr/anzu.git && cd anzu && git checkout aemulus_nu && python3 setup.py clean --all && python3 setup.py install
