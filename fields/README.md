# anzu-fields
The scripts in this directory take you from noiseless ICs to measurements of the basis spectra P_ij at a given snapshot. All that is needed is a single yaml file specifying all of the desired directories/settings for the measurements. You then call the scripts in three easy steps:

1. `python ic_binary_to_field.py $yamlname` will convert binaries dumped by the modified 2LPTIC script used in Kokron et al 21 into numpy arrays for subsequent measurements. 
2. `srun -n $Ntasks python make_lagfields.py $yamlname` will run the MPI job that takes the ICs and generates all of the Lagrangian bias fields. Normally this is run in the context of a SLURM script specifying MPI parameters.
3. `srun -n $Ntasks python measure_basis.py $yamlname` will run the MPI job that then uses particle catalogs and the weights built previously to build the late-time fields. Additionally, the power spectra that form the basis functions of the model are measured in this script and saved. 

And that's it! 

We currently only support Gadget sims (and some parts are hard-coded for sims run across 512 processes but tweaking this isn't hard). We plan to add FastPM support shortly. 

## A note on dependencies

To make these measurements fast we have a few dependencies in place beyond the standard `anzu` dependencies. The main dependency is on `mpi4py` and `mpi4py_fft` which adds support for massively parallel FFTs. This can be installed through 

`conda install -c conda-forge mpifpy-fft h5py=*=mpi*`
