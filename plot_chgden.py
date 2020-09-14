import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pymatgen.io.vasp import Chgcar, VolumetricData


class PartialCharge(object):

    """
    This class contains the partial charge densities from a DFT-simulation
    with a calculator to simulate a STM image.

    """

    def __init__(self,parchg_file):

        """
        Reads in the data from the provided PARCHG file and store as class attributes

        Parameters
        ----------
        parchg_file: (str) full path to PARCHG file

        """

        self.path, self.filename = os.path.split(parchg_file)

        if self.filename.endswith('.hdf5'):
            parchg = VolumetricData.from_hdf5(parchg_file)
        else:
            parchg = Chgcar.from_file(parchg_file)
            ## convert to hdf5 for much quicker reading in the future...
            Chgcar.from_file(parchg_file).to_hdf5(os.path.join(self.path,'%s.hdf5'%self.filename))

        ## structure associated w/ volumetric data
        self.structure = parchg.structure

        ## charge data from PARCHG/CHGCAR
        self.chgdata = parchg.data['total']

        ## grid size (NGXF, NGYF, NGZF)
        (self.ngrid_a, self.ngrid_b, self.ngrid_c) = np.shape(self.chgdata)

        ## grid vectors
        self.gridvec_a = self.structure.lattice.matrix[0]/self.ngrid_a
        self.gridvec_b = self.structure.lattice.matrix[1]/self.ngrid_b


    def extract_data_slice(self,tip_height=5.0):

        """
        Written by: Anne Marie

        Extract the relevant slice(s) of the charge density data
        ** Assumes that the surface lies in the a-b plane; c vector is normal to surface

        Parameters
        ----------
        tip_height: (float) STM tip height above top layer of atoms

        """

        ## top layer of atoms on surface
        height_surface = max([s.coords[2] for s in self.structure.sites])
        height_cell = self.structure.lattice.matrix[2][-1]

        ## determine the index of the slice corresponding to the probe height
        height_probe = height_surface + tip_height
        sliceind = int(round(height_probe/height_cell * self.ngrid_c)) - 1

        self.dataslice1 = self.chgdata[:,:,sliceind]


    def repeat_grid(self,scell_size=(3,3)):

        """
        Written by: Anne Marie

        Repeat the grid to make a supercell

        """

        data_unit = self.dataslice1.copy()

        ## repeat the grid along the first axis
        data_exp_a = data_unit
        for i in range(scell_size[0]-1):
            data_exp_a = np.hstack((data_exp_a,data_unit))

        ## repeat the grid along the second axis
        data_exp_b = data_exp_a
        for j in range(scell_size[1]-1):
            data_exp_b = np.vstack((data_exp_b,data_exp_a))

        self.dataslice = data_exp_b


    def map_grid_coords(self,scell_size=(1,1)):

        """
        Written by: Anne Marie

        Map the grid onto x,y cartesian coords

        """

        ## grid_x(y) will contain the x(y) coords of each corresponding grid element
        grid_x = np.zeros(np.shape(self.dataslice))
        grid_y = np.zeros(np.shape(self.dataslice))

        for i in range(scell_size[0]*self.ngrid_a):
            for j in range(scell_size[1]*self.ngrid_b):
                grid_x[i,j],grid_y[i,j],dummy = i*self.gridvec_a + j*self.gridvec_b

        self.grid_x = grid_x
        self.grid_y = grid_y


    def plot_STM_constant_height(self,scell_size=(1,1),crop_size=(None,None),
                                 colorbar=True,filename=None):

        """
        Written by: Anne Marie
        
        Generate the constant-height mode STM image

        Parameters
        ----------
        scell_size: (tuple) specifying the supercell to plot
        crop_size: (tuple) specifying the region of the plot to display
                   (e.g. to crop a nice square image from a hexagonal cell)
        filename: (str) filename to save image to. Must include extension.

        """

        ## repeat the grid to make a supercell
        self.repeat_grid(scell_size)

        ## map the grid onto x,y cartesian coords
        self.map_grid_coords(scell_size)

        plt.pcolormesh(self.grid_x,self.grid_y,self.dataslice,
                       cmap=cm.get_cmap("gray"),shading='gouraud')
        plt.clim(0,np.max(self.dataslice))
        if colorbar:
            plt.colorbar()
        if crop_size[0]:
            plt.xlim(crop_size[0])
        if crop_size[1]:
            plt.ylim(crop_size[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
#        plt.show()
        if filename:
            plt.savefig(os.path.join(self.path,filename), format=filename.split('.')[1])

        return


if __name__ == '__main__':


    ## initialize PartialCharge object
    ## read in the data from the provided PARCHG file and store as class attributes
    pc = PartialCharge("MoS2/PARCHG_vbm.hdf5")

    ## extract the relevant slice(s) of the charge density data
    pc.extract_data_slice(tip_height=2.0)

    ## generate the constant-height mode STM image
    pc.plot_STM_constant_height(scell_size=(5,5),crop_size=([0,10],[0,10]))
