import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Chgcar
from scipy.ndimage import gaussian_filter

class PartialCharge(object):
    """
    This class contains the partial charge densities from a DFT-simulation
    with a calculator to simulate a STM image.
    """
    def __init__(self, filename):
        """
        Reads PARCHG file and make class attributes
        """
        self.filename   = filename
        self.parchg     = Chgcar.from_file(filename)

        # charge data from CHGCAR
        self.chgdata    = self.parchg.data['total']

        # structure associated w/ volumetric data
        self.structure  = self.parchg.structure

        # fraction of cell occupied by structure
        self.zmax = self.structure.cart_coords[:, 2].max() \
                            / self.structure.lattice.c
        # max charge density
        # rmax should be maximum rho value within the stm_cell
        # i.e., [X, Y, Z[nzmin:nzmax]] rather than the whole box
        self.rmax = None


    def get_rhos(self, z_below, z_above):
        """
        Returns the maximum value of rho within the stm block
        [X, Y, Z[nzmin:nzmax]] in unit cell
        """
        # convert z_below, z_above from Å to frac coords
        z_below /= self.structure.lattice.c
        z_above /= self.structure.lattice.c

        # Get range on c-axis to consider above and below the top surface (zmax)
        nzmin = int(self.parchg.dim[2] * (self.zmax - z_below))
        nzmax = int(self.parchg.dim[2] * (self.zmax + z_above))

        # Get charge density data within (nzmin:nzmax,nrows,ncols)
        rhos  = self.chgdata.copy()[:, :, nzmin:nzmax]/self.structure.volume

        # Assign rmax attribute
        self.rmax = rhos.max()

        return rhos, nzmin, nzmax

    def get_stm_image(self, rhos, rval, rtol, nzmin, nzmax):
        """
        Returns a stm simulation same as the size & shape of the 2D unit cell
        """
        # If absdiff between charge density value and rval is not within rtol,
        # switch pixels to off (i.e. -9E9)
        rhos[abs(rhos - rval) >= rtol] = -9E9

        # Reshape to get z axis (x, y, z) --> (z, x, y)
        rhos = np.swapaxes(np.swapaxes(rhos, 0, 2), 1, 2)

        # Find xy for which all z pixels are off
        mask  = np.sum(rhos,axis=0) == np.shape(rhos)[0]*-9E9

        # Find all on pixels and flip bottom values to top
        # Now flip all bottom pixels to top
        temp  = (rhos.copy() > 0)[::-1]

        # Find index of first non-zero pixel along z
        image = np.argmax(temp,axis=0).astype(np.float64)

        # Find index of first non-zero pixel along z
        image = np.argmax(temp,axis=0).astype(np.float64)

        # Move image relative to top index of stm block
        image = (nzmax-1) - image

        # Replace off pixels with deepest index
        # Add arbitrary tolerance to distinguish off pixels from
        # deepest on pixel
        image[mask==True] = np.min(image[mask==False]) - 5

        # Change pixel values relative to height in Å
        image *= (self.structure.lattice.c / self.parchg.dim[2])

        return image

    def parameter_check(self, z_below, z_above, rval, rtol, rhos=False):
        """
        Validate choice of parameters to ensure feasibility of STM image calculation

        Args:
            z_below: (float) thickness or depth from top in Angstroms
            z_above: (float) distance above the surface to consider
            rval: (float) isosurface charge density plane
            rtol: (float) tolerance to consider while determining isosurface
            rhos: (bool) Whether rmax is attributed from get_rhos
        Returns:
            A boolean (Pass or Fail)
        """
        slab_thickness = self.structure.cart_coords[:, 2].max() - \
                                    self.structure.cart_coords[:, 2].min()

        # Check to make sure z_below is within valid depth range
        # [-1 Ang to 0.5 x slab_thickness]
        if -1 <= z_below <= 0.5*slab_thickness:
            pass
        else:
            print("ParameterError: z_below = {0} outside valid range" + \
                    " [{1} to {2:.5}]".format(z_below,-1,0.5*slab_thickness))
            return False

        # Check to make sure z_above is within bounds of structure
        # if self.zmax + z_above <= 1:
        if 0 < z_above <= (1 - self.zmax) * self.structure.lattice.c:
            pass
        else:
            print("ParameterError: z_above = {0} outside structure " + \
                  "(must be <= {1:.5})".format(
                        z_above, (1-self.zmax)*self.structure.lattice.c))
            return False

        # Check to make sure rval is within valid isosurface (rmax/3 to rmax)
        if not rhos:
            self.get_rhos(z_below, z_above)
        if self.rmax/3 < rval <  self.rmax:
            pass
        else:
            print ("ParameterError: rval = {0} outside valid " + \
                   "range ({1:.5} to {2:.5})".format(
                                        rval, self.rmax/3, self.rmax))
            return False

        # Check to make sure rtol is within proper tolerance
        # (0.0001 to 0.999*rval)
        if 0.0001 < rtol <  0.999*rval:
            pass
        else:
            print ("ParameterError: rtol = {0} outside valid range " + \
                   "({1:.5} to {2:.5})".format(rtol, 0.0001, 0.999*rval))
            return False

        return True

    def plot_stm(self, z_below=0, z_above=4, rval=None, rtol=None,
                 blur_image=False, sigma=None, scell_size=(3, 3),
                 filename='stm.pdf'):
        """
        Plots stm image and saves it to filename
        Different looking simulations of STM plots can be plotted by changing
        following parameters:

        The first two parameters determine the region in
        which states are considered to be accessible by STM.

        z_below: (float) distance below the topmost atom on surface
        z_above: (float) distanve above topmost atom on surface

        The latter two parameters determine which states within the region are
        detected by STM.

        rval: (float) the electron density value for isosurface at which STM is
                      simulated
        rtol: (float) the tolerance for the rho value, rval

        Add blur to the image; vary the sigma

        blur_image: (bool) Whether to blur the final image
        sigma: (float) The sigma for gaussian smearing
        filename: (str) To save image. Must include extension.
                        Eg: 'stm.pdf' or 'parchg_stm.png'

        Defaults are assumed for parameters if not provided. If wrong parameters are given, valid parameters are displayed with error.
        """
        # Get rhos within the stm block
        rhos, nzmin, nzmax = self.get_rhos(z_below, z_above)

        # Make default rval & rtol if not provided based on rmax
        if not rval or not rtol:
            rval = self.rmax * 0.95
            rtol = rval * 0.9
            print ("Using default values for rval & rtol...")
            print ("rval: {0:.5f}\nrtol:{1:.5f}".format(rval, rtol))

        # check if all parameters are satisfy the required constraints
        if not self.parameter_check(z_below, z_above, rval, rtol, rhos=True):
            print ("Please modify parameters and try again..")

        # get stm image in shape of unit cell
        stm_image = self.get_stm_image(rhos, rval, rtol, nzmin, nzmax)
        unit_im = stm_image.copy()

        # expand image size in x-direction
        for i in range(scell_size[0]-1):
            stm_image = np.concatenate((stm_image, unit_im))

        # expand image size in y-direction
        stm_image = np.swapaxes(stm_image, 0, 1)
        copy_2 = stm_image.copy()
        for i in range(scell_size[1]-1):
            stm_image = np.concatenate((stm_image, copy_2))
        stm_image = np.swapaxes(stm_image, 0, 1)

        # Blur the image for experimental effect (optional)
        if blur_image:
            if not sigma:
                print ("Using sigma = 5 for gaussian blur..")
                sigma = 5
            stm_image = gaussian_filter(stm_image, sigma=sigma)

        plt.imshow(stm_image.T, cmap='hot', origin='lower')
        plt.axis('off')
        plt.savefig(filename, format=filename.split('.')[1])

        return unit_im
