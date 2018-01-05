import numpy as np
from scipy import interpolate
import MDSplus as mds

# Add path to EFIT folder for the load_gfile_d3d function.
import sys
sys.path.append("/home/shawn/DIII-D/ORNL-Fusion/EFIT")
import EFIT.load_gfile_d3d as loadg


def RtoPsin(radius, z_loc, shot, time, MDSplusConn):
    """
    This function will return the psin value at a specific radius and z location.

    radius: Desired radius in m.
    z_loc:  Desired Z coord. in m.
    shot:   The shot.
    time:   The requested time to get the location for in ms.
    MDSplusConn: An MDSplus connection object created by MDSplus.Connection("atlas.gat.com").

    """
    # Load the g-file. It will grab the closest time.
    parameterDict = loadg.read_g_file_mds(shot=shot,
                                          time=time,
                                          connection=MDSplusConn,
                                          write2file=False,
                                          tree="EFIT01")

    # Create a grid out of the R,Z values.
    Rs, Zs = np.meshgrid(parameterDict['R'], parameterDict['Z'])
    # R,Z of the magnetic axis.
    RmAxis = parameterDict["RmAxis"]
    ZmAxis = parameterDict["ZmAxis"]
    # The R's and Z's of the LCFS.
    lcfsZs = np.copy(parameterDict['lcfs'][:, 1][13:-12])
    lcfsRs = np.copy(parameterDict['lcfs'][:, 0][13:-12])

    # Function to interpolate along the LCFS. Give it a Z, and it gives you Rsep.
    fLcfs = interpolate.interp1d(lcfsZs, lcfsRs, assume_sorted=False)
    # Only grab the right half of the R's (don't care about the inner wall side).
    RsTrunc = Rs > RmAxis
    # Function to interpolate the psiN values. Give it an R and Z and it gives you psiN at that location.
    fPsiN = interpolate.Rbf(Rs[RsTrunc], Zs[RsTrunc],
                            parameterDict['psiRZn'][RsTrunc], function='linear')

    # Finally find the psin value at the requested radius and z_loc.
    tmp_psin = fPsiN(radius, z_loc)

    return tmp_psin
