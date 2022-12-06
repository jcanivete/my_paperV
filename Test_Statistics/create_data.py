# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import swirlpy
import time
import subprocess

# Code
import swirl

# Functions
def Poyting_z(Bx, By, Bz, vx, vy, vz):
    Sh = -Bz*(vx*Bx+vy*By)/(4.*np.pi)
    Sv = vz*(Bx**2+By**2)/(4.*np.pi)
    return Sv, Sh

def Beta(P, Bx, By, Bz):
    Pm = (Bx**2 + By**2 + Bz**2)/(8.*np.pi)
    return P/Pm
    
def q_in_cells(q, cells, z, b):
    q_cells = np.array([q[int(i),int(j),int(z/b)] for (i,j) in zip(cells[0],cells[1])])
    q_av = np.mean(q_cells)
    q_std = np.std(q_cells)
    return q_av, q_std

def compute_vortexdata(beta, Sv, Sh, Bz, rho, MSz, atmosphere, zrange, xi, yi, dz):
    data_sheet = []
    for (layer,z) in zip(atmosphere,zrange):
        for vortex in layer:
            xc = vortex.center[0]
            yc = vortex.center[1]
            zc = z
            radius = vortex.radius
            rortex = np.mean(vortex.rortex)
            rortex_std = np.std(vortex.rortex)
            orient = vortex.orientation
            cells = vortex.vortex_cells
            beta_av, beta_std = q_in_cells(beta, cells, z, 1)
            Sv_av, Sv_std = q_in_cells(Sv, cells, z, 1)
            Sh_av, Sh_std = q_in_cells(Sh, cells, z, 1)
            Bz_av, Bz_std = q_in_cells(Bz, cells, z, 1)
            MSz_av, MSz_std = q_in_cells(MSz, cells, z, dz)
            rho_av, rho_std = q_in_cells(rho, cells, z, 1)

            data_sheet.append([xc+xi,      #0
                               yc+yi,      #1
                               zc,      #2
                               radius,  #3
                               rortex,  #4
                               rortex_std, #5
                               orient,  #6
                               beta_av, #7
                               beta_std,#8
                               Sv_av,   #9
                               Sv_std,  #10
                               Sh_av,   #11
                               Sh_std,  #12
                               Bz_av,   #13
                               Bz_std,   #14
                               MSz_av,  #15
                               MSz_std,  #16
                               rho_av, # 17
                               rho_std # 18
                              ]) 
    data_sheet = np.array(data_sheet)
    return data_sheet

def main():
    trange = [str(t).zfill(4) for t in np.arange(240,1000,240)]
    #trange = [str(5774).zfill(4)]
    for t in trange:
        
        # Parameters
        N = 960
        xi = 0
        yi = 0
        zi = 93
        zf = 223
        dx = 10.0*1e5
        xf = xi+N
        yf = yi+N

        dz = 1
        dl = 160
        k = int(N/dl)
        slices = []
        for i in range(k):
            for j in range(k):
                slices.append([i*dl,(i+1)*dl,j*dl,(j+1)*dl])
        #print(slices)

        zrange = np.arange(0,zf-zi+1,dz)

        # Load CO5BOLD Data
        box = swirlpy.Box(xi=[xi,yi,zi], xf=[xf,yf,zf+1],boundary=[1,1,1])
        box.read(tsnap=t, 
                dir='/home/cluster/jcaniv/shares/job3D960x960x280_v50_relax/COBOLD_Sim', 
                list_quantity=['v','B','Pre','rho'])
        vx = box.v.x
        vy = box.v.y
        vz = box.v.z
        rho = box.rho.s
        By = box.B.x
        Bx = box.B.y
        Bz = box.B.z
        P = box.Pre.s
        MagSwirl = swirlpy.MagSwirl(box,bin=[1,1,dz])
        
        # Compute quantities
        beta = Beta(P, Bx, By, Bz)
        Sv, Sh = Poyting_z(Bx, By, Bz, vx, vy, vz)
        MagSwirl.compute_swirl()

        atmosphere = []
        i_index = 0
        for [xi,xf,yi,yf] in slices:

            print(xi,xf,yi,yf)
            sub_atmosphere = []
            for z in zrange:
                #print('##################\n',z,'##################\n')
                v = [vx[xi:xf,yi:yf,z],vy[xi:xf,yi:yf,z]]
                grid_dx = [dx,dx]
                vortex = swirl.Identification(v=v,
                                            grid_dx = grid_dx,
                                            param_file = 'CO5BOLD.param',
                                            verbose=False)
                                    
                vortex.run()
                sub_atmosphere.append(vortex)
            
            data = compute_vortexdata(beta[xi:xf,yi:yf,:], 
                                    Sv[xi:xf,yi:yf,:], 
                                    Sh[xi:xf,yi:yf,:], 
                                    Bz[xi:xf,yi:yf,:],
                                    rho[xi:xf,yi:yf,:],
                                    MagSwirl.z[xi:xf,yi:yf,:],
                                    sub_atmosphere, 
                                    zrange,
                                    xi,
                                    yi,
                                    dz)
            
            np.save('newdata/vortexdata_t'+t+'_'+str(i_index)+'.npy', data)
            i_index += 1
        
        v = np.load('../Test_Statistics/newdata/vortexdata_t'+t+'_0.npy')
        for i in np.arange(1,len(slices)):
            v_i = np.load('../Test_Statistics/newdata/vortexdata_t'+t+'_'+str(i)+'.npy')
            v = np.concatenate((v,v_i))
            subprocess.run(['rm', '../Test_Statistics/newdata/vortexdata_t'+t+'_'+str(i)+'.npy'])
        np.save('../Test_Statistics/newdata/vortexdata_t'+t+'_final.npy', v)
#         vortexdata = []
#         for (subatm, [xi,xf,yi,yf]) in zip(atmosphere, slices):
#             data = compute_vortexdata(beta[xi:xf,yi:yf,:], 
#                                     Sv[xi:xf,yi:yf,:], 
#                                     Sh[xi:xf,yi:yf,:], 
#                                     Bz[xi:xf,yi:yf,:],
#                                     rho[xi:xf,yi:yf,:],
#                                     MagSwirl.z[xi:xf,yi:yf,:],
#                                     subatm, 
#                                     zrange,
#                                     xi,
#                                     yi,
#                                     dz)
#             vortexdata.append(data)
#         vortexdata = np.vstack(vortexdata)
#         np.save('newdata/vortexdata_t'+t+'.npy', vortexdata)

if __name__ == '__main__':
    main() # Launches my function when i first start the script