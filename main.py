import taichi as ti
import numpy as np
import constants as cst
import time
import sys
from taiphysics import *
from res_energy import *
from taiparticle import *
from wave_generate import *
from taiwave import Wave

if __name__ == '__main__':

    if len(sys.argv) > 1:
        print("The running id:", sys.argv[1:])
    else:
        print("No input id provided.")
    
    # read the input file
    id = sys.argv[1:][0]
    with open(id + '/input.txt', 'r') as f:
        lines = f.readlines()
    paras = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(': ')
            try:
                paras[key] = int(value)
            except ValueError:
                try:
                    paras[key] = float(value)
                except ValueError:
                    paras[key] = value


    # read the paras
    print("Taiparticle for oblique wave version 2.0 =====================")
    
    print("In this version, the psi, which is the wave normal angle is deprecated.") 
    print("Using gX to give a distribution for wavenormal angle")
    print("=============================================================")
    print("The followings are input parameters:")

    for key, value in paras.items():
        print(key, ':', value)
    B0 = paras['B0']# Background magnetic field
    #  psi0 = np.deg2rad(paras['psi']) # wave normal angle
    # no psi0 in this version
    wce = gyrofrequency(cst.Charge, cst.Me, B0)
    n0 = paras['n0']# density , I need to change this to wpe?
    wpe = plasmafrequency(cst.Charge,cst.Me,n0)
    print('******************')
    print('wce/wpe',wce/wpe)
    #wpe = paras['wpe']
    wave_distribution = paras['wave_distribution']
    nres = paras['nres']
    Np = paras['Np']# particle numbers
    pitch_angle_degree = paras['pitch_angle_degree'] # to the background magnetic field!
    t_total_num = paras['t_total_num']
    record_num = paras['record_num']
    dt_num =paras['dt_num']
    w_res_num =paras['w_res_num']
    wm_num = paras['w_max_num'] # freqeucy where has maximum energy
    w_lc_num = paras['w_lc_num']
    w_uc_num = paras['w_uc_num']
    w_width_num = paras['w_width_num']
    nw = paras['nw']
    dz_num = paras['dz_num']
    Bw = paras['Bw']

    mass = cst.Me
    charge = cst.Charge * -1 # -1 is electron
    
    
    
    
    
    X_min = paras['X_min']
    X_max = paras['X_max']
    nX = paras['nX']
    res_psi0 = np.deg2rad(paras['res_psi0']) # this is for calculating the resonance energy, give one wave normal angle
    
    
    
     ########### init the taichi kernel
    ti.init(arch = ti.cpu,default_fp=ti.f64)
    
    
    
    print("=============================================================")
    print('Calculating resonance energy')
    #
    alpha = np.deg2rad(pitch_angle_degree) # pitch angle, notice the antidirection
    w = w_res_num * wce
    wm = wm_num * wce
    w_lc = w_lc_num * wce
    w_uc = w_uc_num * wce
    w_width = w_width_num * wce
    p0,k0 = get_resonance_whistler(w, wpe,wce,alpha,nres,res_psi0)

    print('momentum of resonating particles are:',p0)
    print('resonant wave number is :', k0)
    print('resonating frequency is (Hz) ', w/(2 * np.pi))
    print('resonating frequency is (rad/s) ', w)
    
    print('E0 is ', erg2ev(p2e(p0))/1000, ' keV')
    
    if paras['energy_keV'] > 0:
        print("not using  the resonance energy but using the given energy")
        E0 = paras['energy_keV']
        print('E0 is ', E0, ' keV')
        p0 = e2p(ev2erg(E0 * 1000) )
    gamma = (1 + p0**2 / (cst.Me**2*cst.C**2))**0.5
    wce_rel = wce/gamma
    T_gyro = 2 * np.pi/ wce_rel
    dt = T_gyro * dt_num
    Nt = int(t_total_num/dt_num) 
    print(' time step is:', dt)
    print(' mass is ', mass)
    print(' charge is ', charge)
    print(' total time is', t_total_num * T_gyro)
    print(' total time step is ', Nt)
    particles = Particle.field(shape = (Np,))
    
    print("Wave distribution is ", wave_distribution)
    #print(wave_distribution == "Larry")
    print(nw == 200)
    if ((nw > 1) and (wave_distribution == "Constant")) :
        iw_res = int((w - w_lc) / ((w_uc - w_lc) / (nw - 1)))
        dw = (w_uc - w_lc) / (nw - 1)
        w_lc_temp = w - iw_res * dw; 
        ws = np.array([i * dw for i in range(nw)] ) + w_lc_temp
    elif ((nw > 1) and (wave_distribution == "Larry")):
        ws = np.linspace(w_lc,w_uc,nw)
    else:
        ws = np.array([w])
    print('resonance frequency:',w)
    print('ws')
    print(ws)
    k_res = k0
    wave_length = 2 * np.pi / k_res
    if Np > 1:
        dz = dz_num * wave_length/(Np - 1)
        
    else:
        dz = dz_num * wave_length

    dphi = 2 * np.pi / Np

    pperp = p0 * np.sin(alpha)

    px_numpy = np.zeros(Np)
    py_numpy = np.zeros(Np)
    pz_numpy = np.zeros(Np)

    for n in range(Np):
        phi = dphi * n
        px_numpy[n] = pperp * np.cos(phi)
        py_numpy[n] = pperp * np.sin(phi)
        pz_numpy[n] = p0 * np.cos(alpha)

    px_init = ti.field(dtype = ti.f64,shape = (Np,))
    py_init= ti.field(dtype = ti.f64,shape = (Np,))
    pz_init= ti.field(dtype = ti.f64,shape = (Np,))

    px_init.from_numpy(px_numpy)
    py_init.from_numpy(py_numpy)
    pz_init.from_numpy(pz_numpy)
    
    # Important change
    # init the wave
    init_ws = np.zeros((nw,nX))
    init_phi0 = np.zeros((nw,nX))
    init_Ewx = np.zeros((nw,nX))
    init_Ewy = np.zeros((nw,nX))
    init_Ewz = np.zeros((nw,nX))
    init_Bwx = np.zeros((nw,nX))
    init_Bwy = np.zeros((nw,nX))
    init_Bwz = np.zeros((nw,nX))
    init_k = np.zeros((nw,nX))
    
    # calculate gX here
    Xs = np.linspace(X_min,X_max,nX)
    X_m = (X_max + X_min)/2
    #psi_Xs = np.arctan(Xs)
    for i,X in enumerate(Xs):
        psi = np.arctan(X)
        gX = np.exp(-1 * ((X-X_m)/(X_max - X_min))**2)
        print("this is gX",gX)
        
        waves_init = Waves_generate(ws, B0, n0, Bw, psi ,w_width, wm,distribution =wave_distribution)
        waves_init.generate_oblique_wave(wce,wpe)
        # TODO: Make sure this is the right normalization. 
        init_ws[:,i] = waves_init.ws 
        init_phi0[:,i] = waves_init.phi0
        init_Ewx[:,i] = waves_init.Ewx * gX
        init_Ewy[:,i] = waves_init.Ewy * gX
        init_Ewz[:,i] = waves_init.Ewz * gX
        init_Bwx[:,i] = waves_init.Bwx * gX
        init_Bwy[:,i] = waves_init.Bwy * gX
        init_Bwz[:,i] =  waves_init.Bwz * gX
        init_k[:,i] = waves_init.k
    
    ws_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    phi0_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    Ewx_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    Ewy_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    Ewz_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))

    Bwx_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    Bwy_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    Bwz_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    k_taichi = ti.field(dtype = ti.f64,shape = (nw,nX))
    psi_taichi = ti.field(dtype = ti.f64,shape = (nX,))
    
    ws_taichi.from_numpy(init_ws)
    phi0_taichi.from_numpy(init_phi0)
    Ewx_taichi.from_numpy(init_Ewx)
    Ewy_taichi.from_numpy(init_Ewy)
    Ewz_taichi.from_numpy(init_Ewz)

    Bwx_taichi.from_numpy(init_Bwx)
    Bwy_taichi.from_numpy(init_Bwy)
    Bwz_taichi.from_numpy(init_Bwz)
    k_taichi.from_numpy(init_k)
    psi_taichi.from_numpy( np.arctan(Xs))
       
        
        
    #waves_init = Waves_generate()
    waves = Wave.field(shape = (nw,nX))
    dt_taichi = ti.field(ti.f64, shape=())
    dt_taichi[None] = dt


    print('Record num is', Nt//record_num)
    if Nt%record_num > 0:
        
        p_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))
        E_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))

        B_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))

        r_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))
        phi_record_taichi = ti.field(dtype = ti.f64,shape = (Nt//record_num+1, Np))
    else:
        
        p_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))
        E_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

        B_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

        r_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))
        phi_record_taichi = ti.field(dtype = ti.f64,shape = (Nt//record_num, Np))


    @ti.kernel
    def init():
        for n in range(Np):
            particles[n].initParticles(mass, charge)
            particles[n].initPos(0.0,0.0,dz * n)
            particles[n].initMomentum(px_init[n], py_init[n],pz_init[n])
        for m in range(nw):
            for j in range(nX):
                waves[m, j].initialize(ws_taichi[m,j],phi0_taichi[m,j],Ewx_taichi[m,j],
                                Ewy_taichi[m,j],Ewz_taichi[m,j],Bwx_taichi[m,j],Bwy_taichi[m,j],Bwz_taichi[m,j],k_taichi[m,j],psi_taichi[j])
        B =ti.Vector([0.0,0.0,B0])
        E = ti.Vector([0.0,0.0,0.0])
        for m in range(nw):
            for j in range(nX):
                waves[m,j].get_wavefield( particles[n].t,particles[n].r)
                B += waves[m,j].Bw
                E += waves[m,j].Ew
        # print('E0 at t0',E)
        # print('B0 at t0',B)
        for n in range(Np):
            particles[n].boris_push(-dt_taichi[None]/2,E,B)
            
    @ti.kernel
    def simulate():
        # one step

        for n in range(Np):
            # get field
            B =ti.Vector([0.0,0.0,B0])
            E = ti.Vector([0.0,0.0,0.0])
            for m in range(nw):
                for j in range(nX):
                
                    waves[m,j].get_wavefield(particles[n].t,particles[n].r )
                    B += waves[m,j].Bw
                    E += waves[m,j].Ew
            
            particles[n].t += dt_taichi[None] 
            particles[n].leap_frog(dt_taichi[None],E,B)
            

    # TODO: how to add the tqdm/progress bar in taichi kernel?
    @ti.kernel
    def simulate_t():
        for n in range(Np): # This will be Parallelized
            #particles
            for tt in range(Nt): # This will be Serialized
                #time
                B =ti.Vector([0.0,0.0,B0])
                E = ti.Vector([0.0,0.0,0.0])

                for m in range(nw):
                    for j in range(nX):

                        waves[m,j].get_wavefield( particles[n].t,particles[n].r) # get Bw and Ew
                        
                        B += waves[m,j].Bw
                        E += waves[m,j].Ew

                particles[n].t += dt_taichi[None] 
                particles[n].leap_frog(dt_taichi[None],E,B) # change nth particle's p and r
                particles[n].Ep = E
                particles[n].Bp = B
                #print('magnetic field!!',particles[n].Bp)
                #phib = ti.atan2(B[1],B[0])
                phip = ti.atan2(particles[n].p[1],particles[n].p[0])
                particles[n].phi = phip
                if (particles[n].phi < 0):
                    particles[n].phi += 2*ti.math.pi
                # save particle info
                if tt%record_num ==0:
                    
                    p_record_taichi[tt//record_num, n] = particles[n].p
                    r_record_taichi[tt//record_num, n] = particles[n].r
                    phi_record_taichi[tt//record_num, n] = particles[n].phi
                    E_record_taichi[tt//record_num, n] = particles[n].Ep
                    B_record_taichi[tt//record_num, n] = particles[n].Bp
    ###################################################
    # Begin of the main loop
    start_time = time.time()
    init()
    #print(particles[1].r)
    simulate_t()
    


    print('finished')
    print('The init p is',p0)
    # End of the main loop
    ###################################################
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    
    ###################################################

    p_results = p_record_taichi.to_numpy()
    r_results = r_record_taichi.to_numpy()
    Ep_results = E_record_taichi.to_numpy()
    Bp_results = B_record_taichi.to_numpy()
    phi_results = phi_record_taichi.to_numpy()



    with open(id + '/' + 'p_r_phi.npy','wb') as f:
        np.save(f,p_results)
        np.save(f,r_results)
        np.save(f,phi_results)


    with open(id + '/' + 'E_B.npy','wb') as f:
        np.save(f,Ep_results)
        np.save(f,Bp_results)
        #np.save(f,phi_results)
    # save the output info for checking
    with open(id + '/' + 'output.txt', 'w') as f:
        f.write(f"The resonating frequency is {w/(2 * np.pi)} (Hz)\n")

        f.write(f"momentum of resonating particles are: {p0}\n")
        f.write(f"E0 is  {erg2ev(p2e(p0))/1000} keV\n")
        f.write(f"--- %s seconds ---" % (time_used))