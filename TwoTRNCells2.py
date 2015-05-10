
import numpy as np
#import random
import matplotlib.pyplot as mpl

# in its current state, this file returns output from both cells but it is not the output it should have, so there is still a bug I need to fix.

#Python code is very similar to matlab code and I hope it is intuitively clear, though I will add more comments along the way.

# this is the old ODE code from the Matlab file:
###ODE test
#y0, t0 = [1.0j, 2.0], 0
#def f(t, y, arg1):
#    return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
#def jac(t, y, arg1):
#    return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
#
#r = ode(f, jac).set_integrator('zvode', method='bdf', with_jacobian=True)
#r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
#t1 = 10
#dt = 1
#while r.successful() and r.t < t1:
#    r.integrate(r.t+dt)
#print r.y
##print("%g %g" % (r.t, r.y))


#ds = TwoTRNCells(t,s)    
def TwoTRNCells(s,t):  
    
    #this is the python function that determines the differential equations for all the variables. The input is the initial or previous parameter state for all parameters in the vector s, and the timestep t.
    
    #global C, g_ca_lts, g_L
    #global istart, istop, iDC, iDC2, A, A2, T1, T2, tA, tA2
    #global g12, g21

    g_nat = 60.5 
    g_nap = 0.0
    g_kd = 60.0 # 60.5 in the paper?
    g_kt = 5.0 
    g_k2 = .5 
    g_ar = 0.025 


    ds = np.zeros([np.size(s)],float) 
    v = s[0]  # voltage of neuron 1
    v2= s[20] # voltage of neuron 2
    if (istart<t)and(t<istop):
        ix = -iDC 
        ix2= -iDC2 
    else:
        ix = 0 
        ix2= 0 
    
    # this is a 2 ms presynaptic 'spike' which 
    # initiates the synaptic activation.
    if t>tA and t < tA + 2 :
        vpre=0 
    else:
        vpre = -100 
    
    # 1 ms square pulse representing presynaptic spike
    K = 1/(1+ np.exp(-(vpre+50.0)/2))   
    
    ds[12] = T1*K*(1-s[12]) - T2*s[12] 


    # determine timing of square pulse input
    if t>tA2 and t < tA2 + 2:
        vpre2=0 
    else:
        vpre2 = -100 
    
    K2 = 1/(1+ np.exp(-(vpre2+50)/2))   
    ds[32] = T1*K2*(1-s[32]) - T2*s[32] 

    #Regular sodium
    minf_nat = 1/(1 +  np.exp((-v - 38)/10)) 
    if v<= -30:
        tau_m_nat = 0.0125 + .1525* np.exp((v+30)/10) 
    else:
        tau_m_nat = 0.02 + .145* np.exp((-v-30)/10) 
    
    hinf_nat = 1/(1 +  np.exp((v + 58.3)/6.7)) 
    tau_h_nat = 0.225 + 1.125/(1+ np.exp((v+37)/15)) 
    dm_nat = -(1/tau_m_nat) * (s[1] - minf_nat) 
    dh_nat = -(1/tau_h_nat) * (s[2] - hinf_nat) 

    minf_nat2 = 1/(1 +  np.exp((-v2 - 38)/10)) 
    if v2<= -30:
        tau_m_nat2 = 0.0125 + .1525* np.exp((v2+30)/10) 
    else:
        tau_m_nat2 = 0.02 + .145* np.exp((-v2-30)/10) 
    
    hinf_nat2 = 1/(1 +  np.exp((v2 + 58.3)/6.7)) 
    tau_h_nat2 = 0.225 + 1.125/(1+ np.exp((v2+37)/15)) 
    dm_nat2 = -(1/tau_m_nat2) * (s[21] - minf_nat2) 
    dh_nat2 = -(1/tau_h_nat2) * (s[22] - hinf_nat2) 

    #Persistent sodium
    minf_nap = 1/(1+ np.exp((-v-48)/10)) 
    if v<= -40:
        tau_m_nap = 0.025 + .14* np.exp((v+40)/10) 
    else:
        tau_m_nap = 0.02 + .145* np.exp((-v-40)/10) 
    
    dm_nap = -(1/tau_m_nap) * (s[3] - minf_nap) 

    minf_nap2 = 1/(1+ np.exp((-v2-48)/10)) 
    if v2<= -40:
        tau_m_nap2 = 0.025 + .14* np.exp((v2+40)/10) 
    else:
        tau_m_nap2 = 0.02 + .145* np.exp((-v2-40)/10) 
    
    dm_nap2 = -(1/tau_m_nap2) * (s[23] - minf_nap2) 

    #Delayed rectifier
    minf_kd = 1/(1+ np.exp((-v-27)/11.5)) 
    if v<= -10:
        tau_m_kd = 0.25 + 4.35* np.exp((v+10)/10) 
    else:
        tau_m_kd = 0.25 + 4.35* np.exp((-v-10)/10) 
    
    dm_kd = -(1/tau_m_kd) * (s[4] - minf_kd) 

    minf_kd2 = 1/(1+ np.exp((-v2-27)/11.5)) 
    if v2<= -10:
        tau_m_kd2 = 0.25 + 4.35* np.exp((v2+10)/10) 
    else:
        tau_m_kd2 = 0.25 + 4.35* np.exp((-v2-10)/10) 
    
    dm_kd2 = -(1/tau_m_kd2) * (s[24] - minf_kd2) 


    #Transient K = A current, McCormick/Huguenard 1992
    minf_kt = 1/(1+ np.exp((-v-60)/8.5)) 
    tau_m_kt = .185 + .5/( np.exp((v+35.8)/19.7) +  np.exp((-v-79)/12.7)) 
    hinf_kt = 1/(1+ np.exp((v+78)/6)) 
    if v<= -63:
        tau_h_kt = .5/( np.exp((v+46)/5) +  np.exp((-v-238)/37.5)) 
    else:
        tau_h_kt = 9.5 
    
    dm_kt = -(1/tau_m_kt) * (s[5] - minf_kt) 
    dh_kt = -(1/tau_h_kt) * (s[6] - hinf_kt) 

    minf_kt2 = 1/(1+ np.exp((-v2-60)/8.5)) 
    tau_m_kt2 = .185 + .5/( np.exp((v2+35.8)/19.7) +  np.exp((-v2-79)/12.7)) 
    hinf_kt2 = 1/(1+ np.exp((v2+78)/6)) 
    if v2<= -63:
        tau_h_kt2 = .5/( np.exp((v2+46)/5) +  np.exp((-v2-238)/37.5)) 
    else:
        tau_h_kt2 = 9.5 
    
    dm_kt2 = -(1/tau_m_kt2) * (s[25] - minf_kt2) 
    dh_kt2 = -(1/tau_h_kt2) * (s[26] - hinf_kt2) 


    #GK2
    minf_k2 = 1/(1+ np.exp((-v-10)/17)) 
    tau_m_k2 = 4.95 + .5/( np.exp((v-81)/25.6) +  np.exp((-v-132)/18)) 
    hinf_k2 = 1/(1+ np.exp((v+58)/10.6)) 
    tau_h_k2 = 60 + .5/( np.exp((v - 1.33)/200) +  np.exp((-v-130)/7.1)) 
    dm_k2 = -(1/tau_m_k2) * (s[7] - minf_k2) 
    dh_k2 = -(1/tau_h_k2) * (s[8] - hinf_k2) 

    minf_k22 = 1/(1+ np.exp((-v2-10)/17)) 
    tau_m_k22 = 4.95 + .5/( np.exp((v2-81)/25.6) +  np.exp((-v2-132)/18)) 
    hinf_k22 = 1/(1+ np.exp((v2+58)/10.6)) 
    tau_h_k22 = 60 + .5/( np.exp((v2 - 1.33)/200) +  np.exp((-v2-130)/7.1)) 
    dm_k22 = -(1/tau_m_k22) * (s[27] - minf_k22) 
    dh_k22 = -(1/tau_h_k22) * (s[28] - hinf_k22) 

    #  voltage part of C-current ... see integrate_nRT.f
    #  PLEH !>!>!>!?!?!??!
    # if (v < -10.d0):
    #     alpham_kc = (2/37.95)* np.exp((v+50.d0)/11.d0 -(v+53.5)/27) 
    #     betam_kc = 2* np.exp((-v-53.5)/27)-alpham_kc 
    # else:
    #     alpham_kc = 2* np.exp((-v-53.5)/27) 
    #     betam_kc = 0 
    # 
    #     
    #T current, as implemented by Traub 2005, which cites Destexhe 1996
    minf_ca_lts = 1/(1+ np.exp((-v-52)/7.4))    #./ traub
    tau_m_ca_lts = 1 + .33/( np.exp((v+27)/10) +  np.exp((-v-102)/15))  # .33./ ./10
    hinf_ca_lts = 1/(1+ np.exp((v+80)/5))  
    tau_h_ca_lts = 28.3 + .33/( np.exp((v+48)/4) +  np.exp((-v-407)/50)) 
    dm_ca_lts = -(1/tau_m_ca_lts) * (s[9] - minf_ca_lts) 
    dh_ca_lts = -(1/tau_h_ca_lts) * (s[10] - hinf_ca_lts) 

    minf_ca_lts2 = 1/(1+ np.exp((-v2-52)/7.4))    #./ traub
    tau_m_ca_lts2 = 1 + .33/( np.exp((v2+27)/10) +  np.exp((-v2-102)/15))  
    hinf_ca_lts2 = 1/(1+ np.exp((v2+80)/5))  
    tau_h_ca_lts2 = 28.3 + .33/( np.exp((v2+48)/4) +  np.exp((-v2-407)/50)) 
    dm_ca_lts2 = -(1/tau_m_ca_lts2) * (s[29] - minf_ca_lts2) 
    dh_ca_lts2 = -(1/tau_h_ca_lts2) * (s[30] - hinf_ca_lts2) 


    #Anonymous rectifier, AR  Traub 2005 calls this 'h'.  ?!
    minf_ar = 1/(1+ np.exp((v+75)/5.5)) 
    tau_m_ar = 1/( np.exp(-14.6  - .086*v) +  np.exp(-1.87 + .07*v)) 
    dm_ar = -(1/tau_m_ar) * (s[11] - minf_ar) 

    minf_ar2 = 1/(1+ np.exp((v2+75)/5.5)) 
    tau_m_ar2 = 1/( np.exp(-14.6  - .086*v2) +  np.exp(-1.87 + .07*v2)) 
    dm_ar2 = -(1/tau_m_ar2) * (s[31] - minf_ar2) 

    ds[1]=dm_nat 
    ds[2]=dh_nat 
    ds[3]=dm_nap 
    ds[4]=dm_kd 
    ds[5]=dm_kt 
    ds[6]=dh_kt 
    ds[7]=dm_k2 
    ds[8]=dh_k2 
    ds[9]=dm_ca_lts 
    ds[10]=dh_ca_lts 
    ds[11]=dm_ar 
    Ina = (g_nat * (s[1]**3)*s[2]  + g_nap*s[3] )*(v - 50) 
    Ik =  (g_kd * (s[4]**4) + g_kt*(s[5]**4)*s[6] + g_k2*s[7]*s[8]) * (v +100) 
    ICa = (g_ca_lts*(s[9]**2)*s[10]) * (v -125) 
    IAR = (g_ar*s[11])*(v +40) 
    IL =  (g_L)*(v +75) 
    Isyn = A*s[12]*(s[0]-20) 

    ds[21]=dm_nat2 
    ds[22]=dh_nat2 
    ds[23]=dm_nap2 
    ds[24]=dm_kd2 
    ds[25]=dm_kt2 
    ds[26]=dh_kt2 
    ds[27]=dm_k22 
    ds[28]=dh_k22 
    ds[29]=dm_ca_lts2 
    ds[30]=dh_ca_lts2 
    ds[31]=dm_ar2 
    Ina2 = (g_nat * (s[21]**3)*s[22]  + g_nap*s[23] )*(v2 - 50) 
    Ik2 =  (g_kd * (s[24]**4) + g_kt*(s[25]**4)*s[26] + g_k2*s[27]*s[28]) * (v2 +100) 
    ICa2 = (g_ca_lts*(s[29]**2)*s[30]) * (v2 -125) 
    IAR2 = (g_ar*s[31])*(v2 +40) 
    IL2 =  (g_L)*(v2 +75) 
    Isyn2 = A2*s[32]*(s[20]-20) 

    ds[0] =  (-1/C)*( Ina + Ik + ICa + IAR + IL)      - ix -  Isyn  - g21*(v-v2) # neuron 1
    ds[20] = (-1/C)*( Ina2 + Ik2 + ICa2 + IAR2 + IL2) - ix2 - Isyn2 - g12*(v2-v) # neuron 2

    return ds
       
###################### RUN:
       

global C,  g_ca_lts, g_L
global istart, istop, iDC, iDC2, A, A2, T1, T2, tA, tA2 
global g12, g21

tmax = 250          #total run time (seconds? ms?)
g_ca_lts = .75      #gca of .75 with leak of .1 is good; 
g_L = .1            # leak conductance
C = 1.0             # membrance capacitance  uF/cm^2
T1 = 5              #rise time constant  #1e-4/5e-4 is good for b-let.
T2 = 25             #fall time constant  #5e-3 / 20e-3 ?? ~50 ms rise.
# was 1 / 10.  ...?

# these were saved after a 5 s. run with 0 input.
#s0 = si.loadmat('s00.mat')
s0 = np.load('s0.npy') # I loaded and saved the input as a numpy vector .npy.

s0[-1]=0 
s0[20:33]=s0[0:13]  #ICs for cell 2. #python-corrected indices!

#DC pulse
iDC = .1             # uA/cm2   DC  0.25 is ~TR for burst; 
iDC2 = .1 
istart = 10   
istop = np.min([200, tmax - 10])   

#g12=.015 
#g21=.75*g12 

gE = np.arange(0.01,0.02+0.001,0.001) #.015 is a good median value
fA = np.arange(0.5,1.5+0.05,0.05)

#g12I=.015*(.5:1/(length(fA)-1)/2:1);
step = 1.0/(len(fA)-1)/2.0
g12I=.015*np.arange(0.5,1+step,step)
g21I=g12I[::-1] #flip left-right

#Alpha/Beta Synapse
A= .25 
tA= 100 
A2=.18 
tA2=100 

fbase='VaryG12_Ap2' 

# asymmetry of the gap junction
#g21=.015 
#g12=g21*fA[9] # for all fA

tstop = 60
deltat = 0.001
y_now = s0[0]

#print np.size(y_now[11])
v_total1 = np.zeros([int(tstop/deltat)])
v_total2 = np.zeros([int(tstop/deltat)])

# 

g12 = g12I[5]
g21 = g21I[0]
#A2=fA[0]*A what is this ??

for tstep in range(0,int(tstop/deltat)):
    
    # midpoint 'integration' method
    y_halfway = y_now + (deltat/2.0)*(TwoTRNCells(y_now,tstep))
    y_next = y_now + deltat*TwoTRNCells(y_halfway,tstep)
    
    y_now = y_next # move to the next iteration
    #save voltage in array
    v_total1[tstep] = y_now[0]
    v_total2[tstep] = y_now[20]

mpl.figure()
mpl.subplot(2,1,1)
mpl.plot(np.arange(len(v_total1)),v_total1)
mpl.subplot(2,1,2)
mpl.plot(np.arange(len(v_total2)),v_total2,'r')

mpl.show()

#################

# PREVIOUS MATLAB CODE:

# the matlab loop

#g21=.015;
#for i=1:length(fA) # for all GJ asymmetric values
#    g12=g21*fA(i); 
#    for j=1:length(fA)
#        A2=fA(j)*A;
#        tspan = [0 tmax];
#        options=odeset('InitialStep',10^(-3),'MaxStep',10^(-2));
#        [T,S] = ode23(@TwoTRNCells,tspan,s0,options);
#        
#        Vm=[T S(:,1) S(:,21) ];
#        fname = ['Output\' fbase int2str(i) '_' int2str(j)  '.mat'];
#        if ~isempty(fbase)
#            save(fname,'Vm')


#################

#
#if __name__ == '__main__':
#    
#    # Start by specifying the integrator:
#    # use ``vode`` with "backward differentiation formula"
#    S = ode(TwoTRNCells).set_integrator('vode', method='adams',with_jacobian=False)#'bdf')
#    
#    # Set the time range
#    t_start = 10^(-3)#0.0
#    t_final = 0.1#10^(-2)#10.0
#    delta_t = 0.001
#    # Number of time steps: 1 extra for initial condition
#    num_steps = np.floor((t_final - t_start)/delta_t) + 1
#    #num_steps = np.floor((istop - istart)/delta_t) + 1
#
#    # Set initial condition(s): for integrating variable and time!
#    #CA_t_zero = 0.5
#    S.set_initial_value(s0, t_start)#.set_TwoTRNCells_params()
#    S.lrw = 317
#    print S.lrw
#    # Additional Python step: create vectors to store trajectories
#    #t = np.zeros((num_steps, 1))
#    #CA = np.zeros((num_steps, 1))
#    #t[0] = t_start
#    #CA[0] = CA_t_zero
#    
#    #lrw = 317
#    
#    Vm1 = np.arange(0,num_steps)
#    Vm2 = np.arange(0,num_steps)
#    T = np.arange(0,num_steps)
#    
#    # Integrate the ODE(s) across each delta_t timestep
#    tspan = [0,tmax]
#    
#    #k = 1
#    #while S.successful() and k < tspan: #num_steps:
#    #      S.integrate(S.t + delta_t)
#        
##    for i in range(0,len(fA)):
##        g12=g21*fA[i] 
##        for j in range(0,len(fA)):
##            A2=fA[j]*A 
#    s=[] 
#    t=[]
#    while S.successful() and S.t < t_final:
#            S.integrate(S.t + delta_t)
#            print S.t
#        
#        
#        # Store the results to plot later
#        #t[k] = r.t
#        #CA[k] = r.y[0]
#        
#        
#        #T[i]=S.t
#        #Vm1[k]=S.s[0]#[:,0]
#        #Vm2[k]=S.s[20]#[:,20] #indices python-corrected
##k += 1



#data = np.zeros([100,len(s0)],float)
#newstate = s0
#for a in range(0,100):
#    oldstate = newstate 
#    newstate = TwoTRNCells(a,oldstate)
#    data[a,:] = newstate
#    

#for i in range(0,len(fA)):
#    g12=g21*fA[i] 
#    for j in range(0,len(fA)):
#        A2=fA[j]*A 
#        tspan = [0,tmax] 

        #options=odeset('InitialStep',10^(-3),'MaxStep',10^(-2)) 
        #[T,S] = ode23(@TwoTRNCells,tspan,s0,options) 

        #Vm=[T, S[:,0], S[:,20]] #indices python-corrected
        #fname = ['Output\' fbase int2str(i) '_' int2str(j)  '.mat'] 
        #if ~isempty(fbase)
            #save(fname,'Vm')
    

#         
# figure(1);clf
# subplot(2,1,1);cla; hold on;
#     plot(T,[S(:,1) S(:,21)]);
#     xlabel('time'), ylabel('Voltage')
#     inp=-80*ones(size(T));
#     inp((T>istart)&(T<istop))=-80+10*iDC;
#     plot(T,inp,'k','linewidth',2)
#     
# subplot(2,1,2); cla;hold on;
#     plot(T,S(:,13).*S(:,1),'b:')  #synaptic inputs in cell 1.
#     plot(T,S(:,33).*S(:,21),'g:')  #synaptic inputs in cell 1.
#    # set(gca,'xlim',[tA-5 tA+50])

# figure(2);clf;
#   for i=1:20
# foo=Vm{i};
# plot(foo(:,1),foo(:,2)+(i-1)*100,'color',colorn(i))
# hold on
# end  

