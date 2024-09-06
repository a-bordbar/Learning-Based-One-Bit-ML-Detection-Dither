import numpy as np 
from itertools import product
from scipy.stats import norm
from multiprocessing import Pool
import concurrent.futures

#Setting the simulation parameters

Nu = 4 #Numnber of users 
Nr = 32 #Number of BS Antennas
Ntr = 45 #Pilot signal repetition 
M = 4   #16-QAM constellation
Ns = 1  #Number of sub-blocks 
#First, I create a dictionary for all 16-QAM symbols 

symbol_map =[ complex(1, 1),
                  complex(1, 3),
                   complex(3, 1),
                  complex(3, 3),
                   complex(-1, 1),
                   complex(-1, 3),
                   complex(-3, 1),
                   complex(-3, 3),
                   complex(1, -1),
                   complex(1, -3),
                   complex(3, -1),
                   complex(3, -3),
                   complex(-1, -1),
                   complex(-1, -3),
                  complex(-3, -1),
                   complex(-3, -3)] /np.sqrt(10)



symbol_map =[ complex(1, 1),
            complex(1, -1),
            complex(-1, 1),
            complex(-1, -1),
                 ]/np.sqrt(2)

#pilot = np.array(list(product(symbol_map)))  #Generate a [M**Nu ,Nu] matrix 
#pilot = np.array(list(product(symbol_map, symbol_map, symbol_map)))  #Generate a [M**Nu ,Nu] matrix 

pilot = np.array(list(product(symbol_map, symbol_map, symbol_map, symbol_map)))  #Generate a [M**Nu ,Nu] matrix 

pilot_repeat = np.repeat(np.expand_dims(pilot, -1) , Ntr ,axis = -1)  # Repeat the pilot matrix along a new axis. Size = [M**Nu , Nu, Ntr]

num_points = 10
SNR_dB_vector = np.linspace(-10, 10, num_points)



#SNR_dB = 4
N0_dB = 0
N0 = 10**(N0_dB/10) 
rho_dB_vector = SNR_dB_vector
rho = 10**(rho_dB_vector/10)
Ed =  rho /1
delta = rho/3



#Training Phase 
#Adaptive dithering is not implemented yet
















    
SER_vec = np.zeros([num_points ,1])               



for snr in range(num_points):    
    if snr <3:
        Ntrial = 1e2
    elif snr <5:
        Ntrial = 1e3
    elif snr <7:
        Ntrial = 1e5
    else:
        Ntrial = 1e6
    

    
    P_plus_D = np.zeros([M**Nu, 2*Nr] ,dtype = np.float64)
    P_plus = np.zeros([M**Nu, 2*Nr] ,dtype = np.float64)
    P_minus = np.zeros([M**Nu, 2*Nr] ,dtype = np.float64)
    Psi = np.zeros([M**Nu, 2*Nr] ,dtype = np.float64)
    quantized_bits = np.zeros([M**Nu, 2*Nr , Ntr] ,dtype = np.float64)
    quantized_bits = np.zeros((M**Nu, 2*Nr, Ntr))
    P_plus_D = np.zeros((M**Nu, 2*Nr))
    Psi = np.zeros((M**Nu, 2*Nr))
    P_plus = np.zeros((M**Nu, 2*Nr))   
    
    
    
    H_complex = np.sqrt(0.5) * (np.random.randn(Nr , Nu) + 1j*np.random.randn(Nr , Nu))

    H_real = np.block([[np.real(H_complex) , -np.imag(H_complex)] ,
                   [np.imag(H_complex) , np.real(H_complex)]])

    for i in range(M**Nu): 
        for k in range(2*Nr): 
            count1 = 0
            for j in range(Ntr):
                s_complex = np.expand_dims(pilot_repeat[i , : , j], -1) *np.sqrt(rho[snr])
                s_real = np.concatenate([np.real(s_complex) , np.imag(s_complex)])
                noise = np.sqrt(N0/2) * (np.random.randn(2*Nr , 1) )
                r_real = H_real@s_real +noise
                dither_real= np.sqrt(Ed[snr]/2) * (np.random.randn(2*Nr , 1) )
                y_real = (np.sign(r_real + dither_real))
            
                quantized_bits[i , k, j] = y_real[k]
            count1 = (np.sum(quantized_bits[i , k, :] +1)/2)
            P_plus_D[i, k] = (1/(Ntr)) * count1  
                # if count1 == Ntr or count1 ==0: # This does not work
                #     Ed += delta
                #     print("Delta")   
            Psi[i , k] = norm.ppf(P_plus_D[i, k]) * np.sqrt(1 + (1*Ed[snr]/N0))
            P_plus [i , k] = norm.cdf(Psi[i , k])
    P_minus = 1-P_plus


    undertrained = (np.abs(Psi)==np.inf).sum()/(M**Nu)

    print("Average Number of Undertrained Likelihoods = {}".format(undertrained))

    S = np.zeros([int(Ntrial), 2*Nu , 1], dtype = np.float64)
    S_HAT = np.zeros([int(Ntrial), 2*Nu , 1], dtype = np.float64)

    #Testing 
    P_test = np.zeros([M**Nu])
    for n in range(int(Ntrial)):
        #random_idx = np.random.choice(range(0, M),Nu)
        random_idx = np.random.choice(range(0, M**Nu),1)
        #k = random_idx[0]*(M**(Nu-1)) +random_idx[1]*(M**(Nu-2)) + random_idx[2]
        #kk = random_idx[0]*(M**(Nu-1)) +random_idx[1]*(M**(Nu-2)) + random_idx[2] + (M**(Nu-3)) + random_idx[3]  #4 Users
        #kk = random_idx[0]*(M**(Nu-1)) +random_idx[1]*(M**(Nu-2)) + random_idx[2]  #3 Users

        #s_test = np.expand_dims( np.array([   symbol_map[random_idx[0]] , symbol_map[random_idx[1]]  ,symbol_map[random_idx[2]]      , symbol_map[random_idx[3]]      ]) , -1) *np.sqrt(rho) # 4Users
        #s_test = np.expand_dims( np.array([   symbol_map[random_idx[0]] , symbol_map[random_idx[1]]  ,symbol_map[random_idx[2]]            ]) , -1) *np.sqrt(rho) # 3 Users
        #s_test = np.expand_dims( pilot[random_idx, :]  , -1) *np.sqrt(rho)
        s_test = pilot[random_idx, :].T *np.sqrt(rho[snr])

        s_test = np.concatenate([np.real(s_test) , np.imag(s_test)])
        r_test = H_real@s_test + np.sqrt(N0/2) * (np.random.randn(2*Nr , 1))
        y_test =  np.sign(r_test) 
        ans = y_test #np.concatenate([np.real(y_test) , np.imag(y_test)])

        for i in range(M**Nu):
            temp = 1
            for k in range(2*Nr):
                temp *= (P_plus[i, k] * np.int64(y_test[k,0] > 0) + P_minus[i, k] * np.int64(y_test[k,0] < 0))  #This is complex
            P_test[i] = temp
        k_hat = np.argmax(P_test)
        s_hat = np.expand_dims(pilot[k_hat,:], -1)
        s_hat = np.concatenate([np.real(s_hat) , np.imag(s_hat)])

        S[n, : , :] = s_test
        S_HAT[n,:,:] = s_hat *np.sqrt(rho[snr])

        
    err = np.abs(S - S_HAT)
    SER = (err != 0+0j).sum() / (Ntrial *2*Nu)
    #MSE = 10*np.log10(np.mean(np.abs(S-S_HAT)**2))
    SER_vec[snr, :] = SER
    for i in range(10):
        if SER*10**i // 1 != 0:
            print( "SNR_dB: ",SNR_dB_vector[snr], "",   " SER:" , SER*10**(i)  , "1e{}".format(-i))
            break
np.savetxt("./SER_VECTOR.txt", SER_vec)
print("Over")