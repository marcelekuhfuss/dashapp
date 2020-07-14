
import numpy as np
import statistics 
from numpy.linalg import inv
import plotly.graph_objects as go
import pdb
import AS
import dash
import dash_core_components as dcc
import dash_html_components as html
################################################################################################
def norm_comp(G):
    K = G.shape[1]
    for i in range(K):
        aux = np.dot(G[:,i].conj().T,G[:,i])
        # if aux==0:
        #     print(aux)
        G[:,i] = G[:,i]/np.sqrt(aux)
    return G

def QAM_mod(number_of_symbols):
    if number_of_symbols==4:
        map_table = {
        (0,0) : 1+1j,
        (0,1) : 1-1j,
        (1,0) : -1+1j,
        (1,1) : -1-1j      
        }
        demap_table = {v : k for k, v in map_table.items()}
    if number_of_symbols==16:
        map_table = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) : 3-3j,
        (1,0,0,1) : 3-1j,
        (1,0,1,0) : 3+3j,
        (1,0,1,1) : 3+1j,
        (1,1,0,0) : 1-3j,
        (1,1,0,1) : 1-1j,
        (1,1,1,0) : 1+3j,
        (1,1,1,1) : 1+1j        
        }


        demap_table = {v : k for k, v in map_table.items()}
    return map_table,demap_table

def Mapping(bits,mapping_table,K,len_mes,mu):
    bits = bits.reshape((K*len_mes, mu))
    q = np.array([mapping_table[tuple(b)] for b in bits])
    return q.reshape((K,len_mes))

def Demapping(QAM,demapping_table):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision])

def SL_MPAS(D,M,S,K,q):

    residue = q
    x = np.zeros((M,),dtype=complex)

    if S < K:
        alpha = 1
    else:
        alpha = (K/S)*(K/(K+20))

    for i in range(S):
        aux = np.zeros((M,1),dtype=complex)
        for k in range(M):
            v = D[:,k]
            v = np.reshape(v,(1,K))
            aux[k]= np.dot(residue,v.conj().T)

        paux = abs(aux) 
        ind = np.unravel_index(np.argmax(paux, axis=None), paux.shape)[0]

        v_rm = (D[:,ind].T)
        v_rm = np.reshape(v_rm,(1,K))
        inner = alpha*np.dot(residue,v_rm.conj().T)

        residue   =  residue  - inner * D[:,ind].T
        D[:,ind] = np.zeros((K,),dtype=complex)
        
        x[ind] = inner
    


    return x



################################################################################################
# Basic SET-UP
################################################################################################
K            =  12                           # number of terminals
M            =  100                         # number of BS antennas
S_all        =  np.arange(40,70,10)  
# S            =  40                         # number of selected antennas
ensemble     =  100                         # number of Monte Carlo runs
snr_range      = np.arange(-12, 15, 3)        # range of SNR
len_mes      = 48                           # message length 
methods      = 6                            # total of AS methods  
tt           = 3                            # number of SL-AS methods
var_noise_rx = 0.9                          # noise variance
mu           = 2                            # number of bits per symbol
################################################################################################
# Channel  matrix: 
G = np.random.randn(M,K) + 1j*np.random.randn(M,K)  

# Bit stream to be transmitted:
bits = np.random.binomial(n=1, p=0.5, size=(K*len_mes*mu, ))
[mapping_table,demapping_table] = QAM_mod(2**mu)
# QAM symbols:
q = Mapping(bits,mapping_table,K,len_mes,mu)

# Normalizing columns of channel matrix G:
G = norm_comp(G)

Dic = norm_comp((G.T)*2/2)
# Precoding matrix:
aux = np.dot(G.T,G.conj())
A_full = np.dot(G.conj(),inv(aux))

myber2 = []
for step in range(len(S_all)):
# Main Loop:
    ber = np.zeros(len(snr_range))
    ber2 = np.zeros(len(snr_range))

    for s in range(len(snr_range)):
        SNR = snr_range[s]
        var_y_full = (var_noise_rx*(10**(SNR/10)))
        aux_var = np.sqrt(M*var_y_full) 

        X_full = np.zeros((M,len_mes),dtype=complex)
        Y_full = np.zeros((K,len_mes),dtype=complex)
        X_slmpas = np.zeros((M,len_mes),dtype=complex)
        Y_slmpas = np.zeros((K,len_mes),dtype=complex)
        q_aux = np.zeros((K,len_mes),dtype=complex)

        for l in range(len_mes):
            noise =  (1/np.sqrt(2))*np.sqrt(var_noise_rx)*(np.random.randn(K,) + 1j*np.random.randn(K,))
            Dic = norm_comp((G.T)*2/2)
            X_full[:,l] = np.dot(A_full,q[:,l])
            varr = np.var(X_full[:,l])
            factor_pot = (1/np.sqrt(varr*M))*aux_var
            X_full[:,l] = X_full[:,l] *factor_pot
            Y_full[:,l] = (np.dot(G.T,X_full[:,l]) + noise)/factor_pot

            q_aux[:,l] = q[:,l]/np.linalg.norm(q[:,l])
            X_slmpas[:,l] = SL_MPAS(Dic,M,S_all[step],K,q_aux[:,l])
            varr2  = np.var(X_slmpas[:,l])        
            factor_pot2 = (1/np.sqrt(varr2*M))*aux_var
            X_slmpas[:,l] = X_slmpas[:,l] *factor_pot2
            Y_slmpas[:,l] = (np.dot(G.T,X_slmpas[:,l])+noise)/factor_pot2

        bits_est = Demapping(Y_full,demapping_table)
        bits_est = bits_est.reshape((-1,))
        ber[s] = np.sum(abs(bits-bits_est))/len(bits) 

        bits_est2 = Demapping(Y_slmpas,demapping_table)
        bits_est2 = bits_est2.reshape((-1,))
        ber2[s] = np.sum(abs(bits-bits_est2))/len(bits)  


    ber[ber==0] = np.nan
    ber2[ber2==0] = np.nan

    myber2.append(ber2)

# pdb.set_trace() 

# trace = go.Scatter(
#     x=snr_range,
#     y=ber
# )

# trace2 = go.Scatter(
#     x=snr_range,
#     y=ber2
# )

fig = go.Figure()

for step in range(len(S_all)):

    fig.add_trace(
            go.Scatter(
                visible=False,
                name = "SL_MPPAS with S =" + str(50+10*step),
                x=snr_range,
                y=myber2[step]))

fig.add_trace(
            go.Scatter(
                visible=True,
                name = "Full",
                x=snr_range,
                y=ber))

# layout
layout = go.Layout(yaxis=dict(range=[0, 1]))

fig.data[0].visible = True


# Plot
# fig = go.Figure(data=data, layout=layout)

fig.update_yaxes(type="log", range=[np.log10(10**(-5)), np.log10(1)])
# fig.show()

# Create and add slider
steps = []
for i in range(len(fig.data)-1):
    step = dict(
        method="update",
        label= "S="+ str(50+10*i),
        args=[{"visible": [False, False, False, True]},
              {"title": " S = " + str(50+10*i)}],  # layout attribute
        
    )

    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)



sliders = [dict(

    active=10,
    currentvalue={"prefix": "Number of selected antennas, : "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)




app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter




