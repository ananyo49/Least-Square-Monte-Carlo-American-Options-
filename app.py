import streamlit as st
import numpy as np

# Function to generate stock paths
def stock_path_generator(s0, r, sig, paths, timesteps, T):
    stock_paths = np.zeros((paths, timesteps))
    delta = T / timesteps
    for i in range(0, paths, 2):
        z_pos = np.random.normal(0, 1, timesteps)
        z_neg = -z_pos
        stock_paths[i, :] = s0 * np.exp((r - 0.5 * sig**2) * delta + sig * np.sqrt(delta) * np.cumsum(z_pos))
        stock_paths[i + 1, :] = s0 * np.exp((r - 0.5 * sig**2) * delta + sig * np.sqrt(delta) * np.cumsum(z_neg))
    initial_stock_price = np.full((paths, 1), s0)
    stock_paths = np.hstack((initial_stock_price, stock_paths))
    return stock_paths

# Polynomial basis functions
def laguerre_upto_k(x, k):
    f1 = np.exp(-x / 2)
    f2 = f1 * (1 - x)
    f3 = f1 * (1 - 2 * x + 0.5 * x**2)
    f4 = f1 * (1 - 3 * x + 1.5 * x**2 - x**3 / 6)
    if k == 2:
        return np.column_stack((f1, f2))
    elif k == 3:
        return np.column_stack((f1, f2, f3))
    elif k == 4:
        return np.column_stack((f1, f2, f3, f4))

def hermite_upto_k(x, k):
    f1 = 1
    f2 = 2 * x
    f3 = 4 * (x**2) - 2
    f4 = 8 * (x**3) - 12 * x
    if k == 2:
        return np.column_stack((f1, f2))
    elif k == 3:
        return np.column_stack((f1, f2, f3))
    elif k == 4:
        return np.column_stack((f1, f2, f3, f4))

def monomials_upto_k(x, k):
    f1 = 1
    f2 = x
    f3 = x**2
    f4 = x**3
    if k == 2:
        return np.column_stack((f1, f2))
    elif k == 3:
        return np.column_stack((f1, f2, f3))
    elif k == 4:
        return np.column_stack((f1, f2, f3, f4))

# American Put Option Pricing Function
def american_put_price_lsmc(s0, r, sig, paths, timesteps, T, func, strike, k):
    stock = stock_path_generator(s0, r, sig, paths, timesteps, T)
    index = np.zeros(stock.shape)
    Y = np.zeros(stock.shape)
    exercisevalues = np.maximum(strike - stock, 0)
    Y[:, -1] = exercisevalues[:, -1]
    index[Y[:, -1] > 0, -1] = 1
    delta = T / timesteps
    for j in range(timesteps - 2, -1, -1):
        Y[:, j] = np.sum(index[:, j + 1:] * np.exp(-r * np.arange(1, timesteps - j) * delta) * exercisevalues[:, j + 1:], axis=1)
        exercise_positions = np.where(Y[:, j] == 0)[0]
        index[exercise_positions, j] = 1
        Y[exercise_positions, j] = exercisevalues[exercise_positions, j]
        exercise_positions = np.where(Y[:, j] == 0)[0]
        index[exercise_positions, j] = 0
        nodes = np.where(Y[:, j] > 0)[0]
        ys = Y[nodes, j]
        xs = stock[nodes, j]
        if func == "L":
            L_xs = laguerre_upto_k(xs, k)
        elif func == "H":
            L_xs = hermite_upto_k(xs, k)
        elif func == "M":
            L_xs = monomials_upto_k(xs, k)
        A = L_xs.T @ L_xs
        b = L_xs.T @ ys
        a = np.linalg.solve(A, b)
        ecv = np.zeros(Y.shape[0])
        ecv[nodes] = L_xs @ a
        ev = np.zeros(Y.shape[0])
        ev[nodes] = exercisevalues[nodes, j]
        index[ev > ecv, j] = 1
        index[ev <= ecv, j] = 0
        Y[:, j] = np.maximum(ev, Y[:, j])
    payoff_each_path = np.sum(np.exp(-r * np.arange(timesteps) * delta) * index * exercisevalues, axis=1)
    return np.mean(payoff_each_path)

# Streamlit App
st.title('American Option Pricing Using LSMC Method')

# User Inputs
s0 = st.number_input('Initial Stock Price (s0)', value=36.0)
r = st.number_input('Risk-Free Rate (r)', value=0.06)
sig = st.number_input('Volatility (sig)', value=0.2)
paths = st.number_input('Number of Simulation Paths', value=100000)
timesteps = st.number_input('Number of Time Steps', value=126)
T = st.number_input('Time to Maturity (T)', value=0.5)
strike = st.number_input('Strike Price', value=40.0)
k = st.number_input('Degree of Basis Functions (k)', value=2)
func = st.selectbox('Basis Function Type', options=['L', 'H', 'M'])

# Button to Calculate Option Price
if st.button('Calculate American Put Option Price'):
    price = american_put_price_lsmc(s0, r, sig, paths, timesteps, T, func, strike, k)
    st.write(f'Estimated American Put Option Price: ${price:.2f}')
