import argparse
import numpy as np
from collections import defaultdict

def display(objective, D1, D2, D1_gradients, D2_gradients, param_gradients):
    print("[Objective Function Value]:", objective)

    print("\n[D1 Values]:")
    for nt, val in D1.items():
        print(f"  D1_{nt}: {val}")

    print("\n[D2 Values]:")
    for nt, val in D2.items():
        print(f"  D2_{nt}: {val}")

    print("\n[D1 Gradients]:")
    for nt, grad in D1_gradients.items():
        print(f"  Gradient D1_{nt}: {grad}")

    print("\n[D2 Gradients]:")
    for nt, grad in D2_gradients.items():
        print(f"  Gradient D2_{nt}: {grad}")

    print("\n[Parameter Gradients]:")
    for pair, grad in param_gradients.items():
        print(f"  Gradient {pair}: {grad}")

# Define Q function
Q = defaultdict(lambda: np.exp(-2), {
    "GC": np.exp(3) + np.exp(-2),
    "CG": np.exp(3) + np.exp(-2),
    "AU": np.exp(2) + np.exp(-2),
    "UA": np.exp(2) + np.exp(-2),
    "GU": np.exp(1) + np.exp(-2),
    "UG": np.exp(1) + np.exp(-2)
})

score = defaultdict(lambda: 0., {
    "GC": -3,
    "CG": -3,
    "AU": -2,
    "UA": -2,
    "GU": -1,
    "UG": -1
})

def expected_free_energy(D1, D2):
    objective = sum(D1[nt1] * D2[nt2] * score[nt1 + nt2] for nt1 in D1 for nt2 in D2)

    # D1 = {A: .5, C: .5, G: .0, U: .0}
    # D2 = {A: 0., C: 0., G: .5, U: .5}

    # GC = D2["C"] * score['GC'] + D2["U"] * score['GU'] + D1["G"] * score['GC']
    # UA = D2["A"] * score['UA'] + D2["G"] * score['UG'] + D1["U"] * score['UA']
    
    grad_D1_A = D2["U"] * score['AU']
    grad_D1_C = D2["G"] * score['CG']
    grad_D1_G = D2["C"] * score['GC'] + D2["U"] * score['GU']
    grad_D1_U = D2["A"] * score['UA'] + D2["G"] * score['UG']

    grad_D2_A = D1["U"] * score['UA']
    grad_D2_C = D1["G"] * score['GC']
    grad_D2_G = D1["C"] * score['CG'] + D1["U"] * score['UG']
    grad_D2_U = D1["A"] * score['AU'] + D1["G"] * score['GU']

    grad_theta_AU = grad_D1_A + grad_D2_U
    grad_theta_UA = grad_D1_U + grad_D2_A
    grad_theta_CG = grad_D1_C + grad_D2_G
    grad_theta_GC = grad_D1_G + grad_D2_C
    grad_theta_GU = grad_D1_G + grad_D2_U
    grad_theta_UG = grad_D1_U + grad_D2_G

    D1_gradients = {
        "A": grad_D1_A,
        "C": grad_D1_C,
        "G": grad_D1_G,
        "U": grad_D1_U
    }
    D2_gradients = {
        "A": grad_D2_A,
        "C": grad_D2_C,
        "G": grad_D2_G,
        "U": grad_D2_U
    }
    param_gradients = {
        "AU": grad_theta_AU,
        "UA": grad_theta_UA,
        "CG": grad_theta_CG,
        "GC": grad_theta_GC,
        "GU": grad_theta_GU,
        "UG": grad_theta_UG
    }

    return objective, D1_gradients, D2_gradients, param_gradients

def calculate_original_objective_and_gradients(theta_values):
    theta_AU, theta_UA, theta_CG, theta_GC, theta_GU, theta_UG = theta_values

    # Calculate Marginal D1 and D2
    # D1 = {
    #     "A": theta_AU,
    #     "C": theta_CG,
    #     "G": theta_GC + theta_GU,
    #     "U": theta_UA + theta_UG
    # }

    # D2 = {
    #     "A": theta_UA,
    #     "C": theta_GC,
    #     "G": theta_CG + theta_UG,
    #     "U": theta_AU + theta_GU
    # }

    D1 = {
        "A": .5,
        "C": .5,
        "G": 0.,
        "U": 0.
    }

    D2 = {
        "A": .5,
        "C": .5,
        "G": 0.,
        "U": 0.
    }

    objective = sum(D1[nt1] * D2[nt2] * np.log(Q[nt1 + nt2]) for nt1 in D1 for nt2 in D2)

    # Compute the gradients
    grad_D1_A = D2["U"] * np.log(Q["AU"]) + (1 - D2["U"]) * np.log(Q["XX"])
    grad_D1_C = D2["G"] * np.log(Q["CG"]) + (1 - D2["G"]) * np.log(Q["XX"])
    grad_D1_G = D2["C"] * np.log(Q["GC"]) + D2["U"] * np.log(Q["GU"]) + (1 - D2["C"] - D2["U"]) * np.log(Q["XX"])
    grad_D1_U = D2["A"] * np.log(Q["UA"]) + D2["G"] * np.log(Q["UG"]) + (1 - D2["A"] - D2["G"]) * np.log(Q["XX"])

    grad_D2_A = D1["U"] * np.log(Q["UA"]) + (1 - D1["U"]) * np.log(Q["XX"])
    grad_D2_C = D1["G"] * np.log(Q["GC"]) + (1 - D1["G"]) * np.log(Q["XX"])
    grad_D2_G = D1["C"] * np.log(Q["CG"]) + D1["U"] * np.log(Q["UG"]) + (1 - D1["C"] - D1["U"]) * np.log(Q["XX"])
    grad_D2_U = D1["A"] * np.log(Q["AU"]) + D1["G"] * np.log(Q["GU"]) + (1 - D1["A"] - D1["G"]) * np.log(Q["XX"])



    grad_theta_AU = (D1["A"] + D2["U"]) * np.log(Q["AU"]) + D1["G"] * np.log(Q["GU"]) \
                    + (2 - D1["A"] - D1["G"] - D2["U"]) * np.log(Q["XX"])

    grad_theta_UA = (D1["U"] + D2["A"]) * np.log(Q["UA"]) + D2["G"] * np.log(Q["UG"]) \
                    + (2 - D2["A"] - D2["G"] - D1["U"]) * np.log(Q["XX"])

    grad_theta_CG = (D1["C"] + D2["G"]) * np.log(Q["CG"]) + D1["U"] * np.log(Q["UG"]) \
                    + (2 - D1["C"] - D1["U"] - D2["G"]) * np.log(Q["XX"])

    grad_theta_GC = (D1["G"] + D2["C"]) * np.log(Q["GC"]) + D2["U"] * np.log(Q["GU"]) \
                    + (2 - D2["C"] - D2["U"] - D1["G"]) * np.log(Q["XX"])

    grad_theta_GU = (D1["G"] + D2["U"]) * np.log(Q["GU"]) + D1["A"] * np.log(Q["AU"]) + D2["C"] * np.log(Q["GC"]) \
                    + (2 - D1["A"] - D1["G"] - D2["C"] - D2["U"]) * np.log(Q["XX"])

    grad_theta_UG = (D1["U"] + D2["G"]) * np.log(Q["UG"]) + D1["C"] * np.log(Q["CG"]) + D2["A"] * np.log(Q["UA"]) \
                    + (2 - D2["A"] - D2["G"] - D1["C"] - D1["U"]) * np.log(Q["XX"])

    # Print the gradients in a pretty way
    D1_gradients = {
        "A": grad_D1_A,
        "C": grad_D1_C,
        "G": grad_D1_G,
        "U": grad_D1_U,
    }
    D2_gradients = {
        "A": grad_D2_A,
        "C": grad_D2_C,
        "G": grad_D2_G,
        "U": grad_D2_U
    }
    param_gradients = {
        "AU": grad_theta_AU,
        "UA": grad_theta_UA,
        "CG": grad_theta_CG,
        "GC": grad_theta_GC,
        "GU": grad_theta_GU,
        "UG": grad_theta_UG
    }

    print('='*64)
    print("Original Objective: E[log Q(x)]")
    print('='*64)
    display(objective, D1, D2, D1_gradients, D2_gradients, param_gradients)
    print('-'*64)
    print('')

    print('='*64)
    print("Original Full Objective: E[log Q(x)] + E[Delta G(x, y)]")
    print('='*64)
    Delta_G, D1_grad_Delta_G, D2_grad_Delta_G, param_grad_Delta_G = expected_free_energy(D1, D2)
    print("Delta G: ", Delta_G)
    objective += Delta_G
    for nuc, grad in D1_grad_Delta_G.items():
        D1_gradients[nuc] += grad
    for nuc, grad in D2_grad_Delta_G.items():
        D2_gradients[nuc] += grad
    for nuc, grad in param_grad_Delta_G.items():
        param_gradients[nuc] += grad
    print(param_grad_Delta_G)
    display(objective, D1, D2, D1_gradients, D2_gradients, param_gradients)
    print('-'*64)
    print('')

def calculate_approximated_objective_and_gradients(theta_values):
    theta_AU, theta_UA, theta_CG, theta_GC, theta_GU, theta_UG = theta_values

    # Calculate Marginal D1 and D2
    D1 = {
        "A": theta_AU,
        "C": theta_CG,
        "G": theta_GC + theta_GU,
        "U": theta_UA + theta_UG
    }

    D2 = {
        "A": theta_UA,
        "C": theta_GC,
        "G": theta_CG + theta_UG,
        "U": theta_AU + theta_GU
    }

    # Calculate the approximated objective
    expobjB = sum(D1[nt1] * D2[nt2] * Q[nt1 + nt2] for nt1 in D1 for nt2 in D2)
    objective = np.log(expobjB)

    # Compute the gradients for D1 and D2
    grad_D1_A = (D2["U"] * Q["AU"] + (1 - D2["U"]) * Q["XX"]) / expobjB
    grad_D1_C = (D2["G"] * Q["CG"] + (1 - D2["G"]) * Q["XX"]) / expobjB
    grad_D1_G = (D2["C"] * Q["GC"] + D2["U"] * Q["GU"] + (1 - D2["C"] - D2["U"]) * Q["XX"]) / expobjB
    grad_D1_U = (D2["A"] * Q["UA"] + D2["G"] * Q["UG"] + (1 - D2["A"] - D2["G"]) * Q["XX"]) / expobjB

    grad_D2_A = (D1["U"] * Q["UA"] + (1 - D1["U"]) * Q["XX"]) / expobjB
    grad_D2_C = (D1["G"] * Q["GC"] + (1 - D1["G"]) * Q["XX"]) / expobjB
    grad_D2_G = (D1["C"] * Q["CG"] + D1["U"] * Q["UG"] + (1 - D1["C"] - D1["U"]) * Q["XX"]) / expobjB
    grad_D2_U = (D1["A"] * Q["AU"] + D1["G"] * Q["GU"] + (1 - D1["A"] - D1["G"]) * Q["XX"]) / expobjB

    D1_gradients = {
        "A": grad_D1_A,
        "C": grad_D1_C,
        "G": grad_D1_G,
        "U": grad_D1_U,
    }
    D2_gradients = {
        "A": grad_D2_A,
        "C": grad_D2_C,
        "G": grad_D2_G,
        "U": grad_D2_U
    }

    # Compute the gradients for theta parameters
    param_gradients = {
        "AU": ((D1["A"] + D2["U"]) * Q["AU"] + D1["G"] * Q["GU"]
               + (2 - D1["A"] - D1["G"] - D2["U"]) * Q["XX"]) / expobjB,
        "UA": ((D1["U"] + D2["A"]) * Q["UA"] + D2["G"] * Q["UG"]
               + (2 - D2["A"] - D2["G"] - D1["U"]) * Q["XX"]) / expobjB,
        "CG": ((D1["C"] + D2["G"]) * Q["CG"] + D1["U"] * Q["UG"]
               + (2 - D1["C"] - D1["U"] - D2["G"]) * Q["XX"]) / expobjB,
        "GC": ((D1["G"] + D2["C"]) * Q["GC"] + D2["U"] * Q["GU"]
               + (2 - D2["C"] - D2["U"] - D1["G"]) * Q["XX"]) / expobjB,
        "GU": ((D1["G"] + D2["U"]) * Q["GU"] + D1["A"] * Q["AU"] + D2["C"] * Q["GC"]
               + (2 - D1["A"] - D1["G"] - D2["C"] - D2["U"]) * Q["XX"]) / expobjB,
        "UG": ((D1["U"] + D2["G"]) * Q["UG"] + D1["C"] * Q["CG"] + D2["A"] * Q["UA"]
               + (2 - D2["A"] - D2["G"] - D1["C"] - D1["U"]) * Q["XX"]) / expobjB
    }


    print('='*64)
    print("Approximated Objective: log E[Q(x)]")
    print('='*64)
    display(objective, D1, D2, D1_gradients, D2_gradients, param_gradients)
    print('-'*64)
    print('')

    print('='*64)
    print("Approximated Full Objective: log E[Q(x)] + E[Delta G(x, y)]")
    print('='*64)
    Delta_G, D1_grad_Delta_G, D2_grad_Delta_G, param_grad_Delta_G = expected_free_energy(D1, D2)
    objective += Delta_G
    for nuc, grad in D1_grad_Delta_G.items():
        D1_gradients[nuc] += grad
    for nuc, grad in D2_grad_Delta_G.items():
        D2_gradients[nuc] += grad
    for nuc, grad in param_grad_Delta_G.items():
        param_gradients[nuc] += grad
    display(objective, D1, D2, D1_gradients, D2_gradients, param_gradients)
    print('-'*64)
    print('')

def main():
    parser = argparse.ArgumentParser(description="Calculate gradients for given theta values.")
    parser.add_argument('thetas', metavar='T', type=float, nargs=6,
                        help='The theta values in the order: AU, UA, CG, GC, GU, UG')

    args = parser.parse_args()

    theta_values = args.thetas
    if len(theta_values) != 6:
        parser.error("Six theta values are required: AU, UA, CG, GC, GU, UG")

    calculate_original_objective_and_gradients(theta_values)
    calculate_approximated_objective_and_gradients(theta_values)


if __name__ == "__main__":
    main()