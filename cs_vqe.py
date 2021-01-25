#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

###################################################
#   William M. Kirby, 2021
#   Theory references: https://arxiv.org/abs/1904.02260, https://arxiv.org/abs/2002.05693, and https://arxiv.org/abs/2011.10027.
###################################################

#   What is included:
#       Test for contextuality of a Hamiltonian.
#       Construction of quasi-quantized (classical) models for noncontextual Hamiltonians.
#       Heuristics to approximate optimal noncontextual sub-Hamiltonians given arbitrary target Hamiltonians.
#       Classical simulation of quantum correction to a noncontextual approximation.
#       Classical simulation of contextual subspace VQE.

import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
import itertools
from functools import reduce
import random
from datetime import datetime
from datetime import timedelta
from copy import deepcopy

# Takes two Pauli operators specified as strings (e.g., 'XIZYZ') and determines whether they commute:
def commute(x,y):
    assert len(x)==len(y), print(x,y)
    s = 1
    for i in range(len(x)):
        if x[i]!='I' and y[i]!='I' and x[i]!=y[i]:
            s = s*(-1)
    if s==1:
        return 1
    else:
        return 0

# Input: S, a list of Pauli operators specified as strings.
# Output: a boolean indicating whether S is contextual or not.
def contextualQ(S,verbose=False):
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).
    T=[]
    Z=[] # complement of T
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    if verbose:
                        return [True,None,None]
                    else:
                        return True
    if verbose:
        return [False,Z,T]
    else:
        return False

# Input: ham, a Hamiltonian specified as a dict mapping Pauli strings to coefficients.                                                                                               
# Output: a boolean indicating whether ham is contextual or not.                                                                                                                                                                                                                                                                                    
def contextualQ_ham(ham,verbose=False):
    S = list(ham.keys())
    # Store T all elements of S that anticommute with at least one other element in S (takes O(|S|**2) time).                                                                        
    T=[]
    Z=[] # complement of T                                                                                                                                                           
    for i in range(len(S)):
        if any(not commute(S[i],S[j]) for j in range(len(S))):
            T.append(S[i])
        else:
            Z.append(S[i])
    # Search in T for triples in which exactly one pair anticommutes; if any exist, S is contextual.                                                                                 
    for i in range(len(T)): # WLOG, i indexes the operator that commutes with both others.                                                                                           
        for j in range(len(T)):
            for k in range(j,len(T)): # Ordering of j, k does not matter.                                                                                                            
                if i!=j and i!=k and commute(T[i],T[j]) and commute(T[i],T[k]) and not commute(T[j],T[k]):
                    return True
    if verbose:
        return False,Z,T
    else:
        return False

# Multiply two Pauli operators p,q, represented as strings;
# output has the form [r, sgn], where r is a Pauli operator specified as a string,
# and sgn is the complex number such that p*q == sgn*r.
def pauli_mult(p,q):
    assert(len(p)==len(q))
    sgn=1
    out=''
    for i in range(len(p)):
        if p[i]=='I':
            out+=q[i]
        elif q[i]=='I':
            out+=p[i]
        elif p[i]=='X':
            if q[i]=='X':
                out+='I'
            elif q[i]=='Y':
                out+='Z'
                sgn=sgn*1j
            elif q[i]=='Z':
                out+='Y'
                sgn=sgn*-1j
        elif p[i]=='Y':
            if q[i]=='Y':
                out+='I'
            elif q[i]=='Z':
                out+='X'
                sgn=sgn*1j
            elif q[i]=='X':
                out+='Z'
                sgn=sgn*-1j
        elif p[i]=='Z':
            if q[i]=='Z':
                out+='I'
            elif q[i]=='X':
                out+='Y'
                sgn=sgn*1j
            elif q[i]=='Y':
                out+='X'
                sgn=sgn*-1j
    return [out,sgn]

# Given a commuting set of Pauli strings, input as a dict mapping each to None,
# return a independent generating set in the same format, together with the dict mapping
# each original element to its equivalent product in the new set.
def to_indep_set(G_w_in):
    G_w = G_w_in
    G_w_keys = [[str(g),1] for g in G_w.keys()]
    G_w_keys_orig = [str(g) for g in G_w.keys()]
    generators = []
    for i in range(len(G_w_keys[0][0])):
        # search for first X,Y,Z in ith position
        fx=None
        fy=None
        fz=None
        j=0
        while fx==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='X' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fx=G_w_keys[j]
            j+=1
        j=0
        while fy==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Y' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fy=G_w_keys[j]
            j+=1
        j=0
        while fz==None and j<len(G_w_keys):
            if G_w_keys[j][0][i]=='Z' and not any(G_w_keys[j][0]==g[0] for g in generators):
                fz=G_w_keys[j]
            j+=1
        # multiply to eliminate all other nonidentity entries in ith position
        if fx!=None:
            generators.append(fx)
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='X': # if any other element of G_w has 'X' in the ith position...
                    # multiply it by fx
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        
        if fz!=None:
            generators.append(fz)
            # if any other element of G_w has 'Z' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Z': 
                    # multiply it by fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fz] # update the factor list for G_w_keys[j]
                    sgn=G_w_keys[j][1]*fz[1] # update the sign for G_w_keys[j]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0]) # multiply G_w_keys[j] by fz...
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn # ... and by the associated sign.
        
        if fx!=None and fz!=None:
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': # if any other element of G_w has 'Y' in the ith position...
                    # multiply it by fx and fz
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fx,fz]
                    sgn=G_w_keys[j][1]*fx[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fx[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
                    sgn=G_w_keys[j][1]*fz[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fz[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
        # If both fx and fz are not None, then at this point we are done with this position.
        # Otherwise, there may be remaining 'Y's at this position:
        elif fy!=None:
            generators.append(fy)
            # if any other element of G_w has 'Y' in the ith position...
            for j in range(len(G_w_keys)):
                if G_w_keys[j][0][i]=='Y': 
                    # multiply it by fy
                    G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[fy]
                    sgn=G_w_keys[j][1]*fy[1]
                    G_w_keys[j]=pauli_mult(G_w_keys[j][0],fy[0])
                    G_w_keys[j][1]=G_w_keys[j][1]*sgn
    for j in range(len(G_w_keys)):
        G_w[G_w_keys_orig[j]]=G_w[G_w_keys_orig[j]]+[G_w_keys[j]]
    
    return generators, G_w

# Input: a noncontextual Hamiltonian encoded as a dict of the form e.g. {'III':0.123, 'XII':1.234, 'YII':-5.678,...}
# Output: a quasi-quantized model for the Hamiltonian (see https://arxiv.org/pdf/2002.05693.pdf),
# in the form [G,{Ci1},reconstruction], where G is a list of Pauli strings -- the universally-commuting subset of R,
# {Ci1} is a list of Pauli strings -- the anticommuting subset of R,
# and reconstruction mapping the terms in the Hamiltonian (as Pauli strings) to the set of elements of R that generate them;
# this set has form [J_G, J_C, sgn], where the Hamiltonian term is obtained as the product of J_G (a subset of G), J_C (a subset of {Ci1}), and the sign sgn.
def quasi_model(ham_dict):
    terms = [str(k) for k in ham_dict.keys()]
    check = contextualQ(terms,verbose=True)
    assert(not check[0]) # Hamiltonian should be noncontextual
    Z = check[1] # get set of universally-commuting terms, Z, and its complement, T
    T = check[2]
    
    # Partition T into cliques:
    C=[]
    while T:
        C.append([T.pop()]) # remove the last element from T and put it in a new sublist in C
        for i in range(len(T)-1,-1,-1): # among the remaining elements in T...
            t=T[i]
            if commute(C[-1][0],t): # check if each commutes with the current clique
                C[-1].append(t) # if so, add it to the current clique...
                T.remove(t) # and remove it from T
                
    # Get full set of universally-commuting component operators:
    Gprime = [[z,1] for z in Z] # elements are stored together with their sign
    Ci1s=[]
    for Cii in C: # for each clique...
        Ci=Cii
        Ci1=Ci.pop() # pull out one element
        Ci1s.append(Ci1) # append it to a list of these
        for c in Ci: Gprime.append(pauli_mult(c,Ci1)) # add the remaining elements, multiplied by Ci1, to the commuting set
    
    # Get independent generating set for universally-commuting component operators:
    G_p = dict.fromkeys([g[0] for g in Gprime],[])
    G,G_mappings = to_indep_set(G_p)
    
    # Remove duplicates and identities from G:
    G = list(dict.fromkeys([g[0] for g in G]))
    # Remove identities from product list:
    i=len(G)-1
    while i>=0:
        if all([G[i][j]=='I' for j in range(len(G[i]))]):
            del G[i]
        i=i-1
    
    # Rewrite the values in G_mappings as lists of the form e.g. [sgn, 'XYZ', 'XZY',...]:
    Gprime = list(dict.fromkeys([g[0] for g in Gprime]))
    for g in G_mappings.keys():
        ps = G_mappings[g]
        sgn = int(np.real(np.prod([p[1] for p in ps])))
        ps = [[p[0] for p in ps],sgn]
        # Remove identities from product list:
        i=len(ps[0])-1
        while i>=0:
            if all([ps[0][i][j]=='I' for j in range(len(ps[0][i]))]):
                del ps[0][i]
            i=i-1
        G_mappings[g] = ps
        
    # Assemble all the mappings from terms in the Hamiltonian to their products in R:
    all_mappings = dict.fromkeys(terms)
    for z in Z:
        mapping = G_mappings[z]
        all_mappings[z] = [mapping[0]]+[[]]+[mapping[1]]
        
    for Ci1 in Ci1s:
        all_mappings[Ci1] = [[],[Ci1],1]
    
    for i in range(len(C)):
        Ci=C[i]
        Ci1=Ci1s[i]
        for Cij in Ci:
            mult = pauli_mult(Cij,Ci1)
            mapping = G_mappings[mult[0]]
            all_mappings[Cij] = [mapping[0]]+[[Ci1]]+[mult[1]*mapping[1]]
    
    return G,Ci1s,all_mappings

# Input: a Hamiltonian ham_dict, two lists of epistemic parameters, q and r, and model, the output of a quasi-model.
# Output: the corresponding energy objective function, encoded as a list with the following form:
# [ dim of q, dim of r, list whose elements have the form [coeff, indices of q's, indices of r's, term in Hamiltonian] ]
# This list shows how each term in the Hamiltonian is written as a product of some q's and some r's.
def energy_function_form(ham_dict,model):
    terms = [str(k) for k in ham_dict.keys()]
    q = model[0]
    r = model[1]
    out = []
    for t in terms:
        mappings = model[2][t]
        coeff = ham_dict[t]*mappings[2] # mappings[2] is the sign
        q_indices = [q.index(qi) for qi in mappings[0]]
        r_indices = [r.index(ri) for ri in mappings[1]]
        out.append([coeff, q_indices, r_indices, t])
    return [len(q),len(r),out]

# Given fn_form, the output of an energy_function_form, returns the corresponding function definition,
# whose arguments should be the q's followed by the r's.
def energy_function(fn_form):
    dim_q = fn_form[0]
    return lambda *args: np.real(
        sum(
            [
                (t[0] if len(t[1])==0 and len(t[2])==0 else
                (t[0]*(reduce(lambda x, y: x * y, [args[i] for i in t[1]]))) if len(t[1])>0 and len(t[2])==0 else
                (t[0]*(reduce(lambda x, y: x * y, [args[dim_q+i] for i in t[2]]))) if len(t[1])==0 and len(t[2])>0 else
                (t[0]*(reduce(lambda x, y: x * y, [args[i] for i in t[1]]))*(reduce(lambda x, y: x * y, [args[dim_q+i] for i in t[2]]))))
                for t in fn_form[2]
            ]
        )
    )

# Given a set of angles, return the unit vector specified by those angles in spherical coordinates in arbitrary dimensions.
def angular(args):
    if len(args) == 1:
        return (np.cos(args[0]), np.sin(args[0]))
    else:
        return (np.cos(args[0]), *[np.sin(args[0])*a for a in angular(args[1:])])

# Find noncontextual ground state using numerical minimization for anticommuting part and brute-force search for commuting part (q variables).
def find_gs_noncon(ham_noncon, method = 'differential_evolution', model = None, fn_form = None, energy = None, timer = False):
    if not model:
        model = quasi_model(ham_noncon)
        
    start_time = datetime.now()
    
    if not fn_form:
        fn_form = energy_function_form(ham_noncon,model)
    
    # objective function
    if not energy:
        energy = energy_function(fn_form)
    
    # bounds for angles in search over hypersphere
    bounds = [(0,np.pi) for i in range(fn_form[1]-2)] + [(0,2*np.pi)]

    # list lowest eigenvalues for each assignment to commuting generators
    best_guesses=[]
    
    if fn_form[1] == 0:
        for q in itertools.product([1,-1],repeat=fn_form[0]):
            best_guesses.append([energy(*q),[list(q),[]]])
            
    if fn_form[1] == 2:
        for q in itertools.product([1,-1],repeat=fn_form[0]):
            sol = minimize_scalar(lambda x: energy(*q,np.cos(x),np.sin(x)))
            best_guesses.append([sol['fun'],[list(q),[np.cos(sol['x']),np.sin(sol['x'])]]])
    
    if fn_form[1] > 2:
        if method == 'shgo':
            for q in itertools.product([1,-1],repeat=fn_form[0]):
                sol = sp.optimize.shgo(lambda x: energy(*q,*angular(x)), bounds)
                best_guesses.append([sol['fun'],[list(q),list(angular(sol['x']))]])
            
        if method == 'dual_annealing':
            for q in itertools.product([1,-1],repeat=fn_form[0]):
                sol = sp.optimize.dual_annealing(lambda x: energy(*q,*angular(x)), bounds)
                best_guesses.append([sol['fun'],[list(q),list(angular(sol['x']))]])
    
        if method == 'differential_evolution':
            for q in itertools.product([1,-1],repeat=fn_form[0]):
                sol = sp.optimize.differential_evolution(lambda x: energy(*q,*angular(x)), bounds)
                best_guesses.append([sol['fun'],[list(q),list(angular(sol['x']))]])
    
        if method == 'basinhopping':
            for q in itertools.product([1,-1],repeat=fn_form[0]):
                sol = sp.optimize.basinhopping(lambda x: energy(*q,*angular(x)), bounds)
                best_guesses.append([sol['fun'],[list(q),list(angular(sol['x']))]])
    
        if method == 'shgo_sobol':
            for q in itertools.product([1,-1],repeat=fn_form[0]):
                sol = sp.optimize.shgo(lambda x: energy(*q,*angular(x)), bounds, n=200, iters=5, sampling_method='sobol')
                best_guesses.append([sol['fun'],[list(q),list(angular(sol['x']))]])

    # find overall lowest eigenvalue
    best = min(best_guesses,key=lambda x: x[0])
    
    if timer:
        return best + [model, fn_form], datetime.now()-start_time
    else:
        return best + [model, fn_form]

# Returns the sequence of rotations to diagonalize the generators for a noncontextual ground state.
# The rotations are represented as [angle, generator], where generator is a string representing a Pauli operator.
# Additionally returns the diagonalized generators (GuA), and their values in the noncontextual ground state (ep_state_trans).
def diagonalize_epistemic(model,fn_form,ep_state):
    
    assert(len(ep_state[0]) == fn_form[0])
    assert(len(model[0]) == fn_form[0])
    assert(len(ep_state[1]) == fn_form[1])
    assert(len(model[1]) == fn_form[1])
    
    rotations = []
    
    # if there are cliques...
    if fn_form[1] > 0:
        # rotations to map A to a single Pauli (to be applied on left)
        for i in range(1,fn_form[1]):
            theta = np.arctan2(ep_state[1][i],np.sqrt(sum([ep_state[1][j]**2 for j in range(i)])))
            if i == 1 and ep_state[1][0] < 0:
                theta = np.pi - theta
            generator = pauli_mult(model[1][0],model[1][i])
            sgn = generator[1].imag
            rotations.append( [sgn*theta, generator[0]] )
    
        # rotations to diagonalize G union with the new A
        GuA = deepcopy(model[0] + [model[1][0]])
        ep_state_trans = deepcopy(ep_state[0] + [1])
    
    # if there are no cliques...
    else:
        # rotations to diagonalize G
        GuA = deepcopy(model[0])
        ep_state_trans = deepcopy(ep_state[0])
    
    for i in range(len(GuA)):
        g = GuA[i]
        
        # if g is not already a single Z...
        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):
        
            # if g is diagonal...
            if all(p == 'I' or p == 'Z' for p in g):
                
                # store locations where g has a Z and none of the previously diagonalized generators do
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in GuA[:i]):
                        Zs.append(m)
                        
                # there must be at least one such location: pick the first one
                assert(len(Zs) > 0)
                m = Zs[0]
                
                # construct a rotation about the single Y operator acting on qubit m
                K = ''
                for o in range(len(g)):
                    if o == m:
                        K += 'Y'
                    else:
                        K += 'I'
                
                # add adjoint rotation to rotations list
                rotations.append( ['pi/2', K] )
                
                # apply R to GuA
                for m in range(len(GuA)):
                    if not commute(GuA[m],K):
                        p = deepcopy(pauli_mult(K,GuA[m]))
                        GuA[m] = p[0]
                        ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]
        
            g = GuA[i]
            # g should now not be diagonal
            if not any(p != 'I' and p != 'Z' for p in g):
                print(model,'\n')
                print(fn_form,'\n')
                print(ep_state,'\n')
                print(GuA)
                print(g)
            assert(any(p != 'I' and p != 'Z' for p in g))
        
            # construct a rotation to map g to a single Z
            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    if found:
                        J += 'X'
                    else:
                        J += 'Y'
                        found = True
                elif g[j] == 'Y':
                    if found:
                        J += 'Y'
                    else:
                        J += 'X'
                        found = True
                else:
                    J += g[j]
        
            # add adjoint rotation to rotations list
            rotations.append( ['pi/2', J] )
        
            # apply R to GuA
            for m in range(len(GuA)):
                if not commute(GuA[m],J):
                    p = deepcopy(pauli_mult(J,GuA[m]))
                    GuA[m] = p[0]
                    ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]
    
    return rotations, GuA, np.real(ep_state_trans)

# Given a rotation (in the form [angle, generator]) and a Pauli p, returns a dict representing the linear combination of Paulis that results
# from applying the rotation to p.
def apply_rotation(rotation,p):
    
    out = {}
    
    if not commute(rotation[1],p):
        if rotation[0] == 'pi/2':
            q = pauli_mult(rotation[1],p)
            out[q[0]] = (1j*q[1]).real
    
        else:
            out[p] = np.cos(rotation[0])
            q = pauli_mult(rotation[1],p)
            out[q[0]] = (1j*q[1]*np.sin(rotation[0])).real
            
    else:
            out[p] = 1.
    
    return out

# For a Pauli operator P (specified as a string),
# returns the matrix representation of P as a scipy.sparse.csr_matrix object.
def pauli_to_sparse(P):
    
    x = ''
    for i in range(len(P)):
        if P[i] == 'I' or P[i] == 'Z':
            x = x + '0'
        else:
            x = x + '1'
    x = int(x,2)
    
    z = ''
    for i in range(len(P)):
        if P[i] == 'I' or P[i] == 'X':
            z = z + '0'
        else:
            z = z + '1'
    z = int(z,2)
    
    y = 0
    for i in range(len(P)):
        if P[i] == 'Y':
            y += 1
            
    rows = [r for r in range(2**len(P))]
    
    cols = [r^x for r in range(2**len(P))]
    
    vals = []
    for r in range(2**len(P)):
        sgn = bin(r&z)
        vals.append( ((-1.0)**sum([int(sgn[i]) for i in range(2,len(sgn))])) * ((-1j)**y) )
        
    m = coo_matrix( (vals, (rows, cols)) )
    
    return m.tocsr()

# For a Hamiltonian ham specified as a dict mapping Pauli terms (strings) to their coefficients,
# returns the matrix representation of ham as a scipy.sparse.csr_matrix object.
def hamiltonian_to_sparse(ham):
    return reduce(lambda x, y: x + y,[ham[t]*pauli_to_sparse(t) for t in ham.keys()])

# Given a Hamiltonian ham, for which the noncontextual part has a quasi-quantized model
# specified by model and fn_form, and a noncontextual ground state specified by ep_state,
# returns the noncontextual ground state energy plus the quantum correction.
def quantum_correction(ham,model,fn_form,ep_state):
    
    rotations, diagonal_set, vals = diagonalize_epistemic(model,fn_form,ep_state)
    
#     print(diagonal_set)
#     print(vals)
#     print(rotations)
    
    n_q = len(diagonal_set[0])
    
    ham_rotated = deepcopy(ham)
    
    for r in rotations: # rotate the full Hamiltonian to the basis with diagonal noncontextual generators
        ham_next = {}
        for t in ham_rotated.keys():
            t_set_next = apply_rotation(r,t)
            for t_next in t_set_next.keys():
                if t_next in ham_next.keys():
                    ham_next[t_next] = ham_next[t_next] + t_set_next[t_next]*ham_rotated[t]
                else:
                    ham_next[t_next] = t_set_next[t_next]*ham_rotated[t]
        ham_rotated = deepcopy(ham_next)
        
#     print(ham_rotated)
       
    z_indices = []
    for d in diagonal_set:
        for i in range(n_q):
            if d[i] == 'Z':
                z_indices.append(i)
                
#     print(z_indices)
        
    ham_red = {}
    
    for t in ham_rotated.keys():
        
        sgn = 1
        
        for j in range(len(diagonal_set)): # enforce diagonal generator's assigned values in diagonal basis
            z_index = z_indices[j]
            if t[z_index] == 'Z':
                sgn = sgn*vals[j]
            elif t[z_index] != 'I':
                sgn = 0
        
        if sgn != 0:
            # construct term in reduced Hilbert space
            t_red = ''
            for i in range(n_q):
                if not i in z_indices:
                    t_red = t_red + t[i]
            if t_red in ham_red.keys():
                ham_red[t_red] = ham_red[t_red] + ham_rotated[t]*sgn
            else:
                ham_red[t_red] = ham_rotated[t]*sgn
        
#         print(t,t_red,sgn,ham_rotated[t])
            
#     print('\n\n',ham_red)
    
    if n_q-len(diagonal_set) == 0:
        assert(len(list(ham_red.keys())) == 1)
        assert(list(ham_red.keys())[0] == '')
        return list(ham_red.values())[0].real
    
    else:
        # find lowest eigenvalue of reduced Hamiltonian
        ham_red_sparse = hamiltonian_to_sparse(ham_red)
        if n_q-len(diagonal_set) <= 4:
            return min(np.linalg.eigvalsh(ham_red_sparse.toarray()))
        else:
            return sp.sparse.linalg.eigsh(ham_red_sparse, which='SA', k=1)[0][0]


"""
Contextual subspace VQE:
"""

# Given a quasi-quantized model specified by fn_form and a noncontextual ground state
# specified by ep_state, returns a dict mapping each Pauli operator determined by
# fn_form to its expectation value in the state ep_state.
def exp_vals(fn_form,ep_state):
    out = {}
    for t in fn_form[2]:
        if len(t[1])==0 and len(t[2])==0:
            out[t[3]] = np.real(1)
        elif len(t[1])>0 and len(t[2])==0:
            out[t[3]] = np.real(reduce(lambda x, y: x * y, [ep_state[0][i] for i in t[1]]))
        elif len(t[1])==0 and len(t[2])>0:
            out[t[3]] = np.real(reduce(lambda x, y: x * y, [ep_state[1][i] for i in t[2]]))
        else:
            out[t[3]] = np.real((reduce(lambda x, y: x * y, [ep_state[0][i] for i in t[1]]))*(reduce(lambda x, y: x * y, [ep_state[1][i] for i in t[2]])))
    return out

# Given ham (the full Hamiltonian), model (the quasi-quantized model for the noncontextual part),
# fn_form (the output of energy_function_form(ham_noncon,model)), and order (a list specifying the order in which to remove the qubits),
# returns a list of approximations to the ground state energy of the full Hamiltonian,
# whose ith element is the approximation obtained by simulating i qubits on the quantum computer,
# with the remaining qubits simulated by the noncontextual approximation.
# (Hence the 0th element is the pure noncontextual approximation.)
# If order is shorter than the total number of qubits, only approximations using qubit removals
# in order are simulated.
def contextual_subspace_approximations(ham,model,fn_form,ep_state,order):
    
    rotations, diagonal_set, vals = diagonalize_epistemic(model,fn_form,ep_state)
    
    n_q = len(diagonal_set[0])
    
    order_len = len(order)
    
    vals = list(vals)
    
    # rectify order
    for i in range(len(order)):
        for j in range(i):
            if order[j] < order[i]:
                order[i] -= 1
    
    out = []
    
    for k in range(order_len+1):
    
        ham_rotated = deepcopy(ham)
    
        for r in rotations: # rotate the full Hamiltonian to the basis with diagonal noncontextual generators
            ham_next = {}
            for t in ham_rotated.keys():
                t_set_next = apply_rotation(r,t)
                for t_next in t_set_next.keys():
                    if t_next in ham_next.keys():
                        ham_next[t_next] = ham_next[t_next] + t_set_next[t_next]*ham_rotated[t]
                    else:
                        ham_next[t_next] = t_set_next[t_next]*ham_rotated[t]
            ham_rotated = deepcopy(ham_next)
       
        z_indices = []
        for d in diagonal_set:
            for i in range(n_q):
                if d[i] == 'Z':
                    z_indices.append(i)
        
        ham_red = {}
    
        for t in ham_rotated.keys():
        
            sgn = 1
        
            for j in range(len(diagonal_set)): # enforce diagonal generator's assigned values in diagonal basis
                z_index = z_indices[j]
                if t[z_index] == 'Z':
                    sgn = sgn*vals[j]
                elif t[z_index] != 'I':
                    sgn = 0
        
            if sgn != 0:
                # construct term in reduced Hilbert space
                t_red = ''
                for i in range(n_q):
                    if not i in z_indices:
                        t_red = t_red + t[i]
                if t_red in ham_red.keys():
                    ham_red[t_red] = ham_red[t_red] + ham_rotated[t]*sgn
                else:
                    ham_red[t_red] = ham_rotated[t]*sgn
    
        if n_q-len(diagonal_set) == 0:
            assert(len(list(ham_red.keys())) == 1)
            assert(list(ham_red.keys())[0] == '')
            out.append(list(ham_red.values())[0].real)
    
        else:
            # find lowest eigenvalue of reduced Hamiltonian
            ham_red_sparse = hamiltonian_to_sparse(ham_red)
            if n_q-len(diagonal_set) <= 4:
                out.append(min(np.linalg.eigvalsh(ham_red_sparse.toarray())))
            else:
#                 print(f'  computing restricted ground state for {n_q-len(diagonal_set)} qubits...')
                out.append(sp.sparse.linalg.eigsh(ham_red_sparse, which='SA', k=1)[0][0])
        
        if order:
            # Drop a qubit:
            i = order[0]
            order.remove(i)
            diagonal_set = diagonal_set[:i]+diagonal_set[i+1:]
            vals = vals[:i]+vals[i+1:]

    return out

# Heuristic search for best qubit removal ordering:
# search starting from all qubits in noncontextual part,
# and moving them to qc two at a time,
# greedily choosing the pair that maximally reduces overall error at each step.
def csvqe_approximations_heuristic(ham, ham_noncon, n_qubits, true_gs):

    model = quasi_model(ham_noncon)
    fn_form = energy_function_form(ham_noncon,model)
        
    gs_noncon = find_gs_noncon(ham_noncon,method = 'differential_evolution')
        
    if gs_noncon[0]-true_gs < 10**-10:
        return [true_gs,[gs_noncon[0] for i in range(n_qubits)],[gs_noncon[0]-true_gs for i in range(n_qubits)],[i for i in range(n_qubits)]]
            
    else:
            
        ep_state = gs_noncon[1]
        
        indices = [j for j in range(n_qubits)]
        order = []
        
        exact = False
        
        while len(indices)>1 and not exact:
            # print('indices',indices,'\n')
            # print('order',order,'\n')
            num_to_remove = 2
            i_subset_improvements = {i_subset:0 for i_subset in itertools.combinations(indices,num_to_remove)}
            for i_subset in itertools.combinations(indices,num_to_remove):
                possible_order = order+list(i_subset)
                current_i = len(possible_order)
                approxs_temp = contextual_subspace_approximations(ham,model,fn_form,ep_state,order=possible_order)
                errors_temp = [a - true_gs for a in approxs_temp]
                if errors_temp[current_i] < 10**-6:
                    exact = True
                improvement = errors_temp[current_i-2]-errors_temp[current_i]
                # print([round(e,3) for e in errors_temp],improvement)
                i_subset_improvements[i_subset] += improvement
                    
            best_i_subset = max(i_subset_improvements, key=i_subset_improvements.get)
            # print('\nbest_i_subset',best_i_subset,'\n')
            order = order+list(best_i_subset)
            # print(f'current order: {order}\n')
            indices = []
            for i in range(n_qubits):
                if not i in order:
                    indices.append(i)
                        
        # add last index if necessary
        order_full = deepcopy(order)
        for i in range(n_qubits):
            if not i in order_full:
                order_full.append(i)
                    
        order2 = []
        for i in range(int(len(order)/2)):
            order2.append(order[2*i+1])
            order2.append(order[2*i])
                
        # print(order)
        # print(order2,'\n')
            
        # print('  getting final approximations...\n')

        approxs = contextual_subspace_approximations(ham,model,fn_form,ep_state,order=deepcopy(order_full))
        errors = [a - true_gs for a in approxs]
            
        # print(errors,'\n')
            
        approxs2 = contextual_subspace_approximations(ham,model,fn_form,ep_state,order=deepcopy(order2))
        errors2 = [a - true_gs for a in approxs2]
            
        # print(errors2,'\n')
            
        order_out = []
        approxs_out = [approxs[0]]
        errors_out = [errors[0]]
            
        for i in range(int(len(order)/2)):
            if errors[2*i+1] <= errors2[2*i+1]:
                order_out.append(order[2*i])
                order_out.append(order[2*i+1])
                approxs_out.append(approxs[2*i+1])
                approxs_out.append(approxs[2*i+2])
                errors_out.append(errors[2*i+1])
                errors_out.append(errors[2*i+2])
            else:
                order_out.append(order2[2*i])
                order_out.append(order2[2*i+1])
                approxs_out.append(approxs2[2*i+1])
                approxs_out.append(approxs2[2*i+2])
                errors_out.append(errors2[2*i+1])
                errors_out.append(errors2[2*i+2])
                    
        for i in range(len(order),len(order_full)):
            order_out.append(order_full[i])
            approxs_out.append(approxs[i+1])
            errors_out.append(errors[i+1])

        # print('FINAL ORDER',order_out,'\n')
        # print('FINAL ERRORS',errors_out,'\n')
        
        return [true_gs, approxs_out, errors_out, order_out]

'''
Counting terms in reduced Hamiltonians to be simulated on qc when using CS-VQE.
'''

# Returns the sequence of multipliers corresponding to the rotations that
# diagonalize the generators G of Z, the universally commuting part of the noncontextual Hamiltonian.
# Also returns the Hamiltonians after the rotations have been applied.
def diagonalize_G(model,fn_form,ep_state,ham,ham_noncon):
    
    assert(len(ep_state[0]) == fn_form[0])
    assert(len(model[0]) == fn_form[0])
    assert(len(ep_state[1]) == fn_form[1])
    assert(len(model[1]) == fn_form[1])
    
    multipliers = []
    
    # rotations to diagonalize G
    G = deepcopy(model[0])
    ep_state_trans = deepcopy(ep_state[0])
    ham_terms = deepcopy(list(ham.keys()))
    ham_terms = {t:[t,1] for t in ham_terms}
    
    for i in range(len(G)):
        g = G[i]
        
        # if g is not already a single Z...
        if not any((all(g[k] == 'I' or k == j for k in range(len(g))) and g[j] == 'Z') for j in range(len(g))):
        
            # if g is diagonal...
            if all(p == 'I' or p == 'Z' for p in g):
                
                # store locations where g has a Z and none of the previously diagonalized generators do
                Zs = []
                for m in range(len(g)):
                    if g[m] == 'Z' and all(h[m] == 'I' for h in G[:i]):
                        Zs.append(m)
                        
                # there must be at least one such location: pick the first one
                assert(len(Zs) > 0)
                m = Zs[0]
                
                # construct a rotation about the single Y operator acting on qubit m
                K = ''
                for o in range(len(g)):
                    if o == m:
                        K += 'Y'
                    else:
                        K += 'I'
                
                # add to multipliers list
                multipliers.append(K)
                
                # apply to G
                for m in range(len(G)):
                    if not commute(G[m],K):
                        p = deepcopy(pauli_mult(K,G[m]))
                        G[m] = p[0]
                        ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]
                        
                # apply to ham
                for t in ham_terms.keys():
                    if not commute(ham_terms[t][0],K):
                        p = deepcopy(pauli_mult(K,ham_terms[t][0]))
                        ham_terms[t][0] = p[0]
                        ham_terms[t][1] = 1j*p[1]*ham_terms[t][1]
        
            g = G[i]
            # g should now not be diagonal
            if not any(p != 'I' and p != 'Z' for p in g):
                print(model,'\n')
                print(fn_form,'\n')
                print(ep_state,'\n')
                print(G)
                print(g)
            assert(any(p != 'I' and p != 'Z' for p in g))
        
            # construct a rotation to map g to a single Z
            J = ''
            found = False
            for j in range(len(g)):
                if g[j] == 'X':
                    if found:
                        J += 'X'
                    else:
                        J += 'Y'
                        found = True
                elif g[j] == 'Y':
                    if found:
                        J += 'Y'
                    else:
                        J += 'X'
                        found = True
                else:
                    J += g[j]
        
            # add to multipliers list
            multipliers.append(J)
        
            # apply to G
            for m in range(len(G)):
                if not commute(G[m],J):
                    p = deepcopy(pauli_mult(J,G[m]))
                    G[m] = p[0]
                    ep_state_trans[m] = 1j*p[1]*ep_state_trans[m]
                    
            # apply to ham
            for t in ham_terms.keys():
                if not commute(ham_terms[t][0],J):
                    p = deepcopy(pauli_mult(J,ham_terms[t][0]))
                    ham_terms[t][0] = p[0]
                    ham_terms[t][1] = 1j*p[1]*ham_terms[t][1]
    
    # Get outcome Hamiltonians
    ham_out = {}
    for t in ham.keys():
        ham_out[ham_terms[t][0]] = ham_terms[t][1]*ham[t]
        
    for t in ham_out.keys():
        assert(abs(np.real(ham_out[t])-ham_out[t])<10**(-10))
        ham_out[t]=np.real(ham_out[t])
        
    ham_noncon_out = {}
    for t in ham_noncon.keys():
        ham_noncon_out[ham_terms[t][0]] = ham_terms[t][1]*ham[t]
        
    for t in ham_noncon_out.keys():
        assert(abs(np.real(ham_noncon_out[t])-ham_noncon_out[t])<10**(-10))
        ham_noncon_out[t]=np.real(ham_noncon_out[t])
    
    return multipliers, G, np.real(ep_state_trans), ham_out, ham_noncon_out

# Get numbers of terms as a function of number of qubits used on quantum computer:
# ham and ham_noncon should be the full and noncontextual Hamiltonians, respectively,
# order should be the CS-VQE order to move the qubits from noncontextual part to
# quantum part, and ep_state should be the noncontextual ground state in the form used above.
def num_of_terms(ham,ham_noncon,order,ep_state):
    n_qubits = len(order)

    model = quasi_model(ham_noncon)
    fn_form = energy_function_form(ham_noncon,model)

    multipliers, diagonal_set, vals, ham_out, ham_noncon_out = diagonalize_G(model,fn_form,ep_state,ham,ham_noncon)
    
    terms_on_qc_by_qubits_on_qc = {}
    
    # i will run over the possible numbers of qubits to be simulated on the q.c.
    for i in range(n_qubits+1):
        # The set of qubits in the noncontextual approximation:
        set_in_noncon = order[i:]
        # The set of terms to be simulated on the quantum computer:
        terms_on_qc = []
        for t in ham_out.keys():
            if all(t[j]=='I' or t[j]=='Z' for j in set_in_noncon):
                term = deepcopy(t)
                for j in range(n_qubits-1,-1,-1):
                    if j in set_in_noncon:
                        term = term[:j]+term[j+1:]
                terms_on_qc.append(term)
        terms_on_qc = set(terms_on_qc)
        if ''.join(['I' for j in range(i)]) in terms_on_qc:
            terms_on_qc.remove(''.join(['I' for j in range(i)]))
        terms_on_qc_by_qubits_on_qc[i] = len(list(terms_on_qc))
        
    return terms_on_qc_by_qubits_on_qc


'''
Heuristic for obtaining noncontextual sub-Hamiltonians
'''

# Input:
#   ham, a Hamiltonian specified as a dict mapping Pauli strings to their coefficients;
#   cutoff, the number of seconds to search for;
#   criterion, either 'weight' or 'size'.
# Output: an approximation to the optimal noncontextual sub-Hamiltonian, calculated by DFS from highest to lowest term weight (coefficient magnitude).
def greedy_dfs(ham,cutoff,criterion='weight'):
    
    weight = {k:abs(ham[k]) for k in ham.keys()}
    possibilities = [k for k, v in sorted(weight.items(), key=lambda item: -item[1])] # sort in decreasing order of weight
    
    best_guesses = [[]]
    stack = [[[],0]]
    start_time = datetime.now()
    delta = timedelta(seconds=cutoff)
    
    i = 0
    
    while datetime.now()-start_time < delta and stack:
        
        while i < len(possibilities):
#             print(i)
            next_set = stack[-1][0]+[possibilities[i]]
#             print(next_set)
#             iscontextual = contextualQ(next_set)
#             print('  ',iscontextual,'\n')
            if not contextualQ(next_set):
                stack.append([next_set,i+1])
            i += 1
        
        if criterion == 'weight':
            new_weight = sum([abs(ham[p]) for p in stack[-1][0]])
            old_weight = sum([abs(ham[p]) for p in best_guesses[-1]])
            if new_weight > old_weight:
                best_guesses.append(stack[-1][0])
                # print(len(stack[-1][0]))
                # print(stack[-1][0],'\n')
            
        if criterion == 'size' and len(stack[-1][0]) > len(best_guesses[-1]):
            best_guesses.append(stack[-1][0])
            # print(len(stack[-1][0]))
            # print(stack[-1][0],'\n')
            
        top = stack.pop()
        i = top[1]
    
    return best_guesses
