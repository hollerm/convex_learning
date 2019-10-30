from matpy import *


########Helper functions and operators

#1D test data
def get_1d_lr():

    ksz = 21

    #Make patch
    blob = np.zeros(ksz)
    
    blob[2:7] = 2.0
    blob[8:12] = 1.0
    blob[13:15] = 0.2
    blob[15:17] = 1.9
    blob[18:20] = 0.8
    
    #blob[0:2] = 0.5
    #blob[19:] = 0.5
    
    blob[0:2] = 2.0
    #blob[19:] = 2.0
    
    
    blob = np.flip(blob,axis=0)


    #Get data
    K = patch_1dsyn([7*ksz,ksz])

    A =  np.zeros(K.indim)
    
    p0 = ksz-1
    
    A[p0,:] = blob   
    A[1*ksz+p0,:] = blob
    A[3*ksz-8+p0,:] = blob
    A[4*ksz-4+p0,:] = blob
    A[5*ksz-3+p0,:] = blob

    u0 = K.fwd(A)

    return u0,A,K
    

#Smoothing operator for nuclear norm evaluation
class patch_1d_average(object):


    def __init__(self,indim,avglen=False):
    
        if not avglen:
            avglen = indim[1]
       
         
        self.k = np.zeros([avglen,1])
        
        if 0:
            self.k[:,0] = scipy.signal.gaussian(avglen,avglen/4.0)
        else:
            x = np.abs(np.linspace(-1.0,1.0,avglen))**10.0
            cc = (avglen+1)/2
            self.k[:cc,0] = x[cc-1:]
            self.k[cc-1:,0] = x[:cc]
        
        
        self.k /= np.abs(self.k).sum()
                    
        
        self.kflip = np.flip(self.k,0)
        self.indim = indim
        self.outdim = indim
        self.avglen = avglen

            
    def fwd(self,c):
    
        return scipy.signal.convolve(c,self.k,mode='same')
        


    def adj(self,c):
    
        
        return scipy.signal.convolve(c,self.kflip,mode='same')


    def test_adj(self):
       
        test_adj(self.fwd,self.adj,self.indim,self.outdim)


#Patch-matrix to signal synthesis operator
class patch_1dsyn(object):


    def __init__(self,indim):
                    
        self.ksz = indim[-1]
        self.indim = indim
        self.outdim = (indim[0]-self.ksz,)
            
    def fwd(self,c):
    
            
        g = np.zeros(self.outdim)
    
        for z1 in range(self.ksz):
            g += c[z1:z1-self.ksz,z1]
            
        return g


    def adj(self,g):
    
        
        c = np.zeros(self.indim)

        for z1 in range(self.ksz):
            c[z1:z1-self.ksz,z1] = g

        return c


    def test_adj(self):
       
        test_adj(self.fwd,self.adj,self.indim,self.outdim)
        

########Algorithms

#Solves the patch synthesis problem in 1D
def mtx_compl_1d(**par_in):


    u0 = 0
    
    A = 0
    K = 0
    
    datainit = False
       
    niter = 1

    ksz = 9 #Size of filter kernels
    
    ld = 200.0 #Data missfit
    nu = 500.0 #low rank vs sparse penalty, in [0,1000]

    noise=0.0
    
    s_t_ratio = False #50.0

    check = 10 #Show results ever "check" iteration
    
    nrm_red = 1.0 #Factor for norm reduction, no convergence guaratee with <1


    sp_type = 'l1infty' #{'l1','l1infty'}    
   
    nuc_smoothing = True
    
    #Set parameter according to par_in ----------
    par = par_parse(locals().copy(),par_in)
    #Update parameter
    for key,val in par.iteritems():
        exec(key + '= val')
    res = output(par) #Initialize output class
    #--------------------------------------------


    #Automatic adaption of s_t_ratio to lambda
    if not np.any(s_t_ratio):
        s_t_ratio = 100*ld/50.0
        print('Adapted s_t_ratio to ' + str(s_t_ratio))


    if np.any(A):
        K = patch_1dsyn(A.shape)
        u0 = K.fwd(np.copy(A))    
  
        ksz = A.shape[-1]
        res.A = A


    res.orig = np.copy(u0)
    
    

    #Add noise
    if noise:
        np.random.seed(1)
        u0 = np.copy(u0)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=noise*rg,size=u0.shape) #Add noise

    #Image size
    N = u0.shape[0]

    #Parameter refactorization
    if nu<0 or nu> 1000:
        raise ValueError('Error: nu must be in [0,1000]')
    nu /=1000.0
    
    #Set parameter
    pnuc = nu
    pl1 = (1-nu)
    
    
    print('Parameter for ld/nuc/l1: ' + str(ld) + '/' + str(pnuc) + '/' + str(pl1))

    #Stepsize
    #The norm of each filter projection was estimated with the square root of the filter size
    #The norm of the convolution was estmated with 1
    nrm = nrm_red*get_product_norm( [ [ksz],[1.0] ])
    
    sig = s_t_ratio/nrm
    tau = 1/(nrm*s_t_ratio)

    print('Stepsizes: sig: '+str(sig)+' / tau: '+str(tau))


    #Operators
    K = patch_1dsyn([N+ksz,ksz])

    if nuc_smoothing:
        S = patch_1d_average(K.indim)
    else:
        S = id(K.indim)
    
    
    #Primal
    cdim = K.indim
    cvdims = (1,)

    c = np.zeros(cdim)        
    cx = np.zeros(c.shape)

    if datainit:
        c = np.copy(A)
        cx = np.copy(A)


    #Dual
    v = np.zeros(K.outdim)
    q = np.zeros(S.outdim)

    #Objective
    ob_val = np.zeros([niter+1])
    
    dnrm = nfun('l2sq',mshift=u0,npar=ld)
    #pnrm = nfun('l1infty',npar=pl1,vdims=cvdims)
    pnrm = nfun(sp_type,npar=pl1,vdims=cvdims)
    nucnrm = nfun('l1-svd',npar=pnuc,vdims=(1,))

    ob_val[0] = dnrm.val(K.fwd(c)) + nucnrm.val(S.fwd(c)) + pnrm.val(c)
        

    for k in range(niter):

        #Dual
        v = dnrm.dprox( v + sig*K.fwd(cx) ,ppar=sig)
        q = nucnrm.dprox( q + sig*S.fwd(cx) ,ppar=sig)
        
        #Primal
        cx = c - tau*( K.adj(v) + S.adj(q) )
        cx = pnrm.prox(cx,ppar=tau)

        c = 2.0*cx - c

        [c,cx] = [cx,c]


        ob_val[k+1] = dnrm.val(K.fwd(c)) + nucnrm.val(S.fwd(c)) + pnrm.val(c)

        if np.remainder(k,check) == 0:
            print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k]))

    #Printing
    if np.any(A):
        print('Ob-val data: ' + str( dnrm.val(K.fwd(A)) + nucnrm.val(S.fwd(A)) + pnrm.val(A) ))
    
    print('Rank c: ' + str( rank(c,tol=1e-05) ))
    print('Rank S(c): ' + str( rank(S.fwd(c),tol=1e-05) ))
    if np.any(A):
        print('Rank data: ' + str(  rank(S.fwd(A),tol=1e-05)))
        
    print('Data missfit: ' + str( (1.0/ld)*dnrm.val(K.fwd(c))))
    if np.any(A):
        print('Data missfit data: ' + str( (1.0/ld)*dnrm.val(K.fwd(A))))
    print('Nuc norm: ' + str( nucnrm.val(S.fwd(c)) ))
    if np.any(A):
        print('Nuc norm data: ' + str( nucnrm.val(S.fwd(A)) ))
    print('p norm: ' + str( pnrm.val(c) ))
    if np.any(A):
        print('p norm data: ' + str( pnrm.val(A) ))
    
    
    res.c = c
    res.u = K.fwd(c)

    res.ob_val = ob_val
    res.u0 = u0
    res.A = A
    
    
    res.K = K
    res.S = S   

    res.dnrm = dnrm
    res.pnrm = pnrm
    res.nucnrm = nucnrm
    
    fig = plot(res.u,title='Data vs. recon')
    plot(res.orig,fig=fig)
    
    
    imshow(res.c,title='Matrix Recon')
    
    if np.any(A):
        imshow(res.A,title='Orig matrix')

    return res


