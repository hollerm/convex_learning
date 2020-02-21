from matpy import *



###########################################################
# Helper functions ########################################

#Generate a cartoon-texture mix image
#N should be a multiple of ksz and 2
#ksz should be a multiple of stride
#Positive flag generates positive textures instead of mean zero
def get_patchtest(N = 120,stride=1,ksz=15,positive=False):



    u0 = np.zeros([N,N])
    

    x,y = np.meshgrid( np.linspace(0,1.0,N),np.linspace(0,1.0,N) )
    
    
    #Linear background
    u0[:,:] = 2.0*x[:,:]+1.2*y[:,:]
    
    z = 4.0 -1*x - 2*y
    
    #Circle
    for ii in range(N):
        for jj in range(N):
            if (N/4.0 - ii)**2 + (3.0*N/4.0 - jj)**2 < 2*N:
                u0[ii,jj] = z[ii,jj]
    
    u0 /= 5.0


    #First texture patch
    f1=1.0

    n1 = 3.0
    n2 = 1.0
    
    l = np.linspace(0,1.0,ksz,endpoint=False)
    x,y = np.meshgrid(l,l)

    b1 = np.sin( 2*np.pi*f1*(n1*x + n2*y))
    
    if positive:
        b1 -= np.min(b1)
    else:
        b1 -= np.sum(b1)/(ksz*ksz)
    #######################
    
    
    #Repeating texture patch
    l = np.linspace(-0.5,0.5,ksz)
    x,y = np.meshgrid(l,l)

    blob = np.abs(np.sqrt(x*x + y*y) - 0.25)
    
    blob[blob>0.1] = 0
    #blob[blob<-0.05] = 0

    
    blob[:,int((ksz-1)/2)] = 0.1
    blob[int((ksz-1)/2),:] = 0.1
    
    blob = blob*20.0
    
    if not positive:
        blob -= np.sum(blob)/(ksz*ksz)

    #######################
    checker = np.zeros([ksz,ksz])
    box = np.zeros([5,5])
    box[1:4,1:4] = 2.0
    
    for ii in range(3):
        for jj in range(3):
            checker[2 + 5*ii-1:2 + 5*ii+2,2+5*jj-1:2+5*jj+2] = 2.0


    
    if not positive:
        checker -= np.sum(checker)/(ksz*ksz)

    
    #Data generation
    K = lconv([120,120,ksz],stride=stride)
    
    A = np.zeros(K.indim)
    
    
    for ii in range(int(N/(2*ksz))):
        for jj in range(int(N/(2*ksz))):
        
            tt = int(ksz/stride)
            A[K.bln+ii*tt,K.bln+jj*tt,:,:] = b1
            A[K.bln+ii*tt + int(N/(2*stride)),K.bln+jj*tt + int(N/(2*stride)),:,:] = blob
            A[K.bln+ii*tt + int(N/(2*stride)),K.bln+jj*tt,:,:] = checker
    
    A /=5.0
    
    u0 += K.fwd(A)    




    return u0,A,K

#Generation of texture test image
def get_texttest(N = 120,stride=3,ksz=15): #N should be a multiple of ksz and 2



    u0 = np.zeros([N,N])
    

    x,y = np.meshgrid( np.linspace(0,1.0,N),np.linspace(0,1.0,N) )

    #First texture patch
    f1=1.0

    n1 = 3.0
    n2 = 1.0
    
    l = np.linspace(0,1.0,ksz,endpoint=False)
    x,y = np.meshgrid(l,l)

    b1 = np.sin( 2*np.pi*f1*(n1*x + n2*y))
    
    b1 -= np.sum(b1)/(ksz*ksz)
    
    #Second texture patch
    f1=2.0

    n1 = -1.5
    n2 = 1.0
    
    l = np.linspace(0,1.0,ksz,endpoint=False)
    x,y = np.meshgrid(l,l)

    b2 = np.sin( 2*np.pi*f1*(n1*x + n2*y))
    
    b2 -= np.sum(b2)/(ksz*ksz)

    #Third texture patch
    f1=2.0
    f2 = 1.0

    l = np.linspace(0,1.0,ksz,endpoint=False)
    x,y = np.meshgrid(l,l)

    b3 = np.sin(2.0*np.pi*f1*x) * np.sin(2.0*np.pi*f2*y) 
    b3 -= np.sum(b3)/(ksz*ksz)
    

    #######################
    checker = np.zeros([ksz,ksz])
    box = np.zeros([5,5])
    box[1:4,1:4] = 2.0
    
    for ii in range(3):
        for jj in range(3):
            checker[2 + 5*ii-1:2 + 5*ii+2,2+5*jj-1:2+5*jj+2] = 2.0

    checker -= np.sum(checker)/(ksz*ksz)
    
    #Data generation
    K = lconv([120,120,ksz],stride=stride)
    
    A = np.zeros(K.indim)
    
    
    
    for ii in range( int(N/(2*ksz))):
        for jj in range( int(N/(2*ksz))):
        
            kos = int(ksz/stride)
            
            A[K.bln+ii*kos,K.bln+jj*kos,:,:] = b1
            A[K.bln+ii*kos + int(N/(2*stride)),K.bln+jj*kos + int(N/(2*stride)),:,:] = b2
            A[K.bln+ii*kos,K.bln+jj*kos + int(N/(2*stride)),:,:] = b3
            A[K.bln+ii*kos + int(N/(2*stride)),K.bln+jj*kos ,:,:] = checker
    
    A /=5.0
    
    u0 += K.fwd(A)    




    return u0,A,K

#Get mask for inpainting
def get_mask(shape,mtype='rand',perc=10):

    if mtype=='rand':
        
        np.random.seed(1)
        
        rmask = np.random.rand(*shape)
        idx = rmask < perc/100.0
        
        mask = np.zeros(shape,dtype='bool')
        mask[idx] = True

    return mask    
    
    
  

#Function to evaluate results for given parameter range(s)
#Rescale: Compute optimal global rescaling befor evaluating signal
def eval_results(basename,pars={},folder='.',show_all=0,decomp=False,rescaled=False):

    #Get sortet list of filenames, parnames and values
    flist,parnames,parvals = get_file_par_list(basename,pars=pars,folder=folder)
        
        
    #Determine parmeter values
    p1 = {}
    p2 = {}
    for pos,parval in enumerate(parvals):
        if parval[0] not in p1:
            p1[parval[0]] = []
        if parval[1] not in p2:
            p2[parval[1]] = []
            
        p1[parval[0]].append(pos)
        p2[parval[1]].append(pos)
        

        
    #p1list = p1.keys()
    #p2list = p2.keys()
    p1list = list(p1)
    p2list = list(p2)
    p1list.sort()
    p2list.sort()
    #p1list = sorted(p1list)
    #p2list = sorted(p2list)
    
    init = 0
    immse = np.zeros( ( len(p1list),len(p2list) ) )
    imranks = np.zeros( ( len(p1list),len(p2list) ) )
    
    for k1 in p1.keys():
        for pos in p1[k1]:
            for k2 in p2.keys():
                if pos in p2[k2]:
                    
                    fullname = folder + '/' + flist[pos]
                    fullname = fullname.replace('//','/')    
                    res = pload(fullname)
                    
                    if hasattr(res, 'ob_val'):
                        if res.ob_val[0]<0.5*res.ob_val[-1]:
                            print('Warning: bad concergence: obval[0]: '+str(res.ob_val[0]) + ' ob_val[-1]' + str(res.ob_val[-1]))
                            print(parnames)
                            print(parvals[pos])
                        
                    if not init:
                        N,M = res.orig.shape
                        imcol = np.zeros((N,M,len(p1list),len(p2list)))
                        imdif = np.zeros((N,M,len(p1list),len(p2list)))                      
                        
                        if decomp:#Show decomposition
                            imcart = np.zeros((N,M,len(p1list),len(p2list)))
                            imtext = np.zeros((N,M,len(p1list),len(p2list)))                   
                            impatch = np.zeros((3*res.K.ksz,3*res.K.ksz,len(p1list),len(p2list)))
                            
                        init = 1
                        
                    imcol[:,:,p1list.index(k1),p2list.index(k2)] = res.u
                    imdif[:,:,p1list.index(k1),p2list.index(k2)] = np.abs(res.u - res.orig)
                    immse[p1list.index(k1),p2list.index(k2)] = mse(res.u,res.orig,rescaled=rescaled)
                    
                    if decomp:
                        imcart[:,:,p1list.index(k1),p2list.index(k2)] = res.u - res.K.fwd(res.c)
                        imtext[:,:,p1list.index(k1),p2list.index(k2)] = res.K.fwd(res.c)
                        impatch[:,:,p1list.index(k1),p2list.index(k2)] = imshowstack(res.patch[...,:9])
                    #immse[p1list.index(k1),p2list.index(k2)] = part_mse(res.u,res.orig)
                    if hasattr(res,'ranks'):
                        imranks[p1list.index(k1),p2list.index(k2)] = sum(res.ranks)
                    if hasattr(res,'rank'):
                        imranks[p1list.index(k1),p2list.index(k2)] = res.rank


    #Reorder image
    imlist = []
    for pos in range(len(p1list)):
        imlist.append( np.concatenate( [imcol[:,:,pos,i] for i in range(len(p2list)) ],axis=1 ) )
    
    imcol = np.concatenate(imlist,axis=0)
    
    #Reorder difference image
    imlist = []
    for pos in range(len(p1list)):
        imlist.append( np.concatenate( [imdif[:,:,pos,i] for i in range(len(p2list)) ],axis=1 ) )
    
    imdif = np.concatenate(imlist,axis=0)

    #Reorder cart/text images
    if decomp:
        imlist = []
        for pos in range(len(p1list)):
            imlist.append( np.concatenate( [imcart[:,:,pos,i] for i in range(len(p2list)) ],axis=1 ) )
        
        imcart = np.concatenate(imlist,axis=0)
        
        imlist = []
        for pos in range(len(p1list)):
            imlist.append( np.concatenate( [imtext[:,:,pos,i] for i in range(len(p2list)) ],axis=1 ) )
        
        imtext = np.concatenate(imlist,axis=0)
        
        imlist = []
        for pos in range(len(p1list)):
            imlist.append( np.concatenate( [impatch[:,:,pos,i] for i in range(len(p2list)) ],axis=1 ) )
        
        impatch = np.concatenate(imlist,axis=0)



    #Show image and difference
    imshow(imcol,title='Imresult\n V: ' +parnames[0] + ': ' + str(p1list) + '\n H: '+parnames[1] + ': '  + str(p2list))
    imshow(imdif,title='Dif to ground truth\n V: ' +parnames[0] + ': ' + str(p1list) + '\n H: '+parnames[1] + ': '  + str(p2list))
    
    if decomp:
        imshow(imcart,title='V: ' +parnames[0] + ': ' + str(p1list) + '\n H: '+parnames[1] + ': '  + str(p2list) )
        imshow(imtext,title='V: ' +parnames[0] + ': ' + str(p1list) + '\n H: '+parnames[1] + ': '  + str(p2list) )
        imshow(impatch,title='V: ' +parnames[0] + ': ' + str(p1list) + '\n H: '+parnames[1] + ': '  + str(p2list) )
    
    #Show scaled mse
    immse[immse==0] = 2.0*immse.max()
    imshow(np.log(immse),cmap='hot',title='log(mse)')
    
    
    #Show ranks
    imshow(imranks,cmap='hot',title='Sum of ranks')

    
    return imcol,imdif,immse


#Compute coefficient image from lifted representation. nfs gives number of filters to use
def decomp_lifted(A,K,nfs=16,show=False):

    N,M = K.outdim
    ksz = K.ksz

    try:
        u,s,v = linalg.svd(A,compute_uv=1,full_matrices=0)
    except:
        print('decomp_lifted: Error with svd!')
        return 0.0,0.0,0.0
        
    coeff = np.zeros([N,M,s.shape[0]])
    patch = np.zeros([ksz,ksz,s.shape[0]])
    
    
        
    for i in range(s.shape[0]):
        #Coefficients
        im = np.reshape(u[:,i]*s[i],(K.indim[0],K.indim[1]))
        im = im[K.bln:,K.bln:]
        coeff[::K.stride,::K.stride,i] = im
        #Patches
        im = np.reshape(v[i,:],(ksz,ksz))
        patch[...,i] = im
        
    if show:
        imshow(np.abs(coeff[...,:nfs]),title='Coeffcients')
        imshow(patch[...,:nfs],title='Patches')
        plot(s,title='Singular values')

    return coeff,patch,s
    
    


###########################################################
# Operators ###############################################

#Patch-synthesis via lifted convolution with stride
#Note: ksz must be a multiple of stride
class lconv(object):


    def __init__(self,dims,stride = 1):
                    
        #Set dimensions
        N = dims[0]
        M = dims[1]
        ksz = dims[2]
        
        #Dimensions of lifted variable
        self.nx = int(np.ceil(float(N)/float(stride))) + int(np.ceil(float(ksz)/float(stride))) - 1
        self.ny = int(np.ceil(float(M)/float(stride))) + int(np.ceil(float(ksz)/float(stride))) - 1

        #Number of boundary points
        self.bln = int(np.ceil(float(ksz)/float(stride))) - 1

        #Input and output dimensions
        self.indim = [self.nx,self.ny,ksz,ksz]
        self.outdim = [N,M]
        
        #Kernel size and stride
        self.ksz = ksz
        self.stride=stride
        
        #The norm  was estimated with ceil(ksz/stride)
        self.nrm = np.ceil(float(self.ksz)/float(self.stride))
        
    def fwd(self,c):
    
        #Artifical extension to avoid case distinction at boundary
        u = np.zeros([ self.outdim[ii]+np.remainder(-self.outdim[ii],self.stride) for ii in range(2)])
        
        
        for dx in range(self.stride):
            for dy in range(self.stride):
                for z1 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                    for z2 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                      
                        u[dx::self.stride,dy::self.stride]+=c[self.bln-z1:self.nx-z1,self.bln-z2:self.ny-z2,self.stride*z1+dx,self.stride*z2+dy]
                
                
        return u[:self.outdim[0],:self.outdim[1]]


    def adj(self,u):


        c = np.zeros(self.indim)
    
        #Artifical extension to avoid case disctinction at boundary
        ux = np.zeros([ self.outdim[ii]+np.remainder(-self.outdim[ii],self.stride) for ii in range(2)]) 
        ux[:self.outdim[0],:self.outdim[1]] = u
        
        for dx in range(self.stride):
            for dy in range(self.stride):
                for z1 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                    for z2 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                      
                        c[self.bln-z1:self.nx-z1,self.bln-z2:self.ny-z2,self.stride*z1+dx,self.stride*z2+dy] = ux[dx::self.stride,dy::self.stride]
                
                
        return c
        
 

    def test_adj(self):
    
    
        test_adj(self.fwd,self.adj,self.indim,self.outdim)



#Image synthesis via convlution of coefficient and dictionary. Operator provides fwd, adj and adj_ker
class cp_conv(object):


    def __init__(self,dims,stride = 1):
                    
        #Set dimensions
        N = dims[0]
        M = dims[1]
        ksz = dims[2]
        nf = dims[3]
        
        #Dimensions of lifted variable
        self.nx = int(np.ceil(float(N)/float(stride))) + int(np.ceil(float(ksz)/float(stride))) - 1
        self.ny = int(np.ceil(float(M)/float(stride))) + int(np.ceil(float(ksz)/float(stride))) - 1

        #Number of boundary points
        self.bln = int(np.ceil(float(ksz)/float(stride))) - 1
        
        #Input and output dimensions
        self.indim = [self.nx,self.ny,nf]
        self.indim2 = [ksz,ksz,nf]
        self.outdim = [N,M]
        
        #Kernel size, stride and number of filters
        self.ksz = ksz
        self.stride=stride
        self.nf = nf


    def fwd(self,c,k):
    
    
        #Artifical extension to avoid case distinction at boundary
        u = np.zeros([ self.outdim[ii]+np.remainder(-self.outdim[ii],self.stride) for ii in range(2)])
        
        for ff in range(self.nf):
            for dx in range(self.stride):
                for dy in range(self.stride):
                    for z1 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                        for z2 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                      
                            u[dx::self.stride,dy::self.stride]+=c[self.bln-z1:self.nx-z1,self.bln-z2:self.ny-z2,ff]*k[self.stride*z1+dx,self.stride*z2+dy,ff]
                
                
        return u[:self.outdim[0],:self.outdim[1]]        


    def adj(self,u,k):


        c = np.zeros(self.indim)

    
        #Artifical extension to avoid case disctinction at boundary
        ux = np.zeros([ self.outdim[ii]+np.remainder(-self.outdim[ii],self.stride) for ii in range(2)]) 
        ux[:self.outdim[0],:self.outdim[1]] = u

        for ff in range(self.nf):        
            for dx in range(self.stride):
                for dy in range(self.stride):
                    for z1 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                        for z2 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                          
                            c[self.bln-z1:self.nx-z1,self.bln-z2:self.ny-z2,ff] += ux[dx::self.stride,dy::self.stride]*k[self.stride*z1+dx,self.stride*z2+dy,ff]
                
                
        return c
        
    def adj_ker(self,u,c):


        k = np.zeros(self.indim2)

    
        #Artifical extension to avoid case disctinction at boundary
        ux = np.zeros([ self.outdim[ii]+np.remainder(-self.outdim[ii],self.stride) for ii in range(2)]) 
        ux[:self.outdim[0],:self.outdim[1]] = u

        for ff in range(self.nf):        
            for dx in range(self.stride):
                for dy in range(self.stride):
                    for z1 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                        for z2 in range(int(np.ceil(float(self.ksz)/float(self.stride)))):
                          
                            k[self.stride*z1+dx,self.stride*z2+dy,ff] += np.sum(ux[dx::self.stride,dy::self.stride]*c[self.bln-z1:self.nx-z1,self.bln-z2:self.ny-z2,ff])
                
                
        return k

    def test_adj(self):

        #Coefficient adjoint
        k = np.random.rand(*self.indim2)

        fwd = lambda x: self.fwd(x,k)
        adj = lambda x: self.adj(x,k)

        test_adj(fwd,adj,self.indim,self.outdim)

        #Kernel adjoint
        c = np.random.rand(*self.indim)

        fwd = lambda x: self.fwd(c,x)
        adj = lambda x: self.adj_ker(x,c)


        test_adj(fwd,adj,self.indim2,self.outdim)



class patch_moment(object):


    def __init__(self,indim):
                    
        #Set 2D convolution function
        self.ksz = indim[-1]
        self.indim = indim
        self.outdim = (self.indim[0],self.indim[1],1,1,2)
        
        #Generate linspace for broadcasting
        tmpx = np.linspace(-1.0,1.0,self.ksz)
        mg1,mg2 = np.meshgrid(tmpx,tmpx)
        
        
        self.x = np.zeros((1,1,self.ksz,self.ksz))
        self.x[0,0,:,:] = mg1
        
        self.y = np.zeros((1,1,self.ksz,self.ksz))
        self.y[0,0,:,:] = mg2
        
        
        #Set norm squared
        self.nrm = np.sqrt(2.0*np.square(mg1).sum())
            
    def fwd(self,c):
    
        #return 0.5*(c[:,:-1]*self.x[:,:-1] + c[:,1:]*self.x[:,1:]).sum(axis=1,keepdims=True)
    
        return np.stack( [(c*self.x).sum(axis=(2,3),keepdims=True),(c*self.y).sum(axis=(2,3),keepdims=True)],axis=4 )


    def adj(self,g):
    
        #c = g*self.x
        #c[:,0] = 0.5*c[:,0]
        #c[:,-1] = 0.5*c[:,-1]
        #return c
        
        return g[...,0]*self.x + g[...,1]*self.y


    def test_adj(self):
       
        test_adj(self.fwd,self.adj,self.indim,self.outdim)




class patch_mean(object):


    def __init__(self,indim):
                    
        #Set dimensions
        self.ksz = indim[-1]
        self.indim = indim
        self.outdim = (self.indim[0],self.indim[1],1,1)
        
        
        self.x = np.ones([1,1,self.ksz,self.ksz])
        
        self.nrm = self.ksz
        
    def fwd(self,c):
    
        return c.sum(axis=(2,3),keepdims=True)

    def adj(self,g):
    
        
        return g*self.x


    def test_adj(self):
       
        test_adj(self.fwd,self.adj,self.indim,self.outdim)



###########################################################
# Algorithms ##############################################

#Solves min_{u,c} \|u-u0\| + TGV(u-Kc) + \|c\|_nuc + \|c\|_{1,\infty} s.t. 0 and 1 moment of c[i,:]=0
def tgv_patch(**par_in):

    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})
    
    ##Set data
    data_in.u0 = 0 #Direct image input
    #Ground truth data inputcoefficient input
    data_in.data = {} #Expecting {'A':A,'K':K}
    data_in.mask = 0 #Inpaint requires a mask


    #Possible forward operator. Standard (False) sets identity. Needs to have F.outdim, F.nrm, F.fwd, F.adj
    data_in.F = False

    #Version information:
    #version='Version 1, zero-moment prox included, decoupled nuc and positivity prox'
    #version='Version 2, added optional zero mean constraint'
    par.version='Version 3, added delta-paramter for semi-convex prior'

    par.imname = 'barbara_crop.png'
   
    par.dtype='l2sq' #Data type: {'l1','l2sq','inpaint','I0'}
    
    par.sp_type = 'l1infty' #{'l1','l1infty','linfty'}
    par.nuc_type = 'l1-svd' #{'l1-svd','semiconv-svd','l1_2-svd','l0-svd'}
    
    # Parameter for semi-convex penalty; we need 1>= 2*delta*eps*tau*nucpar
    par.semiconv_eps = 2.0 
    par.semiconv_delta = 0.99
    
    par.use_mean = True #Flag to use mean instead of 1-moment + positivity
    
    
    par.stride = 3
    
    par.niter = 1

    par.ksz = 9 #Size of filter kernels
    
    par.ld = 20.0 #Data missfit
    par.mu = 500.0 #TGV vs. dict penalty, in (-infty,infty), where negative values penalize TGV, positive values penalize the dict
    par.nu = 1000.0 #low rank vs sparse penalty, in [0,1000]

    par.alpha0 = np.sqrt(2.0)
    
    par.noise=0.1 #Standard deviaion of gaussian noise in percent of image range
    
    par.s_t_ratio = False

    par.check = 10 #Show results ever "check" iteration
    
    par.nrm_red = 1.0 #Factor for norm reduction, no convergence guarantee with <1

    par.show = False


    ##Data and parmaeter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])


    #Load data or image
    if np.any(data_in.data):
        K = data_in.data['K']
        A = data_in.data['A']
        
        #Update kernel size ans stride to fit with K
        par.ksz = K.ksz
        par.stridestride = K.stride
    
        if not np.any(data_in.u0):
            data_in.u0 = K.fwd(A)
        
    if not np.any(data_in.u0):
        data_in.u0 = imread(par.imname)


    
    #Set froward operator if necessary
    par.datadual = True
    if not data_in.F:
        data_in.F = id(data_in.u0.shape)
        par.datadual = False  


    ##Parameter parsing
    
    #Automatic adaption of s_t_ratio to lambda
    if not np.any(par.s_t_ratio):
        par.s_t_ratio = 5.0*par.ld
        print('Adapted s_t_ratio to ' + str(par.s_t_ratio))

    
    #Parameter refactorization
    if par.nu<0 or par.nu> 1000:
        raise ValueError('Error: nu must be in [0,1000]')
    par.nu /=1000.0

    #Set reguarlization parameter
    par.ptv = 1.0 - min(par.mu,0.0)
    par.pnuc = par.nu*(1.0 + max(par.mu,0.0))
    par.plp = (1-par.nu)*(1.0 + max(par.mu,0.0))    

    print('Parameter for TV/nuc/l1: ' + str(par.ptv) + '/' + str(par.pnuc) + '/' + str(par.plp))


    ## Data initilaization
    F = data_in.F
    u0 = np.copy(data_in.u0)
    u0 = F.fwd(u0)
    
    #Add noise
    if par.noise:
        np.random.seed(1)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise
    if np.any(data_in.mask):
        u0 = np.copy(u0)
        u0[~data_in.mask] = 0.0
    

    #Image size
    N,M = u0.shape

    #Operators and norms
    if not np.any(data_in.data):
        K = lconv([N,M,par.ksz],stride=par.stride)

    grad = gradient(u0.shape)
    sgrad = symgrad(grad.outdim)
    
    #Used for moment or positivity constraints
    M = patch_moment(K.indim)
    P = patch_mean(K.indim)

    zconst = nfun('I0')
    pconst = nfun('I0')

    nucnrm = nfun(par.nuc_type,npar=par.pnuc,eps=par.semiconv_eps,delta=par.semiconv_delta,vdims=(2,3))
    pnrm = nfun(par.sp_type,npar=par.plp,vdims=(2,3))
    dnrm = nfun(par.dtype,mshift=u0,npar=par.ld,mask=data_in.mask)
    
    l1vec = nfun('l1',vdims=(2),npar=par.ptv)
    l1mat = nfun('l1',vdims=(2),npar=par.alpha0*par.ptv,symgrad=True)

    
    #nucnrm = nfun('usprox',npar=pnuc,vdims=(2,3))
    #pnrm = nfun('zero')

    #Modification to positivity constraint
    if not par.use_mean:
        P = idfun(K.indim)
        pconst = nfun('IP')
        

    ##Stepsize
    opnorms = [ 
    [grad.nrm,grad.nrm*K.nrm,1],
    [0,0,sgrad.nrm],
    [0,P.nrm,0],
    [0,M.nrm,0],
    [0,1,0] ]
    if par.datadual:
        opnorms.append([F.nrm,0,0])
        
    nrm = par.nrm_red*get_product_norm(opnorms) 

    print('Estimated norm: ' + str(nrm))
    
    par.sig = par.s_t_ratio/nrm
    par.tau = 1/(nrm*par.s_t_ratio)

    print('Stepsizes: sig: '+str(par.sig)+' / tau: '+str(par.tau))

    #Adapt stepsize to ensure uniquness in prox
    if (par.nuc_type == 'semiconv-svd') & (2.0*par.semiconv_eps*par.semiconv_delta*par.tau*par.pnuc>1.0):
    
        tmp = 0.995/(2.0*par.semiconv_eps*par.semiconv_delta*par.pnuc)
        
        par.sig = par.sig/(tmp/par.tau)
        par.tau = par.tau*(tmp/par.tau)
            
        print('Adapted stepsizes for semiconvex prox to:  sig: '+str(par.sig)+' / tau: '+str(par.tau))


    ##Variables

    #Primal
    u = np.zeros(u0.shape)
    ux = np.zeros(u.shape)
    
    v = np.zeros(grad.outdim)
    vx = np.zeros(v.shape)

    c = np.zeros(K.indim)        
    cx = np.zeros(c.shape)
            
    #Dual
    p = np.zeros(grad.outdim) #Gradient

    q = np.zeros(sgrad.outdim) #Symgrad
    
    r = np.zeros(c.shape) #Pnorm
    s = np.zeros(P.outdim) #Positivity or zero moment
    
    m = np.zeros(M.outdim) #First moment
    
    if par.datadual: #If we have a forward operator, the data term needs to be dualized
        d = np.zeros(F.outdim)

    ob_val = np.zeros([par.niter+1])

    ob_val[0] = dnrm.val(F.fwd(u)) + l1vec.val(grad.fwd(u - K.fwd(c)) - v) + l1mat.val(sgrad.fwd(v)) + nucnrm.val(c) + pnrm.val(c) + zconst.val(M.fwd(c)) + pconst.val(P.fwd(c))
    
    for k in range(par.niter):

        
        #Dual   
        if par.datadual:
            d = d + par.sig*( F.fwd(ux) )
            d = dnrm.dprox(d,ppar=par.sig)
        
        p = p + par.sig*( grad.fwd(ux - K.fwd(cx)) - vx )
        p = l1vec.dprox(p,ppar=par.sig)
                
        q = q + par.sig*( sgrad.fwd(vx) )
        q = l1mat.dprox(q,ppar=par.sig)
        
        r = r + par.sig*( cx )
        r = pnrm.dprox(r,ppar=par.sig)
        
        s = s + par.sig*( P.fwd(cx) )
        s = pconst.dprox(s,ppar=par.sig)

        m = m + par.sig*( M.fwd(cx) )
        m = zconst.dprox(m,ppar=par.sig)


        #Primal
        if par.datadual: 
            ux = u - par.tau*(grad.adj(p) + F.adj(d) )
        else:
            ux = dnrm.prox( u - par.tau*(grad.adj(p))   ,ppar=par.tau)
            
        cx = nucnrm.prox( c - par.tau*( - K.adj(grad.adj(p)) + r + P.adj(s) + M.adj(m) ),ppar=par.tau)
        vx =            v - par.tau*( -p + sgrad.adj(q))

        u = 2.0*ux - u
        c = 2.0*cx - c
        v = 2.0*vx - v

        [u,ux] = [ux,u]
        [v,vx] = [vx,v]
        [c,cx] = [cx,c]

        ob_val[k+1] = dnrm.val(F.fwd(u)) + l1vec.val(grad.fwd(u - K.fwd(c)) - v) + l1mat.val(sgrad.fwd(v)) + nucnrm.val(c) + pnrm.val(c) + zconst.val(M.fwd(c)) + pconst.val(P.fwd(c))

        if np.remainder(k,par.check) == 0:
            print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k]) + ' / Image: '+par.imname)

    ## Data output

    #Initialize output class
    
    #Initialize output class
    res = output(par)


    #Set original input
    res.orig = data_in.u0
    res.u0 = u0


    #Save variables
    res.u = u
    res.v = v
    res.c = c
    res.ob_val = ob_val
    if np.any(data_in.data):
        res.A = A
    
    res.rank,res.sig = rank(mshape(res.c),tol=1e-5)
    
    print('Rank: ' + str(res.rank))
    

    #Save operators
    res.K = K
    res.grad = grad
    res.sgrad = sgrad
    res.M = M
    res.P = P
    
    #Save parmeters
    res.par = par
    res.par_in = par_in
    
    #Show and get coefficients
    mc = mshape(res.c)
    res.coeff,res.patch,res.sig = decomp_lifted(mc,res.K,nfs=16,show=par.show)


    if par.show:
        imshow(np.concatenate((res.orig,res.u0,res.u),axis=1),title='Orig vs noisy vs rec')

        #Composition
        comp = np.concatenate([res.u - res.K.fwd(res.c),res.K.fwd(res.c)],axis=1)
        imshow(comp,title='Components: Cartoon vs text')

        
        #Show matrix
        imshow(mc[:min(5*mc.shape[1],mc.shape[0]),:],title='Recon matrix')   
        #Show data matrix
        if np.any(data):
            mA = mshape(res.A)
            imshow(mA[:min(5*mA.shape[1],mA.shape[0]),:],title='Data matrix')
     

        #Plot objective
        plot(ob_val,title='Objective value')    


    return res





#Solves min_{c} \|Kc-u0\| + \|c\|_nuc + \|c\|_{1,\infty}
def patch_recon(**par_in):

    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})


    ##Set data
    data_in.u0 = 0 #Direct image input
    #Ground truth data inputcoefficient input
    data_in.data = {} #Expecting {'A':A,'K':K}
    data_in.mask = 0 #Inpaint requires a mask

    #Possible forward operator. Standard (False) sets identity. Needs to have F.outdim, F.nrm, F.fwd, F.adj
    data_in.F = False

    #version='Version 1, zero-moment prox included, decoupled nuc and positivity prox'
    #version='Version 2, added optional zero mean constraint'
    #version='Version 3, added optional forward operator'
    par.version='Version 4, added semiconvex parameter delta'

    par.imname = 'barbara_crop.png'



    par.dtype='l2sq' #Data type: {'l1','l2sq','inpaint','I0'}
    
    par.sp_type = 'l1infty' #{'l1','l1infty','linfty'}
    par.nuc_type = 'l1-svd' #{'l1-svd','semiconv-svd','l1_2-svd','l0-svd'}
    
    
    # Parameter for semi-convex penalty; we need 1>= 2*delta*eps*tau*nucpar
    par.semiconv_eps = 2.0 
    par.semiconv_delta = 0.99
    
    
    par.stride = 3
    
    par.niter = 1

    par.ksz = 9 #Size of filter kernels
    
    
    
    par.ld = 20.0 #Data missfit
    par.nu = 1000.0 #low rank vs sparse penalty, in [0,1000]
    
    par.noise=0.1 #Standard deviaion of gaussian noise in percent of image range
    
    par.s_t_ratio = False

    par.check = 10 #Show results ever "check" iteration
    
    par.nrm_red = 1.0 #Factor for norm reduction, no convergence guarantee with <1

    par.show = False
    
    ##Data and parmaeter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])


    #Load data or image
    if np.any(data_in.data):
        K = data_in.data['K']
        A = data_in.data['A']
        
        #Update kernel size ans stride to fit with K
        par.ksz = K.ksz
        par.stridestride = K.stride
    
        if not np.any(data_in.u0):
            data_in.u0 = K.fwd(A)
        
    if not np.any(data_in.u0):
        data_in.u0 = imread(par.imname)
        
        
    #Set forward operator if necessary
    if not data_in.F:
        data_in.F = id(data_in.u0.shape)

     ##Parameter parsing
    
    #Automatic adaption of s_t_ratio to lambda
    if not np.any(par.s_t_ratio):
        par.s_t_ratio = 5.0*par.ld
        print('Adapted s_t_ratio to ' + str(par.s_t_ratio))

    
    #Parameter refactorization
    if par.nu<0 or par.nu> 1000:
        raise ValueError('Error: nu must be in [0,1000]')
    par.nu /=1000.0
    
    #Set reguarlization parameter
    par.pnuc = par.nu
    par.plp = 1-par.nu
    

    print('Parameter for nuc/lp: ' + str(par.pnuc) + '/' + str(par.plp))


   ## Data initilaization
    F = data_in.F
    u0 = np.copy(data_in.u0)
    u0 = F.fwd(u0)
    
    #Add noise
    if par.noise:
        np.random.seed(1)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise
    if np.any(data_in.mask):
        u0 = np.copy(u0)
        u0[~data_in.mask] = 0.0
    

    #Image size
    N,M = u0.shape


   #Operators and norms
    if not np.any(data_in.data):
        K = lconv([N,M,par.ksz],stride=par.stride)

    
    nucnrm = nfun(par.nuc_type,npar=par.pnuc,eps=par.semiconv_eps,delta=par.semiconv_delta,vdims=(2,3))
    pnrm = nfun(par.sp_type,npar=par.plp,vdims=(2,3))
    dnrm = nfun(par.dtype,mshift=u0,npar=par.ld,mask=data_in.mask)
    

    ##Stepsize
    nrm = par.nrm_red*get_product_norm( [ 
    [K.nrm*F.nrm],
    [1.0] ] )

    print('Estimated norm: ' + str(nrm))
    
    par.sig = par.s_t_ratio/nrm
    par.tau = 1/(nrm*par.s_t_ratio)

    print('Stepsizes: sig: '+str(par.sig)+' / tau: '+str(par.tau))

    #Adapt stepsize to ensure uniquness in prox
    if (par.nuc_type == 'semiconv-svd') & (2.0*par.semiconv_eps*par.semiconv_delta*par.tau*par.pnuc>1.0):
    
        tmp = 0.995/(2.0*par.semiconv_eps*par.semiconv_delta*par.pnuc)
        
        par.sig = par.sig/(tmp/par.tau)
        par.tau = par.tau*(tmp/par.tau)
            
        print('Adapted stepsizes for semiconvex prox to:  sig: '+str(par.sig)+' / tau: '+str(par.tau))



    ##Variables

    #Primal
    c = np.zeros(K.indim)        
    cx = np.zeros(c.shape)
            
    #Dual
    p = np.zeros(F.outdim)
    r = np.zeros(c.shape) #Pnorm

    ob_val = np.zeros([par.niter+1])

    ob_val[0] = dnrm.val(F.fwd(K.fwd(c))) + nucnrm.val(c) + pnrm.val(c)
    
    for k in range(par.niter):

        
        #Dual        
        p = p + par.sig*( F.fwd(K.fwd(cx)) )
        p = dnrm.dprox(p,ppar=par.sig)
                
        r = r + par.sig*( cx )
        r = pnrm.dprox(r,ppar=par.sig)

        #Primal
        cx = nucnrm.prox( c - par.tau*( K.adj(F.adj(p)) + r ),ppar=par.tau)

        c = 2.0*cx - c

        [c,cx] = [cx,c]

        ob_val[k+1] = dnrm.val(F.fwd(K.fwd(c))) + nucnrm.val(c) + pnrm.val(c)

        if np.remainder(k,par.check) == 0:
            print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[k]) + ' / Image: '+par.imname)

    
    ## Data output

    #Initialize output class
    
    #Initialize output class
    res = output(par)


    #Set original input
    res.orig = data_in.u0
    res.u0 = u0

    
    #Save variables
    res.c = c
    res.u = K.fwd(c)
    res.ob_val = ob_val
    if np.any(data_in.data):
        res.A = A

    
    res.rank,res.sig = rank(mshape(res.c),tol=1e-5)
    
    print('Rank: ' + str(res.rank))
    

    #Save operators
    res.K = K
    
    
    
    #Save parmeters and input data
    res.par = par
    res.par_in = par_in
    res.data_in = data_in
    


    #Show and get coefficients
    mc = mshape(res.c)
    res.coeff,res.patch,res.par.sig = decomp_lifted(mc,res.K,nfs=16,show=par.show)

    if par.show:
        imshow(np.concatenate((res.orig,res.u0,res.K.fwd(res.c)),axis=1),title='Orig vs noisy vs rec')

        
        #Show matrix
        imshow(mc[:min(5*mc.shape[1],mc.shape[0]),:],title='Recon matrix')   
        #Show data matrix
        if np.any(data_in.data):
            mA = mshape(res.A)
            imshow(mA[:min(5*mA.shape[1],mA.shape[0]),:],title='Data matrix')
     

        #Plot objective
        plot(ob_val,title='Objective value')    

    
    return res




#Solves min_{c} \|Kc-u0\| + \|c\|_nuc + \|c\|_{1,\infty}
def svd_thresh_opt(**par_in):


    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})


    ##Set data
    data_in.u0 = 0 #Direct image input
    #Ground truth data and coefficient input
    data_in.data = {} #Expecting {'A':A,'K':K}

    ##Set parameter
    par.version='Version 1, update to python 3'

    par.imname = 'imsource/barbara_crop.png'
    
    par.stride = 3 #Stride for convoution
    
    par.niter = 1 #Number of iterations

    par.ksz = 9 #Size of filter kernels
    
    
    par.sp_type = 'l1-svd' #Sparsity-type: {'l1-svd','l0-svd'}
    
    #Test range
    par.rg = range(10,100,10)

    par.ksz = 9 #Size of filter kernels

    par.noise=0.1
    
    par.show = False

    ##Data and parmaeter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])


    #Load data or image
    if np.any(data_in.data):
        K = data_in.data['K']
        A = data_in.data['A']
        
        #Update kernel size ans stride to fit with K
        par.ksz = K.ksz
        par.stridestride = K.stride
    
        if not np.any(data_in.u0):
            data_in.u0 = K.fwd(A)
        
    if not np.any(data_in.u0):
        data_in.u0 = imread(par.imname)
        
        
    ## Data initilaization
    u0 = np.copy(data_in.u0)


    #Add noise
    if par.noise:
        np.random.seed(1)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise
    

    #Image size
    N,M = u0.shape


    #Operators and norms
    if not np.any(data_in.data):
        K = lconv([N,M,par.ksz],stride=par.stride)


    #Norm function for threshold
    nrm = nfun(par.sp_type)


    #Array for mse
    mseval = np.zeros(len(par.rg))
    for pos,i in enumerate(par.rg):
    
        mseval[pos] = mse(K.fwd(nrm.prox(K.adj(u0),ppar=float(i))),data_in.u0,rescaled=1)
    
    
    mpos = np.argmin(mseval)

    fac = np.ceil(float(K.ksz)/float(K.stride))**2.0



    #Initialize output class
    res = output(par)


    #Set original input
    res.orig = data_in.u0
    res.u0 = u0


    res.c = nrm.prox(K.adj(res.u0),ppar=float(par.rg[mpos]))
    res.u = K.fwd(res.c)/fac    
    
    res.mseopt = mse(res.u,res.orig,rescaled=1)    

    res.coeff,res.patch,res.sig = decomp_lifted(mshape(res.c),K,nfs=16,show=True)
    
    
    if par.show:
        
        imshow(np.concatenate([res.orig,res.u0,res.u],axis=1),title='Orig vs noisy vs rec')

        #Plot objective
        plot(mseval,title='MSE value')    


    #Save version
    res.version = version

    #Save operator
    res.K = K
    
    return res



def conv_lasso_tv(**par_in):


    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})


    ##Set data
    data_in.u0 = 0 #Direct image input
    #Ground truth data and coefficient input
    data_in.data = {} #Expecting {'A':A,'K':K}

    ##Set parameter
    par.version='Version 2, update to python 3'

    par.imname = 'imsource/barbara_crop.png'
    
    par.stride = 3 #Stride for convoution
    
    par.niter = 1 #Number of iterations

    par.ksz = 9 #Size of filter kernels
    
    par.ld = 20.0 #Data missfit
    par.mu = 500.0 #TGV vs. dict penalty, in (-infty,infty), where negative values penalize TGV, positive values penalize the dict

    par.noise=0.1 #Standard deviaion of gaussian noise in percent of image range
    

    par.check = 10 #Show results ever "check" iteration
    par.show = False
    
    par.eps = 10e-08 #Smoothing of TV norm
    par.nf = 2 #Number of filters


    #Initial stepsizes
    par.Lc = np.exp2(10)
    par.LD = np.exp2(10)
    
    par.beta = 0.707 #Inertia parameter


    ##Data and parmaeter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])


    #Load data or image
    if np.any(data_in.data):
        K = data_in.data['K']
        A = data_in.data['A']
        
        #Update kernel size ans stride to fit with K
        par.ksz = K.ksz
        par.stridestride = K.stride
    
        if not np.any(data_in.u0):
            data_in.u0 = K.fwd(A)
        
    if not np.any(data_in.u0):
        data_in.u0 = imread(par.imname)



    ##Parameter parsing
     
    #Set reguarlization parameter
    par.ptv = 1.0 - min(par.mu,0.0)
    par.pcoeff = 1.0 + max(par.mu,0.0)
    print('Parameter for TV/patch-coefficients: ' + str(par.ptv) + '/' + str(par.pcoeff) )


    #Initialize stepsize
    Lc = par.Lc
    LD = par.LD

    ## Data initilaization
    u0 = np.copy(data_in.u0)
    
    #Add noise
    if par.noise:
        np.random.seed(1)
        
        rg = np.abs(u0.max() - u0.min()) #Image range
        
        u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise
    

    #Image size
    N,M = u0.shape

    #Operators and norms
    if not np.any(data_in.data):
        K = cp_conv([N,M,par.ksz,par.nf],stride=par.stride)


    grad = gradient(u0.shape)


    pnrm = nfun('l1',npar=par.pcoeff,vdims=())
    dnrm = nfun('l2sq',mshift=u0,npar=par.ld)
    
    l1vec = nfun('l1eps',vdims=(2),npar=par.ptv)
    
    l2nsq = nfun('l2sq')

    #Initialize variables
    u = np.zeros(K.outdim)
    D = np.random.rand(*K.indim2)    
    c = np.zeros(K.indim)

    u_old = u
    c_old = c
    D_old = D

    
   
    ob_val = np.zeros(par.niter+1)

    ob_val[0] = dnrm.val(u) + l1vec.val( grad.fwd(u - K.fwd(c,D)) ) + pnrm.val(c) 
        
    print('Iter: ' + str(0) + ', E: ' + str(ob_val[0]) + ', Lc: ' + str(Lc) + ', LD: ' + str(LD))

    for k in range(par.niter):

        uI = u + par.beta*(u - u_old)
        cI = c + par.beta*(c - c_old)
        DI = D + par.beta*(D - D_old)

        u_old = u
        c_old = c
        D_old = D

        ## Descent in image and coefficients
        #Temporary storage
        guc = grad.fwd(uI - K.fwd(cI,D))
        gucn = grad.adj(l1vec.grad(guc))
        
        #Image
        gradu = dnrm.grad(uI) + gucn
        
        #Coefficients
        gradc = -K.adj(gucn,D)

        #Backtracking
        bt = 0
        desc = False
        while bt <=10 and not desc:
            
            #Forward step
            u = uI - (1/Lc)*gradu
            c = cI - (1/Lc)*gradc
            
            #Prox
            c = pnrm.prox(c,ppar=1/Lc)
            
            #Quadratic approximation
            Q = dnrm.val(uI) + l1vec.val(guc) + pnrm.val(cI) + (gradu*(u - uI)).sum() + 0.5*Lc*l2nsq.val( u - uI) + (gradc*(c - cI)).sum() + 0.5*Lc*l2nsq.val( c - cI)
            #New energy
            E = dnrm.val(u) + l1vec.val( grad.fwd(u - K.fwd(c,D) )) + pnrm.val(c)

            
            if E <= 1.0001*Q:
                Lc = Lc/2.0
                desc = True
            else:
                #print(E)
                #print(Q)
                Lc = Lc*2.0
                bt += 1
            

        ## Descent in Dictionary
        gD = grad.fwd(u - K.fwd(c,DI))
        gradD = -K.adj_ker(grad.adj( l1vec.grad(gD) ),c)


        #Backtracking
        bt = 0
        desc = False
        while bt <=10 and not desc:

            #Forwad step
            D = DI - (1/LD)*gradD
            #Prox
            D -= D.sum(axis=(0,1),keepdims=True)/(par.ksz*par.ksz) #Mean
            D /= np.maximum(1.0, np.sqrt(np.square(D).sum(axis=(0,1),keepdims=True)) ) #Norm <=1
            
            #Quadratic approximation
            Q = l1vec.val(gD) + (gradD*(D - DI)).sum() + 0.5*LD*l2nsq.val( D - DI )

            #New Energy
            E = l1vec.val( grad.fwd(u - K.fwd(c,D) ))


            if E <=1.0001*Q:
                LD = LD/2.0
                desc = True
            else:
                LD = LD*2.0
                bt += 1


        ob_val[k+1] = dnrm.val(u) + l1vec.val( grad.fwd(u - K.fwd(c,D) )) + pnrm.val(c) 

        if np.remainder(k,par.check) == 0:

            
            print('Iter: ' + str(k+1) + ', E: ' + str(ob_val[k+1]) + ', Lc: ' + str(Lc) + ', LD: ' + str(LD))
            
    if par.show:        
        #Noise vs denoised
        imshow(np.concatenate([u0,u],axis=1),title='Noisy vs. Denoised')

        #Composition
        comp = np.concatenate([u - K.fwd(c,D),K.fwd(c,D)],axis=1)
        imshow(comp,title='Components: Cartoon vs text')
        imshow(c,title='Coefficients')

        #Filter kernels
        imshow(D,title='filter kernels')


    #Initialize output class
    res = output(par)


    #Set original input
    res.orig = data_in.u0
    res.u0 = u0
            
    res.D = D
    res.c = c
    res.u = u
    res.imsyn = K.fwd(c,D)
    res.K = K
    res.ob_val = ob_val
    
    return res




