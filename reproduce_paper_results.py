import matpy as mp
import convex_learning as cl
import numpy as np
import os


#Function to save the output and compute PSNR values
def save_output(res,psnr=True,folder=''):

    #Create folder if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    #Generate output name
    outname = res.output_name(folder=folder)

    #Save result file
    res.save(folder=folder)
    #Save original image
    rg=[res.orig.min(),res.orig.max()]
    mp.imsave(outname+'_orig.png',res.orig,rg=rg)
    #Save data image
    mp.imsave(outname+'_data.png',res.u0,rg=rg)
    #Save reconstructed image
    mp.imsave(outname+'_recon.png',res.u,rg=rg)
    
    if hasattr(res,'K') and hasattr(res,'c'):
        #Save cartoon part
        mp.imsave(outname+'_cart.png',res.u - res.K.fwd(res.c),rg=[])
        #Save texture part
        mp.imsave(outname+'_text.png',res.K.fwd(res.c),rg=[])
    if hasattr(res,'patch'):
        #Save patches
        patches = mp.imshowstack(res.patch[...,:9])
        mp.imsave(outname+'_patches.png',patches,rg=[])

    if psnr:
        psnr_val = np.round(mp.psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=True),decimals=2)
        print( res.output_name() + '\nPSNR: ' + str(psnr_val) )


################################
################################


## Choose result type to compute
types = []
types = ['decomp','inpaint','denoise','deblurring']
outfolder = 'paper_results/'

#Cartoon-texture decomposition
if 'decomp' in types:

    folder = outfolder + 'decomp/'
    fixpars = {'niter':5000,'sp_type':'l1','ksz':15,'stride':3,'noise':0.0,'ld':100}

    #Texture
    u0,A,K = cl.get_texttest(N=120,ksz=15,stride=3)
    res = cl.patch_recon(imname='texttest.png',u0=u0,nu=750,**fixpars)
    save_output(res,psnr=False,folder=folder)
    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    res = cl.tgv_patch(imname='patchtest.png',u0=u0,mu=20,nu=950,**fixpars)
    save_output(res,psnr=False,folder=folder)
    #Mix
    res = cl.tgv_patch(imname='imsource/cart_text_mix.png',mu=40,nu=950,**fixpars)
    save_output(res,psnr=False,folder=folder)
    #Barbara
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',mu=40,nu=950,**fixpars)
    save_output(res,psnr=False,folder=folder)



#Inpainting
if 'inpaint' in types:

    folder = outfolder + 'inpaint/'
    fixpars = {'niter':5000,'sp_type':'l1','ksz':15,'stride':3,'noise':0.0,'ld':1,'dtype':'inpaint'}

    #Texture
    u0,A,K = cl.get_texttest(N=120,ksz=15,stride=3)
    mask = cl.get_mask(u0.shape,perc=20.0)
    res = cl.patch_recon(imname='texttest.png',u0=u0,mask=mask,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    mask = cl.get_mask(u0.shape,perc=30.0)
    res = cl.tgv_patch(imname='patchtest.png',u0=u0,mask=mask,mu=30,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Mix
    mask = cl.get_mask([128,128],perc=20.0)
    res = cl.tgv_patch(imname='imsource/cart_text_mix.png',mask=mask,mu=40,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    mask = cl.get_mask([128,128],perc=30.0)
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',mask=mask,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)

#TGV inpainting

    folder = outfolder + 'inpaint_tgv/'
    fixpars = {'niter':2000,'noise':0.0,'ld':1,'dtype':'inpaint'}

    #Texture
    u0,A,K = cl.get_texttest(N=120,ksz=15,stride=3)
    mask = cl.get_mask(u0.shape,perc=20.0)
    res = mp.tgv_recon(imname='texttest.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    mask = cl.get_mask(u0.shape,perc=30.0)
    res = mp.tgv_recon(imname='patchtest.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Mix
    mask = cl.get_mask([128,128],perc=20.0)
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    mask = cl.get_mask([128,128],perc=30.0)
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)

#Semiconvex variant

    folder = outfolder + 'inpaint_semiconvex/'
    fixpars = {'niter':5000,'sp_type':'l1','ksz':15,'stride':3,'noise':0.0,'ld':1,'dtype':'inpaint','nuc_type':'semiconv-svd','semiconv_delta':0.99,'semiconv_eps':0.1}

    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    mask = cl.get_mask(u0.shape,perc=30.0)
    res = cl.tgv_patch(imname='patchtest.png',u0=u0,mask=mask,mu=30,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    mask = cl.get_mask([128,128],perc=30.0)
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',mask=mask,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)



#Denoising
if 'denoise' in types:

    folder = outfolder + 'denoise/'
    fixpars = {'niter':5000,'sp_type':'l1','ksz':15,'stride':3}

    #Texture
    u0,A,K = cl.get_texttest(N=120,ksz=15,stride=3)
    res = cl.patch_recon(imname='texttest.png',u0=u0,noise=0.5,ld=0.09,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    res = cl.tgv_patch(imname='patchtest.png',u0=u0,noise=0.1,ld=10.0,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Mix
    res = cl.tgv_patch(imname='imsource/cart_text_mix.png',noise=0.1,ld=10.0,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',noise=0.1,ld=12.5,mu=70,nu=975,**fixpars) #Should be ld15,mu70,nu950
    save_output(res,psnr=True,folder=folder)
    

#TGV results
        
    folder = outfolder + 'denoise_tgv/'
    fixpars = {'niter':2000}

    #Texture
    u0,A,K = cl.get_texttest(N=120,ksz=15,stride=3)
    res = mp.tgv_recon(imname='texttest.png',u0=u0,noise=0.5,ld=7.5,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    res = mp.tgv_recon(imname='patchtest.png',u0=u0,noise=0.1,ld=10.0,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Mix
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',noise=0.1,ld=12.5,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',noise=0.1,ld=17.5,**fixpars)
    save_output(res,psnr=True,folder=folder)


#Semiconvex variant

    folder = outfolder + 'denoise_semiconvex/'
    fixpars = {'niter':5000,'sp_type':'l1','ksz':15,'stride':3,'noise':0.1,'nuc_type':'semiconv-svd','semiconv_eps':2.0,'semiconv_delta':0.99}

    #Patches
    u0,A,K = cl.get_patchtest(N=120,ksz=15,stride=3)
    res = cl.tgv_patch(imname='patchtest.png',u0=u0,ld=7.5,mu=100,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',ld=12.5,mu=100,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
 
    
    
#Deblurring
if 'deblurring' in types:

    folder = outfolder + 'deblurring/'
    F = mp.gconv([128,128],9,0.25)
    fixpars = {'niter':5000,'noise':0.025,'ksz':15,'stride':3,'sp_type':'l1','F':F}


    #Mix
    res = cl.tgv_patch(imname='imsource/cart_text_mix.png',ld=500.0,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    res = cl.tgv_patch(imname='imsource/barbara_crop.png',ld=350,mu=50,nu=975,**fixpars)
    save_output(res,psnr=True,folder=folder)
    

#TGV results
        
    folder = outfolder + 'deblurring_tgv/'
    F = mp.gconv([128,128],9,0.25)
    fixpars = {'niter':5000,'noise':0.025,'F':F}


    #Mix
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',ld=750,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',ld=500,**fixpars)
    save_output(res,psnr=True,folder=folder)
    
    


