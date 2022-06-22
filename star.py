import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class StarFilter(nn.Module):

    def __init__(self, alpha=0.001, beta=0.0001, pI = 1.5, pR = 0.5, r=3, , vareps=0.01, K=20, debug=True):
        self.alpha = alpha
        self.beta = beta
        self.pI = pI
        self.pR = pR
        self.r = (r-1) // 2
        self.eps = 1e-5
        self.debug = debug
        self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)

    def star(self, S):
        I = S.copy()
        R = np.ones_like(S)
        if self.debug:
            print('-- Stop iteration until eplison < {} or K > {}'.format(self.vareps, self.K))

        for iter in range(1, self.K):
            preI = I
            preR = R
            I = S / R

            '''
            Ix = np.diff(I, n=1, axis=1)
            Ix = np.pad(Ix, [(0, 0), (0, 1)], 'edge')
            Iy = np.diff(I, n=1, axis=0);                
            Iy = np.pad(Iy, [(0, 1), (0, 0)], 'edge')
            '''
            Ix = diff(I,1,2);                   % Estimate diff on the x-axis
            Ix = padarray(Ix, [0 1], 'post');   % Post ->
            Iy = diff(I,1,1);                   %
            Iy = padarray(Iy, [1 0], 'post');   %

            '''
            avgIx=convbox(Ix, r)
            avgIy=convbox(Iy, r)
            '''
            avgIx=convBox( single(Ix), r);      % so convbox should be a specific thing?? r - is
            avgIy=convBox( single(Iy), r);

            '''
            ux = np.maximum(np.abs(avgIx)**pI,eps)**(-1)
            uy = np.maximum(np.abs(avgIy)**pI,eps)**(-1)
            ux[:,-1] = 0
            uy[-1,:] = 0
            '''
            ux = max(abs(avgIx).^pI,eps).^(-1);  % structure map avgIx.^pI > avgIx.*Ix > Ix.^2
            uy = max(abs(avgIy).^pI,eps).^(-1);  % structure map
            ux(:,end) = 0;
            uy(end,:) = 0;

            I = solveLinearSystem(S, R, ux, uy, alpha);  % Eq.(12)
            eplisonI = norm(I-preI, 'fro')/norm(preI, 'fro');   % iterative error of I

            %% algorithm for P2
            %pR=min(pI,pR);
            R=S./I;
            '''
            Ix = np.diff(I, n=1, axis=1)
            Ix = np.pad(Ix, [(0, 0), (0, 1)], 'edge')
            Iy = np.diff(I, n=1, axis=0);                
            Iy = np.pad(Iy, [(0, 1), (0, 0)], 'edge')
            '''
            Rx = diff(R,1,2);
            Rx = padarray(Rx, [0 1], 'post');
            Ry = diff(R,1,1);
            Ry = padarray(Ry, [1 0], 'post');

            '''
            avgIx=convbox(Ix, r)
            avgIy=convbox(Iy, r)
            '''
            avgRx=convBox( single(Rx), r);
            avgRy=convBox( single(Ry), r);

            '''
            ux = np.maximum(np.abs(avgIx)**pI,eps)**(-1)
            uy = np.maximum(np.abs(avgIy)**pI,eps)**(-1)
            ux[:,-1] = 0
            uy[-1,:] = 0
            '''
            vx = max(abs(avgRx).^pR,eps).^(-1);  % texture map
            vy = max(abs(avgRy).^pR,eps).^(-1);  % texture map
            vx(:,end) = 0;
            vy(end,:) = 0;

            R = solveLinearSystem(S, I, vx, vy, beta);            	% Eq.(13)
            eplisonR = norm(R-preR, 'fro')/norm(preR, 'fro');   % iterative error of R

            %% iteration until convergence
            if debug == true
                fprintf('Iter #%d : eplisonI = %f; eplisonR = %f\n', iter, eplisonI, eplisonR);
            end
            if(eplisonI<vareps||eplisonR<vareps)
                break;
            end
        end
        I(I<0)=0;
        R(R<0)=0;
        return I, R



    '''
    function dst = solveLinearSystem(s, ir, uvx, uvy, alphabet, b, lambda, method)
        if (~exist('b','var')) 
           b = 0;
           lambda = 0;
        end
        if (~exist('method','var'))
           method = 'pcg';
        end

        [h, w] = size(s);
        hw = h * w;
        %% calculate the five-point positive definite Laplacian matrix
        uvx = uvx(:);
        uvy = uvy(:);
        ux = padarray(uvx, h, 'pre'); 
        ux = ux(1:end-h);
        uy = padarray(uvy, 1, 'pre'); 
        uy = uy(1:end-1);
        D = uvx + ux + uvy + uy;
        T = spdiags([-uvx, -uvy],[-h,-1],hw,hw);
        %% calculate the variable of linear system
        MN = T + T' + spdiags(D, 0, hw, hw);                % M in Eq.(12) or N in Eq.(13)    
        ir2 = ir.^2;                                        % R^{T}R in Eq.(12) or I^{T}I in Eq.(13)
        ir2 = spdiags(ir2(:), 0, hw, hw); 
        DEN = ir2 + alphabet * MN + lambda * speye(hw,hw);  % denominator in Eq.(12) or Eq.(13)
        NUM = ir.*s + lambda * b;                           % numerator in Eq.(12) or Eq.(13) 
        %% solve the linear system
        switch method
            case 'pcg'
                L = ichol(DEN,struct('michol','on'));    
                [dst,~] = pcg(DEN, NUM(:), 0.01, 40, L, L'); 
            case 'minres'
                [dst,~] = minres(DEN,NUM(:), 0.01, 40);
            case 'bicg'
                [L,U] = ilu(DEN,struct('type','ilutp','droptol',0.01));
                [dst,~] = bicg(DEN,NUM(:), 0.01, 40, L, U);
            case 'direct'
                [dst,~] = DEN\NUM(:); %#ok<RHSFN>
        end
        dst = reshape(dst, h, w);
    end
    '''

    '''
    subroutine prog5(a,b,c,d,f,y,x,m)
      implicit real*8 (a-h,o-z)
      dimension a(m),b(m),c(m),d(m),f(m),y(m),x(m)
      do j=3,m
        e=b(j-1)/c(j-2)
        c(j-1)=c(j-1)-e*d(j-2)
        d(j-1)=d(j-1)-e*f(j-2)
        y(j-1)=y(j-1)-e*y(j-2)
        e=a(j)/c(j-2)
        b(j)=b(j)-e*d(j-2)
        c(j)=c(j)-e*f(j-2)
        y(j)=y(j)-e*y(j-2)
      end do
      e=b(m)/c(m-1)
      c(m)=c(m)-e*d(m-1)
      y(m)=y(m)-e*y(m-1)
      x(m)=y(m)/c(m)
      x(m-1)=(y(m-1)-d(m-1)*x(m))/c(m-1)
      do j=m-2,1,-1
        x(j)=(y(j)-d(j)*x(j+1)-f(j)*x(j+2))/c(j)
      end do
      return
    end
    '''

    def forward(self, x):
        I, R = None, None
        return I, R

if __name__=='__main__':
    test_low_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    test_low_img = np.transpose(test_low_img, (2, 0, 1))
    input_low_test = np.expand_dims(test_low_img, axis=0)
    input_low_test = Variable(torch.FloatTensor(torch.from_numpy(input_low_test))).cuda()
    R_low, I_low = net(input_low_test)
    R_low = np.clip(np.transpose(R_low.cpu().detach().numpy().squeeze(), (1, 2, 0)), 0, 1)
    I_low = np.clip(I_low.cpu().detach().numpy().squeeze(), 0, 1)
    return R_low, I_low

if __name__ == '__main__':
    net = get_decom()
    FNAME = 'input/input/010.png'
    dehazed_image = cv2.imread(FNAME)
    reflectance, illumination = decom_image(dehazed_image)

    fig, axs = plt.subplots(2, figsize=(16, 8))
    axs[0].imshow(reflectance)
    axs[1].imshow(illumination, cmap='gray')