function [dom_zeros, residues, X, Y, nr_solves] = ...
           samdz(A, E, B, C, D, s0, options ) 
% [dom_zeros, residues, leftev, rightev, nr_solves] = ...
%     samdz(A, E, B, C, D, s0, options)
% computes the dominant zeros of MIMO transfer function
%               H(s) = C'*(sE-A)^{-1}B + D 
% of the mimo system
%               E*x' = A*x + B*u
%                 y  = C' * x + D * u
% by computing the dominant poles of the corresponding inverse system
%
%                   Az = [A, B; -C', -D] ;
%                   Ez = blkdiag(E, sparse(m,m)) ; %m = size(B,2) = size(C,2)
%                   Bz = [B; speye(m,m)] ;
%                   Cz = [C; speye(m,m)] ;
%                   Dz = D ;
% using the Subspace Accelerated MIMO Dominant Pole Algorithm of Joost Rommes
% and Nelson Martins. This algorithm is specifically designed for large sparse 
% matrices A and E that allow *cheap* solves  using lu or \.
%
% When using this software, please cite
%    (for computing dominant zeros)
%    Nelson Martins, Paulo Pellanda, and Joost Rommes
%    Computation of transfer function dominant zeros 
%    with applications to oscillation damping control of large power systems
%    IEEE Trans. on Power Systems, 22 (4), Nov. 2007, pp. 1657-1664
%
%    (for SAMDP)
%    Joost Rommes and Nelson Martins: 
%    Efficient computation of multivariable transfer function dominant poles 
%    using subspace acceleration.
%    IEEE Trans. on Power Systems 21 (4), pp. 1471-1483, 2006
%
%    (for several improvements and advanced options)
%    Joost Rommes: 
%    Methods for eigenvalue problems with applications in model order
%    reduction. (Chapter 3, 4)
%    PhD thesis, Utrecht University, 2007.
%
% Input:
%      A, E: (nxn) matrix
%      B: (nxm) matrix
%      C: (nxm) matrix
%      D: (mxm) direct term of system
%      s0: (kx1) initial estimates for pole
%          if size(s0,1) > 1, s0(i) will be injected after the
%              the i-th pole has been found (ie, one solve with
%              (s0(i)*E-A) in the next iteration)
%      options: struct with
%         nwanted : number of wanted poles (5)
%         tol  : tolerance for eigenpair residual (1e-10)
%         displ: show convergence information (1)
%         strategy: select pole p and residue with maximal ('LM'):
%                          one of
%                           'LM' : largest |res| magnitude
%                           'LS' : largest |res|/|p| magnitude
%                           'LR' : largest |res|/|re(p)| magnitude
%         kmin: minimal searchspace dimension (min(1,n))       
%         kmax: maximal searchspace dimension (min(10,n))
%         maxrestart: maximal nr of restarts (100)
%
%         Function handles to user functions for MVs and sovles
%         can be used instead of A and E:
%         f_ax    :   Y = f_ax( tr, X) should return
%                         Az*X if tr = 'N'
%                         Az'*X if tr = 'T'
%         f_ex    :   Y = f_ex( tr, X) should return
%                         Ez*X if tr = 'N'
%                         Ez'*X if tr = 'T'
%         f_semax :   Y = f_semax( tr, X, s) should return
%                         (sEz - Az)*X  if tr = 'N'
%                         (sEz - Az)'*X if tr = 'T'
%         f_semax_s : Y = f_semax_s( tr, X, s) should return
%                         inv(sEz - Az) * X  if tr = 'N'
%                         inv((sEz - Az)') * X if tr = 'T'
%         Note that when using function handles, still dummy A and E need
%         to be supplied as arguments to sadpa_adi, typically just A=E=sparse(n,n).         
%
%         Advanced options:
%         use_lu: if use_lu==1, use LU for solves (default), otherwise \ (0)
%         use_lu_w_amd : if 1 use complete UMFPACK reordering
%         dpa_bordered: if dpa_bordered==1, use bordered matrices in solves
%                           ie [sE - A, -b ; c', d] (default with use_lu=0)
%                       else use just sE - A (default with use_lu=1) 
%         yEx_scaling: if yEx_scaling == 1, scale approximate eigenvectors
%                         during selection such that y'Ex = 1
%                      else scale ||x|| = ||y|| = 1 (default)
%         rqitol: tolerance on ||r|| = ||Ax-lambda Ex||. If ||r||<rqitol,
%                 refine using twosided Rayleigh quotient iteration (1e-4). Note
%                 rqitol > tol
%         turbo_deflation: if turbo_deflation == 1, use b and c to efficiently deflate
%                             (default, see Chapter 3 of PhD Thesis Joost Rommes)
%                          else use classical deflation via explicit
%                               projections with found eigenvectors
%         use_fixed_svd:   if use_fixed_svd == 1 (default)
%                             only compute new singular vectors after restart and
%                             convergence
%                          else compute new singular vectors every iteration
%         do_newton:      if do_newton == 0 (default)
%                             use extracted schur vectors in convergence test
%                         else use newton vectors in convergence test
% Output:
%      dom_zeros: converged poles
%      residues : corresponding residues of inverse system
%      X, Y     : X,Y(1:n)   : corresponding right and left zeroing vectors
%                 X,Y(1:n+1) : corresponding eigenvectors of inverse system
%      nr_solves: number of LU factorizations used
%
% Joost Rommes (C) 2005--2010
% rommes@gmail.com
% http://sites.google.com/site/rommes

% Form matrices of inverse system

m = size(B, 2) ;

Az = [A, B; -C', -D] ;
Ez = blkdiag(E, sparse(m,m)) ;
Bz = [B; speye(m,m)] ;
Cz = [C; speye(m,m)] ;
Dz = D ;

%call dpa to compute dominant poles of inverse system
[dom_zeros, residues, X, Y, nr_solves ] = ...
           samdp(Az, Ez, Bz, Cz, Dz, s0, options) ;

if isfield( options, 'displ' )
    displ = options.displ ;
else
    displ = 0 ;
end

if displ
    fprintf( 'Dominant zeros:\n' ) ;
    disp( dom_zeros ) ;
end
