function [dom_zeros, X, Y, residues, proc_admin ] = ...
           dza_tdefl( A, E, b, c, d, s0, nwanted, tol, max_iter, displ, adv_opts)
% [dom_zeros, X, Y, residues, proc_admin ] = ...
%           dza_tdefl( A, E, b, c, d, s0, nwanted, tol, max_iter, displ, adv_opts)
% computes the dominant zeros of the transfer function
%                   H(s) = c'*(sE-A)^{-1}b + d
% by computing the dominant poles of the corresponding inverse system
%
%                   Az = [A, b; -c', -d] ;
%                   Ez = blkdiag(E, 0) ;
%                   bz = [b; 1] ;
%                   cz = [c; 1] ;
%                   dz = d ;
%
% with the Dominant Pole Algorithm with Turbo Deflation.
%
% Since z = infty is also a dominant zero in many cases, DZA may converge to
% this zero as well, which may be inconvenient. For a more robust variant, use
% SADZA or SAMDZ.
%
% Basic usage : [dom_zeros, X, Y, proc_admin] = dza_tdefl ( A, E, b, c, d, s0, nwanted, tol)
% Input:
%        A,E       : nxn state and descriptor matrices
%        b,c,d     : nx1 input and output map vectors, direct transmission
%        s0        : array with <= nwanted initial shifts
%                    (if size(s0,1) < nwanted, s0(end) will be recycled) (1i)
%        nwanted   : number of wanted zeros (1)
%        tol       : convergence tolerance ||Ax - sEx||_2 (1e-10)
%        max_iter  : maximum number of iterations (100)
%        displ     : display DPA process output (1)
%        adv_opts  : structure with advanced process options
%                    use_lu       : if 1 use lu(sE-A) (default)
%                                   elseif 0 use \
%                    use_lu_w_amd : if 1 use complete UMFPACK reordering
%                    dpa_bordered : if dpa_bordered==1, 
%                                      use bordered matrices in solves
%                                      ie [sE - A, -b ; c', d] 
%                                          (default with use_lu=0)
%                                   else use just sE - A 
%                                          (default with use_lu=1) 
%                    newton_update: if 1 use skp = sk + upd 
%                                   elseif 0 use twosided Rayleigh quotient (default)
%                     
% Output:
%		dom_zeros  : found dominant zeros
%       X, Y       : X,Y(1:n)   : corresponding right and left zeroing vectors
%                    X,Y(1:n+1) : corresponding eigenvectors of inverse system
%       residues   : corresponding residues of inverse system
%       proc_admin : structure with several DPA process figures:
%                    b_defl, c_defl  : deflated b and c vectors
%                    nr_solves       : total number of solves with s_iE-A
%                    dpa_shifts      : shifts used by DPA
%                    shift_cnt       : total number of shifts per found pole
%
% When using this software, please cite
%
%   (for computing dominant zeros)
%   Nelson Martins, Paulo Pellanda, and Joost Rommes
%   Computation of transfer function dominant zeros 
%   with applications to oscillation damping control of large power systems
%   IEEE Trans. on Power Systems, 22 (4), Nov. 2007, pp. 1657-1664
%
%   (for DPA)
%   N. Martins and L. T. G. Lima and H. J. C. P. Pinto
%	Computing Dominant Poles of Power System Transfer Functions
%   IEEE Trans.~Power Syst. 11 (1), pp 162--170, 1996
%
%   (for Turbo Deflation)
%   J. Rommes and N. Martins
%   Efficient computation of transfer function dominant poles of 
%   large second-order dynamical systems
%   SIAM Journal on Scientific Computing 30 (4) pp. 2137-2157, 2008
%
%   (for convergence aids and analysis)
%   J. Rommes and G.L.G. Sleijpen
%   Convergence of the dominant pole algorithm and Rayleigh quotient iteration
%   SIAM Journal on Matrix Analysis and Applications 30 (1), pp. 346--363, 2008
%
%   J. Rommes
%   Methods for eigenvalue problems with applications in model order reduction
%   PhD Thesis, Utrecht University, 2007
%
% Joost Rommes (C) 2006 -- 2010
% rommes@gmail.com
% http://sites.google.com/site/rommes

% Form matrices of inverse system
Az = [A, b; -c', -d] ;
Ez = blkdiag(E, 0) ;
bz = [b; 1] ;
cz = [c; 1] ;
dz = d ;

%call dpa to compute dominant poles of inverse system
[dom_zeros, X, Y, residues, proc_admin ] = ...
           dpa_tdefl( Az, Ez, bz, cz, s0, nwanted, tol, max_iter, displ, adv_opts) ;

if displ
    fprintf( 'Dominant zeros:\n' ) ;
    disp( dom_zeros ) ;
end
