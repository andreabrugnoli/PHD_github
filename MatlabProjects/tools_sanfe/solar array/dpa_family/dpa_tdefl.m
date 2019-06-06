function [dom_poles, X, Y, residues, proc_admin ] = ...
           dpa_tdefl( A, E, b, c, s0, nwanted, tol, max_iter, displ, adv_opts)
% The Dominant Pole Algorithm with Turbo Deflation for computing dominant poles of
% transfer function H(s) = c'*(sE-A)^{-1}b
%
% Basic usage : [dom_poles, X, Y, proc_admin] = dpa_tdefl ( A, E, b, c, s0, nwanted, tol)
% Input:
%        A,E       : nxn state and descriptor matrices
%        b,c       : nx1 input and output map vectors
%        s0        : array with <= nwanted initial shifts
%                    (if size(s0,1) < nwanted, s0(end) will be recycled) (1i)
%        nwanted   : number of wanted poles (1)
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
%		dom_poles  : found dominant poles
%       X, Y       : corresponding right and left eigenvectors
%       residues   : corresponding residues
%       proc_admin : structure with several DPA process figures:
%                    b_defl, c_defl  : deflated b and c vectors
%                    nr_solves       : total number of solves with s_iE-A
%                    dpa_shifts      : shifts used by DPA
%                    shift_cnt       : total number of shifts per found pole
%
% When using this software, please cite
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

% Initialization
if nargin < 10
    use_lu = 1 ;
    use_lu_w_amd = 1 ;
    newton_update = 0 ;
    dpa_bordered = 0 ;
else
    if isfield( adv_opts, 'use_lu')
        use_lu = adv_opts.use_lu ;
    else
        use_lu = 0 ;
    end  
    if isfield( adv_opts, 'use_lu_w_amd' )
        use_lu_w_amd = 1 ;
    else
        use_lu_w_amd = 0 ;
    end
    if isfield( adv_opts, 'newton_update')
        newton_update = adv_opts.newton_update ;
    else
        newton_update = 0 ;
    end  
    if isfield( adv_opts, 'dpa_bordered')
        dpa_bordered = adv_opts.dpa_bordered ;
    else
        if use_lu == 0
            dpa_bordered = 1 ;
        else
            dpa_bordered = 0 ;
        end
    end
    if use_lu == 0 && dpa_bordered == 0
        fprintf( 'Warning: using \\ for solves and *no* dpa_bordered\n' ) ;
        fprintf( '         might lead to unstable iterations/stagnation\n' ) ;
    end
end

if nargin < 9
    displ = 1 ;
end

if nargin < 8
    max_iter = 100 ;
end

if nargin < 7
    tol = 1e-10 ;
end

if nargin < 6
    nwanted = 1 ;
end

if nargin < 5
    s0 = 1i ;
end

rand( 'state', 0 ) ;
n = size(A, 1) ;
nr_its = 0 ;
sk = s0(1) ;
res = 10. ;
b_orig = b ;
c_orig = c;
nfound = 0 ;
dom_poles = zeros(nwanted*2, 1) ;
residues = zeros(nwanted*2, 1) ;
X = zeros( n, nwanted*2 ) ;
Y = zeros( n, nwanted*2 ) ;
pole_idx = 1 ;
nr_solves = 0 ;

if displ
    disp('  ');
    disp( 'DPA with Turbo Deflation' ) ;
    disp(' iter            eigenvalue             norm (A*X - s*E*X)     abs(u(sk))     ');
    disp(' ----   ------------------------------  ------------------ ------------------');
    output = sprintf(' %2d     %14.7g %+14.7gi  %14.7g %14.7g',0,real(full(sk)),imag(full(sk)),1,1);
    disp( output);
end
  
dpa_shifts(1, 1) = sk ;
shift_cnt(1) = 2 ;
skm = sk ;

rhs = sparse(n+1,1) ; 
rhs(n+1,1) = 1 ;

%Main loop of DPA
while res > tol && nr_its < max_iter && nfound < nwanted
    nr_its = nr_its+1 ;

    sImA = sk * E - A ;
    
    if use_lu %(re)use LU factorization
        if use_lu_w_amd
            [L,U,p,q] = lu(sImA,'vector') ;
            %btemp = S\b ;
            x = L\b(p) ;
            x(q) = U\x ;
            
            ctemp = c ;
            y = U'\ctemp(q) ;
            y(p) = (L'\y) ;
            %y = S' \ y ;
            
        else
            [L,U] = lu(sImA) ;
            x = U \ ( L \ b ) ;
            y = L' \ ( U' \ c ) ;
        end    
    else %use efficient \ operator
        if dpa_bordered
            sImA = [sk * E - A, -b; c', 0 ] ;
            x = sImA  \ rhs ; x = x(1:n) ;
            y = sImA' \ rhs ; y = y(1:n) ;
        else
            x = sImA  \ b ;
            y = sImA' \ c ;
        end
    end

    nr_solves = nr_solves + 1 ;
    skm = sk ;

    if newton_update %classical Newton update as in original DPA
        u = c' * x  ;
        un = -u / (y'*(E* x)) ;
        sk = sk + un ;
    else %twosided Rayleigh Quotient
        sk = y'*(A*x) / (y'*(E*x)) ;
    end

    if ~isfinite( sk ) %strcmp( num2str(sk), 'NaN' )
        fprintf( 'Warning: NaN shift found, restart with perturbed shift\n') ;
        sk = skm + 1e-10 ;
        x = rand(n, 1) ;
        y = rand(n, 1) ;
    end

    dpa_shifts( nfound+1, shift_cnt(nfound+1) ) = sk ;
    shift_cnt( nfound+1 ) = shift_cnt(nfound+1) + 1 ;

    % (experimental) check to avoid Newton getting into period >2 cycle
    % typically occurs if all (dominant) poles are found
    un = sk - skm ;
    cyclecheck = abs( sk - conj( sk-un ) ) ;
    if cyclecheck < 1e-6 && abs(imag(sk)) > 1e-6
        fprintf( 'Possible period 2 fixed point: diff = %s...\n', num2str(cyclecheck) ) ;
        fprintf( '...injecting new shift\n' ) ;
        sk = rand * j ;
    end

    x = x / norm(x) ;
    res = norm (A*x - sk*E*x) ;
	  
    if displ
        output = sprintf(' %2d     %14.7g %+14.7gi  %14.7g %14.7g',nr_its,real(full(sk)),imag(full(sk)),full(res),abs(full(un)));
        disp( output);
    end

    if res < 1e-5 %1e-8
        %rqi speed up: use twosided RQI in final phase of convergence
        [lambda, x, y, nlu] = ...
             twosided_rqi(A, E, x, y, sk, tol, use_lu, use_lu_w_amd, res, dpa_bordered) ;
        nr_solves = nr_solves + nlu ;
        x = x / norm(x) ;
        res = norm (A*x - sk*E*x) ;
    end
	  
    if res < tol
        if displ
            fprintf( 'Pole #%d %s found with ||r||=%s\n', nfound+1, num2str(full(sk)), num2str(res) ) ;
        end
        nfound = nfound + 1 ;
        
        if abs(imag(sk)) <= 1e-8 %found pole is real (probably...)
            x = real(x) ;
            y = real(y) ;
            sk = real(sk) ;
        end
        dom_poles( pole_idx, 1 ) = sk ;
        nyex = ( y'*(E*x) ) ;
        x = x / nyex ;
        X( :, pole_idx ) = x ;
        Y( :, pole_idx ) = y ;
        pole_idx = pole_idx + 1 ;
        %Turbo Deflation!
        b = b - E * (x * (y'*b)) ;
        c = c - E'* (y * (x'*c)) ;

        if abs(imag(sk)) > 1e-8 %deflate complex conjugate triplet as well
            conj_x = conj(x) ;
            conj_y = conj(y) ;
            dom_poles( pole_idx, 1 ) = conj(sk) ;
            X( :, pole_idx ) = conj_x ;
            Y( :, pole_idx ) = conj_y ;
            pole_idx = pole_idx + 1 ;
            b = b - E *(conj_x*(conj_y'*b)) ;
            c = c - E'*(conj_y*(conj_x'*c)) ;
        end            
        res = 1 ;
        if nfound < nwanted
            if length(s0) > nfound
                sk = s0(nfound+1) ;
            else
                sk = s0(end) ;
            end
            dpa_shifts( nfound+1, 1 ) = sk ;
            shift_cnt(nfound+1) = 2 ;   
        end
    end %end if res < tol
end %end while

%post processing
pole_idx = pole_idx - 1 ;
X = X(:, 1: pole_idx) ;
Y = Y(:, 1: pole_idx ) ;
dom_poles = dom_poles( 1 : pole_idx ) ;
residues = zeros( pole_idx, 1 ) ;
for i = 1 : size(X,2)
	residues(i,1) = (X(:,i).' * c_orig) * (Y(:,i)' * b_orig) ;
end

%sort on largest residual scaled by real part
[y,idx] = sort(-abs(residues) ./ abs(dom_poles)) ;
residues = residues(idx) ;
dom_poles = dom_poles(idx) ;
X = X(:, idx) ;
Y = Y(:, idx) ;

proc_admin.b_defl = b ;
proc_admin.c_defl = c ;
proc_admin.nr_solves = nr_solves ;
proc_admin.dpa_shifts = dpa_shifts ;
proc_admin.shift_cnt = shift_cnt ;

if displ
    fprintf( 'Dominant poles:\n' ) ;
    disp( dom_poles ) ;
end


%%%% other functions %%%%%
function [theta, schurvec, lschurvec, nr_solves] =  ...
    twosided_rqi(A, E, x, y, theta, tol, use_lu, use_lu_w_amd, r_orig, dpa_bordered)
%twosided Rayleigh quotient iteration
nr_solves = 0 ;
nrq = 1 ;
maxit = 10 ; %usually two iterations are enough
iter = 0 ;
imagtol = 1e-8 ;

n = size(A, 1) ;
rhs = sparse(n+1,1) ; 
rhs(n+1,1) = 1 ;

while nrq > tol && iter < maxit
    iter = iter + 1 ;
    Ex = E*x ;
    Ey = E'*y ;
%    sEmA = theta*E-A ;
    if use_lu
        if use_lu_w_amd
            [L,U,p,q] = lu(theta*E-A,'vector') ;
            %btemp = S\Ex ;
            x_rqi = L\Ex(p) ;
            x_rqi(q) = U\x_rqi ;
            
            ctemp = Ey ;
            v_rqi = U'\ctemp(q) ;
            v_rqi(p) = (L'\v_rqi) ;
            %v_rqi = S' \ v_rqi ;            
        else
            [L,U] = lu(theta*E-A) ;
            x_rqi = U \ (L \ (Ex)) ;
            v_rqi = L' \ (U' \ (Ey)) ;
        end    
    else
        if dpa_bordered
            sEmA = [theta * E - A, -Ex; Ey', 0 ] ;
            x_rqi = sEmA \ rhs ; x_rqi = x_rqi(1:n,1) ;
            v_rqi = sEmA' \ rhs ; v_rqi = v_rqi(1:n,1) ;
        else
            sEmA = theta*E-A ;
            x_rqi = (sEmA \ (Ex) ) ;
            v_rqi =  (sEmA' \ (Ey) ) ;
        end
    end

    nr_solves = nr_solves + 1 ;
    x_rqi = x_rqi / norm(x_rqi) ; 
    v_rqi = v_rqi / norm(v_rqi) ; 
    Ax_rqi = A*x_rqi ;
    Ex_rqi = E*x_rqi ;
    x_rq = (v_rqi'*Ax_rqi) / (v_rqi' * Ex_rqi) ;
    
    if ~isfinite( x_rq )
        %it may happen that x_rq is Inf or Nan, if exact pole is used
        %in that case, use previous data and perturb theta
        x_rqi = x ;
        v_rqi = y ;
        x_rq  = theta + 1e-10 ;
    end
    
    v_rq = x_rq' ;
    rqi_res = Ax_rqi - x_rq * (Ex_rqi) ;
    if abs(imag(x_rq)) / abs( x_rq ) < imagtol
        %check for possible real eigenpair
        rx_rqi = real(x_rqi) ;
        
        rx_rqi = rx_rqi / norm(rx_rqi) ;
        rres = A*rx_rqi - real(x_rq)*E*rx_rqi ;
        nrr = norm( rres) 
        if nrr < norm(rqi_res)
            x_rqi = rx_rqi ;
            v_rqi = real(v_rqi ) ; 
            v_rqi = v_rqi / norm(v_rqi) ;
            x_rq = real(x_rq ) ;
            rqi_res = rres ;
        end
    end    
    x = x_rqi ;
    y = v_rqi ;
    theta = x_rq ;
    nrq = norm(rqi_res) ;
    if ~isfinite( nrq )
        nrq = 1 ;
    end
end

if nrq < norm( r_orig )
%fprintf( 'RQI estimation used\n' ) ;
    schurvec = x_rqi ;
    lschurvec = v_rqi ;
    theta = x_rq ;
    r = rqi_res ;
    nr = nrq ;
else
    schurvec = x ;
    lschurvec = y ;
    r = r_orig ;
    nr = norm(r) ;
end

