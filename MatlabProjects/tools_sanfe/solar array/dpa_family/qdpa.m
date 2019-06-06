function [pole, X, Y] = qdpa( A, E, M, b, c, s0, tol, max_iter, displ, use_lu, use_lu_w_amd)
% Dominant Pole Algorithm for Quadratic transfer functions:
%        H(s) = c' * ( s^2M + sE + A ) ^{-1}b
%
% Usage : [pole, X, Y] = qdpa( A, E, M, b, c, s0, tol)
%
% Input:
%        A, E, M   : nxn system matrices
%        b,c       : nx1 input and output map vectors
%        s0        : initial shift (1i)
%        tol       : convergence tolerance ||Ax + sEx + s^2Mx||_2 (1e-10)
%        max_iter  : maximum number of iterations (100)
%        displ     : display QDPA process output (1)
%        use_lu    : if use_lu == 1 use LU factorization (default)
%                    else use \
%        use_lu_w_amd: if use_lu and use_lu_w_amd, use AMD
%
% When using this software, please cite
%
%   Joost Rommes and Nelson Martins
%   Efficient computation of transfer function dominant poles of 
%   large second-order dynamical systems
%   SIAM Journal on Scientific Computing 30 (4), 2008, pp. 2137-2157
%
% Note that this implementation does not contain any convergence improving
% aids and hence convergence may be slow or stagnate. Use saqdpa instead for
% a much more robust implementation.
%
% Joost Rommes (C) 2006 -- 2008
% rommes@gmail.com
% http://sites.google.com/site/rommes

if nargin < 10
    use_lu = 1 ;
end

if nargin < 9
    displ = 1 ;
end

if nargin < 8
    max_iter = 100 ;
end

if nargin < 7
    tol = 1e-6 ;
end

if nargin < 6
    s0 = 1i ;
end

n = size(A, 1) ;
nr_its = 0 ;
sk = s0 ;
res = 10.;

I = speye( n ) ;

if displ
    disp(' iter            eigenvalue             |(A + s*E + s^2M)X|  abs(u(sk))     ');
    disp(' ----   ------------------------------  ------------------  ------------------');
    output = sprintf(' %2d     %14.7g %+14.7gi  %14.7g %14.7g',0,real(full(sk)),imag(full(sk)),1,1);
    disp( output);
end

rhs = sparse( n + 1, 1 ) ;
rhs( end ) = 1 ;
rhs0 = sparse( n, 1 ) ;
rhs0( end ) = 1 ;

ests(1) = sk ;

while res > tol && nr_its < max_iter
    nr_its = nr_its+1 ;

    if ~isfinite( sk )
        fprintf( 'Warning: NaN shift found, restart with perturbed shift\n') ;
        sk = skm + 1e-10 ;
    end

    sImA = sk^2 * M + sk * E + A ;

    if use_lu
        [x,v] = solve_with_lu( sEmA, b, c, use_lu_w_amd ) ;
        %[L,U] = lu(sImA) ;
        %x = U \ ( L \ b ) ;
        %v = L'\ ( U'\ c ) ;
    else
        x = sImA  \ b ;
        v = sImA' \ c ;
    end
    
    u = c' * x;
    un = -u / (v'*((2*sk*(M*x) + E*x)));

%      un = -u / (v'*((2*sk*M + E)*x));
%although mathematically the same, in practice there may be great diffs!
    
    skm = sk ;
    sk = sk + un ;
    ests(nr_its+1) = sk ;

    x = x / norm(x) ;
    res = norm (A*x + sk*(E*x) + sk^2*(M*x)) ;
    
    if displ
        output = sprintf(' %2d     %14.7g %+14.7gi  %14.7g %14.7g',nr_its,real(full(sk)),imag(full(sk)),full(res),abs(full(un)));
        disp( output);
    end
end

pole = sk ;
X = x/ norm(x) ;
Y = v/norm(v) ;

function [x,y] = solve_with_lu( sEmA, b, c, use_amd )
%solves sEmA*x=b and sEmA'*y = c
        if use_amd
            [L,U,p,q] = lu(sEmA,'vector') ;
            %b = S\b ;
            x = L\b(p,:) ;
            x(q,:) = U\x ;
            
            if nargout > 1
                y = U'\c(q,:) ;
                y(p,:) = (L'\y) ;
                %y = S' \ y ;
            end
        else
            [L,U] = lu(sEmA) ;
            x = U \ ( L \ b ) ;
            if nargout > 1
                y = L' \ ( U' \ c ) ;
            end
        end    
