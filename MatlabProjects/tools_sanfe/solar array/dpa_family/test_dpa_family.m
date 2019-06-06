function [] = test_dpa_family()
% Testscript for running several algorithms from the DPA family
%
%
% Joost Rommes (C) 2006 -- 2010
% rommes@gmail.com
% http://sites.google.com/site/rommes
%
%
%

%clear all ;

disp('******TEST-SYSTEM DATA******');
disp(' ')
disp('testcase  1 => ww36 (SISO, n=66)')
disp('testcase  2 => cdplayer from Niconet (SISO, n=120)')
%                    %http://www.icm.tu-bs.de/NICONET/
disp('testcase  3 => ww_vref_6405 (SISO, n=13251)')
disp('testcase  4 => BIPS 07 xingo_afonso_itaipu (SISO, n=13250)')
disp('testcase  5 => cdplayer from niconet (2x2 MIMO, n=120)')
%                    %http://www.icm.tu-bs.de/NICONET/
disp('testcase  6 => BIPS 97 model (8x8 MIMO, n=13309)')
disp('testcase  7 => BIPS 97 model (28x28 MIMO, n=13251)')
disp('testcase  8 => Butterfly Gyro from Oberwolfach Benchmarks (2nd order, n=17361, do not use lu)')
%                    %http://www.imtek.de/simulation/index.php?page=http://www.imtek.de/simulation/benchmark/
disp('testcase  9 => Shaft from Zhaojun Bai (2nd order, n=400)')
%                    %http://www.cs.ucdavis.edu/~bai/
disp('testcase 10 => BIPS 97 model (46x46 MIMO, n=13250)')

testcase = input('Use testcase =  ') ;
[A, b, c, d, E, M, bodeopt] = call_system_data(testcase) ;

%-----------------------
% Method
disp('******Method******');
disp(' ')
disp('algorithm 1 =>    DPA : (SISO) with turbo deflation')
disp('algorithm 2 =>  SADPA : (SISO) Subspace Accelerated DPA')
disp('algorithm 3 =>  SAMDP : (MIMO) Subspace Accelerated MIMO DPA')
disp('algorithm 4 =>   QDPA : (SISO) Quadratic DPA for second-order systems')
disp('algorithm 5 => SAQDPA : (SISO) Subspace Accelerated QDPA for second-order systems')
disp('algorithm 6 =>    DZA : (SISO) Dominant Zero Algorithm via DPA')
disp('algorithm 7 =>  SADZA : (SISO) Dominant Zero Algorithm via SADPA')
disp('algorithm 8 =>  SAMDZ : (MIMO) Dominant Zero Algorithm via SAMDP')
disp('algorithm 9 =>  SASPA : (MIMO) SA Sensitive Pole Algorithm')
disp('algorithm 10=>  SARQI : (----) SA Rayleigh Quotient Iteration (Rightmost and damping ratio)')
disp(' ')
disp('Note: all methods use *default* settings. Please refer to') ;
disp('      inline comments in this file and function headers for details');
disp('      on how to tune parameters') ;

testfunction = input('Use method =  ');


if testfunction == 1 %dpa_tdefl
    
    if size(b,2) > 1 || size(c,2) > 1 || ~isscalar(M)
        fprintf( 'DPA_TDEFL operates on first-order SISO systems\n') ;
        return ;
    end
    nwanted = 5 ;
    s0 = (1:nwanted)*i ;
    tol = 1e-10 ; %optional
    max_iter = 100 ; %optional
    displ = 1 ; %optional
    adv_opts.use_lu = 1 ; %optional 
    %adv_opts.dpa_bordered = 0 ; %optional
    adv_opts.newton_update = 0 ; %optional
    adv_opts.use_lu_w_amd = 1 ;
    [dom_poles, X, Y, residues, proc_admin] = ...
          dpa_tdefl( A, E, b, c, s0, nwanted, tol, max_iter, displ, adv_opts) ;
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dom_poles, X, Y, bodeopt ) ;
end

if testfunction == 2 %sadpa
    if size(b,2) > 1 || size(c,2) > 1 || ~isscalar(M)
        fprintf( 'SADPA operates on first-order SISO systems\n') ;
        return ;
    end
    opt.kmin = 5 ; %min searchspace
    opt.kmax = 10 ; %max searchspace
    opt.nwanted = 10 ; %number of poles wanted
    opt.maxrestarts = 10 ; %maximum number of restarts
    opt.displ = 1 ; %set to zero to disable verbose output
    opt.strategy = 'LR' ; %select approximation with largest |residue|/|real(p)|
    opt.turbo_deflation = 1 ; %deflation via b and c
    opt.use_lu = 1 ;
    opt.use_lu_w_amd = 1 ;

    s0 = 1i ;
%    s0 = (1:opt.nwanted)*i ; %inject new shift after each found pole
%
%   %Explanation: inject equally spaced new shifts, and disregard automatically generated 
%   %shifts by the methods of the SADPA family (only for first iteration after
%   %having found a pole). Users are encouraged to try out combinations of 
%   %injections and automatic generation of shifts, to obtain better results 
%   %for their specific applications.    
    
    opt.dpa_bordered = 0 ;
    [dompoles_sadpa, residues, X, Y, nr_solves, ress] = ...
        sadpa(A, E, b, c, d, s0, opt) ;
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dompoles_sadpa, X, Y, bodeopt ) ;

end

if testfunction == 3 %samdp
    if  ~isscalar(M)
        fprintf( 'SAMDP operates on first-order MIMO systems\n') ;
        return ;
    end
    opt.kmin = 5 ; %min searchspace
    opt.kmax = 10 ; %max searchspace
    opt.nwanted = 20 ; %number of poles wanted
    opt.maxrestarts = 30 ; %maximum number of restarts
    opt.displ = 1 ; %set to zero to disable verbose output
    opt.strategy = 'LM' ; %select approximation with largest |residue|
    opt.turbo_deflation = 1 ; %deflation via b and c
    opt.use_lu = 1 ;
    opt.use_lu_w_amd = 1 ;
    opt.use_fixed_svd = 0 ; %if 0, compute svd(H(s_k)) every iteration
                            %      (more expensive, better convergence)
                            %else only every restart
    %opt.dpa_bordered = 0 ;

    s0 = 1i ;
%    s0 = (1:0.5:opt.nwanted)*i ; %inject new shift after each found pole
    [dompoles, residues, X, Y, nr_solves] = ...
        samdp(A, E, b, c, d, s0, opt) ;
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dompoles, X, Y, bodeopt ) ;

end

if testfunction == 4 %qdpa
    if  size(b,2) > 1 || size(c,2) > 1 || isscalar(M)
        fprintf( 'QDPA operates on second-order SISO systems\n') ;
        return ;
    end
    s0 = 1i ;
    tol = 1e-6 ; %optional
    max_iter = 100 ; %optional
    displ = 1 ; %optional
    use_lu = 0 ; %optional 
    use_lu_w_amd = 0 ;

    [dom_pole, X, Y] = ...
          qdpa( A, E, M, b, c, s0, tol, max_iter, displ, use_lu, use_lu_w_amd) ;

end

if testfunction == 5 %saqdpa
    if  size(b,2) > 1 || size(c,2) > 1 || isscalar(M)
        fprintf( 'SAQDPA operates on second-order SISO systems\n') ;
        return ;
    end
    sa_opt = struct('strategy', 'LR', ...
                    'kmin', 4, ...
                    'kmax', 10, ...
                    'nwanted', 5,  ...
                    'tol', 1e-6, ...
                    'use_lu', 0,...
                    'displ', 1, ...
                    'turbo_deflation', 1,...
                    'rqi_tol', 5e-3 ) ; 
    sa_opt.use_lu_w_amd = 1 ;
%    s0 = [1i;2i] ;
    s0 = 1.5e5i * 2 * pi ;
%    s0 = (1:opt.nwanted)*i ; %inject new shift after each found pole

    [poles, residuals, X, Y, nr_solves] = ...
        saqdpa(A, E, M, b, c, d, s0, sa_opt) ;
end

if testfunction == 6 %dza_tdefl
    if  size(b,2) > 1 || size(c,2) > 1 || ~isscalar(M)
        fprintf( 'DZA_TDEFL operates on first-order SISO systems\n') ;
        return ;
    end
    nwanted = 3 ; %converges to infinity after three zeros...use sadza
    s0 = 1i ;
    tol = 1e-10 ; %optional
    max_iter = 100 ; %optional
    displ = 1 ; %optional
    adv_opts.use_lu = 1 ; %optional 
    %adv_opts.dpa_bordered = 0 ; %optional
    adv_opts.newton_update = 0 ; %optional
    [dom_zeros, X, Y, residues, proc_admin] = ...
          dza_tdefl( A, E, b, c, d, s0, nwanted, tol, max_iter, displ, adv_opts) ;
    
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dom_zeros, [], [], bodeopt ) ;

end

if testfunction == 7 %sadza
    if  size(b,2) > 1 || size(c,2) > 1 || ~isscalar(M)
        fprintf( 'SADZA operates on first-order SISO systems\n') ;
        return ;
    end
    opt.kmin = 5 ; %min searchspace
    opt.kmax = 10 ; %max searchspace
    opt.nwanted = 10 ; %number of poles wanted
    opt.maxrestarts = 10 ; %maximum number of restarts
    opt.displ = 1 ; %set to zero to disable verbose output
    opt.strategy = 'LR' ; %choose 'LR' or 'LS': prevents convergence to inf
                          %'LM' has high risk of convergence to inf
    opt.turbo_deflation = 1 ; %deflation via b and c
    opt.use_lu = 1 ;

    s0 = 1i ;
%    s0 = (1:opt.nwanted)*i ; %inject new shift after each found pole
    opt.dpa_bordered = 0 ;
    [dom_zeros, residues, X, Y, nr_solves, ress] = ...
        sadza(A, E, b, c, d, s0, opt) ;
    
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dom_zeros, [], [], bodeopt ) ;
end

if testfunction == 8 %samdz
    if   ~isscalar(M)
        fprintf( 'SAMDZ operates on first-order MIMO systems\n') ;
        return ;
    end
    opt.kmin = 5 ; %min searchspace
    opt.kmax = 10 ; %max searchspace
    opt.nwanted = 10 ; %number of poles wanted
    opt.maxrestarts = 20 ; %maximum number of restarts
    opt.displ = 1 ; %set to zero to disable verbose output
    opt.strategy = 'LR' ;%choose 'LR' or 'LS': prevents convergence to inf
                          %'LM' has high risk of convergence to inf
    opt.turbo_deflation = 1 ; %deflation via b and c
    opt.use_lu = 1 ;
    opt.use_fixed_svd = 0 ;
    %opt.dpa_bordered = 0 ;

    s0 = 1i ;
%    s0 = (1:opt.nwanted)*i ; %inject new shift after each found pole

    [dom_zeros, residues, X, Y, nr_solves] = ...
        samdz(A, E, b, c, d, s0, opt) ;
    
    fprintf( 'Computing Bodeplots...\n' ) ;
    bodeplot( A, E, b, c, d, dom_zeros, [], [], bodeopt ) ;
end

if testfunction == 9 %SASPA
    disp( ' ' ) ;
    disp( 'Note: SASPA operates on xingo_afonso_itaipu.mat') ;
    disp( ' ' ) ;
    test_saspa() ;
end

if testfunction == 10 %SARQI
    opt.kmin = 5 ; %min searchspace
    opt.kmax = 10 ; %max searchspace
    opt.nwanted = 10 ; %25 ; %number of poles wanted
    opt.maxrestarts = 100 ; %maximum number of restarts
    opt.displ = 1 ; %set to zero to disable verbose output
    opt.turbo_deflation = 0 ; %deflation via b and c
    opt.dpa_bordered = 0 ;
    opt.use_lu = 1 ;

    s0 = 1i ; %1.013599476501230e+00 + 8.083197270071082e+00i ; %1i ;
    
    opt.strategy = 'DR' ;
    opt.damping_ratio = -1 ;

    [rightmostpoles_sarqi,  X, Y, nr_solves, ress] = ...
        sarqi(A, E, b(:,1), c(:,1), d, s0, opt) ;

end

function [A, b, c, d, E, M, bodeopt] = call_system_data( test_case )
%first order systems: (sE-A)x = bu ; y = c'x
M = 0 ; %only for second order systems (A + sE + s^2M)x = bu; y = c'x

%default settings for plotting bodeplots
bodeopt.lowf = 0.1 ;
bodeopt.hif  = 20 ;
bodeopt.title = '' ;
bodeopt.npts = 100 ;

switch test_case
    case 1
        load ww_36_pmec_36.mat ;
        c = c' ;
        d = 0 ;
        E = sparse(eye( size(A) )) ;
        A = sparse(A) ;

    case 2
        load CDplayer.mat ;
        b = B(:, 2) ;
        c = C(1, :)' ;
        n = size(A,1) ;
        E = speye(n,n) ;
        d=0;
        bodeopt.loglogok = 1 ;
        bodeopt.hif = 1e6 ;
    
    case 3 
        load ww_vref_6405.mat ;
        
    case 4
        load xingo_afonso_itaipu.mat ;
        c = C ;
        b = B ;
        d = D ;
    case 5
        load CDplayer.mat ;
        b = B ;
        c = C' ;
        n = size(A,1) ;
        E = speye(n,n) ;
        d=sparse(size(c,2), size(B,2)) ;
        bodeopt.loglogok = 1 ;
        bodeopt.hif = 1e6 ;
    case 6
        load mimo8x8_system.mat ;
    case 7
        load mimo28x28_system.mat ;
    case 8
        load gyro.mat ;
        alpha = 0 ;
        beta_c = 1e-7 ; %personal communication with Liendemann
        E = alpha*M + beta_c*K ;
        A = K ;
        M = M ;
        c = C(:, 1) ;
        b = B ;
        d = 0 ;
    case 9
        load y6103f.mat ;
        A = sparse(K) ;
        E = sparse(D) ;
        M = sparse(M) ;
        b = sparse(size(A,1), 1) ; b(1) = 1 ;
        c = b ;
        d = 0 ;
    case 10
        load mimo46x46_system.mat ;
        b = B ;
        c = C ;
        d = D ;
end 

function figh = bodeplot(A, E, b, c, d, poles, X, Y, bodeopt)
%plot original TF, modal equivalent, and dominant poles

ismimo = size(b,2) > 1 ;

%original TF
[H,w,fighandle] = plot_bodeplot(A,b,c,d,E, bodeopt) ;

bodeopt.handle = fighandle ;
hold on ;
if size(X,1) > 0
    As = Y'*A*X ;
    Es = Y'*E*X ;
    bs = Y'*b ;
    cs = X'*c ;

    %reduced TF
    [H,w,figh] = plot_bodeplot(As,bs,cs,d,Es, bodeopt) ;
    %set( lineh, 'color', 'r' ) ;
end

%plot the poles in bodeplot
nselect = 1 ;
while nselect <= length(poles)
    ww = abs(imag(poles(nselect))) ;
    if ww < 0.01
        ww = 0.01 ;
    end
    plot( ww, 20*log10(norm(full(c'*( (ww*sqrt(-1)*E-A) \ b)+d))), 'ko', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r' ) ;
    if imag( poles(nselect) ) == 0
        nselect = nselect + 1 ;
    else
        nselect = nselect + 2 ;
    end
end

if ismimo
    if size(X,1) > 0
		legend( 'Original \sigma_{max}', 'Original \sigma_{min}', ...
                'Modal Equiv. \sigma_{max}', 'Modal Equiv. \sigma_{min}',...
                'Dominant Poles' ) ;
    else
		legend( 'Original \sigma_{max}', 'Original \sigma_{min}', ...
                'Dominant Zeros' ) ;
    end
else
    if size(X,1) > 0
        legend('Original', 'Modal Equiv.', 'Dominant Poles')
    else
        legend('Original',  'Dominant Zeros')
    end
end
