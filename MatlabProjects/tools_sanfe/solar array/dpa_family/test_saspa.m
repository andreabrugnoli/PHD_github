function [] = test_saspa()
% Testscript for running Subspace Accelerated Sensitive Pole Algorithm
%
% Set make_movie = 1 to make movie
%
% Joost Rommes (C) 2008--2010
% rommes@gmail.com
% http://sites.google.com/site/rommes

make_movie = 0 ;

%Xingo KPSS + Paulo Afonso + Itaipu
	
    %%SASPA settings
    s0 = 1i ; %-2 + 6i ; %1i ;
	saspa_opt.nwanted = 10 ;
	saspa_opt.kmin = 2 ;
	saspa_opt.kmax = 10 ;
    saspa_opt.use_lu = 1 ;
    saspa_opt.use_lu_w_amd = 1 ;
%    saspa_opt.strategy = 'LR' ;

    	
    %% Setup all data, directional derivatives, etc
    load xingo_afonso_itaipu.mat ;
    n = size(A,1) ;
	
	multiparam = 1 ;
    dA = sparse(n, n) ;
	idx_r = 4971 ; %Xingo location; range (0,15)
	idx_c = 4970 ;
	if multiparam
		idx_r_PA4 = 4901 ; %Paulo Afonso 4 location; range (0,15)
		idx_c_PA4 = 4900 ;
		idx_r_Itaipu = 4473 ; %Itaipu location 1; range (0,2.2)
		idx_c_Itaipu = 4472 ; 
		idx_r2_Itaipu = 4478 ; %Itaipu location 2; range (0,10.35)
		idx_c2_Itaipu = 4477 ;
	end
	
	dA( idx_r, idx_c ) = 1 ;
	b_fixed = sparse(n,1) ; b_fixed(idx_r,1) = 1 ;
	c_fixed = sparse(n,1) ; c_fixed(idx_c,1) = 1 ;
	if multiparam
		b_fixed_PA4 = sparse(n,1) ; b_fixed_PA4(idx_r_PA4,1) = 1 ;
		c_fixed_PA4 = sparse(n,1) ; c_fixed_PA4(idx_c_PA4,1) = 1 ;
		b_fixed_Itaipu = sparse(n,1) ; b_fixed_Itaipu(idx_r_Itaipu,1) = 1 ;
		c_fixed_Itaipu = sparse(n,1) ; c_fixed_Itaipu(idx_c_Itaipu,1) = 1 ;
		b_fixed_Itaipu2 = sparse(n,1) ; b_fixed_Itaipu2(idx_r2_Itaipu,1) = 1 ;
		c_fixed_Itaipu2 = sparse(n,1) ; c_fixed_Itaipu2(idx_c2_Itaipu,1) = 1 ;
	
		%directional derivative needed
		direction = [(15-0)/30, (15-0)/30, (2.2-0)/30, (10.35-0)/30] ;
		direction = direction / norm(direction ) ;
		b_fixed = b_fixed + b_fixed_PA4 + b_fixed_Itaipu + b_fixed_Itaipu2 ;
		c_fixed = [c_fixed, c_fixed_PA4, c_fixed_Itaipu, c_fixed_Itaipu2] * direction' ;

		dA( idx_r, idx_c ) = direction(1) ;
		dA( idx_r_PA4, idx_c_PA4 ) = direction(2) ;
		dA( idx_r_Itaipu, idx_c_Itaipu) = direction(3) ;
		dA( idx_r2_Itaipu, idx_c2_Itaipu) = direction(4) ;		
	end
			
	minval = 0 ; %Xingo, PA4
	maxval = 15 ;
	nsteps = 30 ;
	facs = linspace( minval, maxval, nsteps ) ;

	minval = 0 ; %Itaipu 1
	maxval = 2.2 ;
	nsteps = 30 ;
	facs_itaipu1 = linspace( minval, maxval, nsteps ) ;

	minval = 0 ; %Itaipu 2
	maxval = 10.35 ;
	nsteps = 30 ;
	facs_itaipu2 = linspace( minval, maxval, nsteps ) ;

	A( idx_r, idx_c ) = facs(1) ;
	A( idx_r_PA4, idx_c_PA4 ) = facs(1) ;
	A( idx_r_Itaipu, idx_c_Itaipu ) = facs_itaipu1(1) ;
	A( idx_r2_Itaipu, idx_c2_Itaipu ) = facs_itaipu2(1) ;
	
%	tic ;
	%SASPA run for first parameter value
    tic;
    [poles, X, Y, derivs, nrSolves] = saspa(A, E, dA, s0, saspa_opt) ;	
    toc;
    return
%	poles, derivs

%start movie procedure
figh = figure ;
movfig2d = figh ;
xlabel( 'Real' ) ;
ylabel( 'Imag' ) ;
hold on ;
lhdp = plot( real(poles), imag(poles), 'ro', 'LineWidth', 2, 'MarkerSize', 14 ) ;
lhstart = plot( real( s0), imag(s0), 'ms', 'LineWidth', 3,'MarkerSize', 14 ) ;
axis( [-3.5,0.5,0,14] ) ;
axis manual ;
framecnt = 1 ;
handles = [] ;
legend( 'bla', 'bka' ) ;
[legh,objh,lhdp_s, outm] = legend ;
lhp_s = lhdp_s(2) ;
legend( [lhstart lhdp],  'Initial shift', 'New poles (SASPA)', 'Location', 'West' ) ;

tt = sprintf( 'Xingo/PA4 kpss = %s', num2str(facs(1)) ) ;
lhtext = text( -3.0, 13.5, tt ) ;
tt = sprintf( 'Itaipu I kpss = %s', num2str(facs_itaipu1(1)) ) ;
lhtext2 = text( -3.0, 13, tt ) ;
tt = sprintf( 'Itaipu II kpss = %s', num2str(facs_itaipu2(1)) ) ;
lhtext3 = text( -3.0, 12.5, tt ) ;
pause(0.5)        
if make_movie
    pause(0.5)
    F = getframe(movfig2d);
        %mov = addframe(mov,F);
    saspa_2d(framecnt) = F ;
    framecnt = framecnt + 1 ;
end

%loop over parameter values and recompute sensitive poles
total_solves = nrSolves ;
for k = 2:nsteps
	A(idx_r,idx_c) = facs(k) ; %set KPSS gains
	A( idx_r_PA4, idx_c_PA4 ) = facs(k) ;
	A( idx_r_Itaipu, idx_c_Itaipu ) = facs_itaipu1(k) ;
	A( idx_r2_Itaipu, idx_c2_Itaipu ) = facs_itaipu2(k) ;
	hold on ;
	[poles, X, Y, derivs, nrSolves] = saspa(A, E, dA, s0, saspa_opt) ;
    total_solves = total_solves + nrSolves ;
    
    set( lhtext, 'Visible', 'off' ) ;
    tt = sprintf( 'Xingo/PA4 kpss = %s', num2str(facs(k)) ) ;
    lhtext = text( -3.0, 13.5, tt ) ;
    set( lhtext2, 'Visible', 'off' ) ;
    tt = sprintf( 'Itaipu I kpss = %s', num2str(facs_itaipu1(k)) ) ;
    lhtext2 = text( -3.0, 13, tt ) ;
    set( lhtext3, 'Visible', 'off' ) ;
    tt = sprintf( 'Itaipu II kpss = %s', num2str(facs_itaipu2(k)) ) ;
    lhtext3 = text( -3.0, 12.5, tt ) ;
    
    %set previous color to blue
    set( lhdp, 'color', 'blue' ) ;
    lhdp_o = lhdp ;
    lhdp = plot( real(poles), imag(poles), 'ro', 'LineWidth', 2, 'MarkerSize', 14 ) ;
	%opt.sigma = lambda ; %tracking	
    leg = legend( [lhstart lhdp_o lhdp],  'Initial shift',  'Found poles (SASPA)', 'New poles (SASPA)', 'Location', 'West' ) ;

    pause(0.5)
    if make_movie        
        F = getframe(movfig2d);
        %mov = addframe(mov,F);
        saspa_2d(framecnt) = F ;
        framecnt = framecnt + 1 ;
    end
end

if make_movie
    movie2avi(saspa_2d, 'saspa_m2_2i_2d.avi', 'fps', 1) ;
end
