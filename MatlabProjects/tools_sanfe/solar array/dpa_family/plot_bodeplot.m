function [H,w, handle] = plot_bodeplot(A, b, c, d, E, options)
%[H,w] = plot_bodeplot(A, b, c, d, E, options) plot the transfer function
%of the system Edx/dt = Ax + bu ; y = c'x + d
%
% Note: Computation is done via a \ of A for every s
%       This may be expensive; use function for reference and small sys
%       only.
% Input:
%     A,b,c,d, E  : system
%     options: structure with
%        lowf: lower frequency (1 rad/s)
%        hif : upper frequency (10^10 rad/s)
%        npts: nr of points for H (1001)
%        w : predefined frequency vector
%        s0  : frequency shift
%        title: optional title of figure
%        displ : show the bode-plot
%        loglog: use loglog instead of dB scale
% Output:
%     H     : transferfunction values
%     w     : corresponding frequencies (rad/s)
%     handle: figure handle
%
% Joost Rommes (C) 2004--2008
% rommes@gmail.com
% http://sites.google.com/site/rommes

%opt = struct( 'lowf', 0.1, 'hif', 20, 'npts', 100, 'loglogok', 0, 'radpers',1 ) ;
%opt = struct( 'lowf', 0.01, 'hif', 10^9, 'npts', 200, 'loglogok', 0, 'radpers',1 ) ;

if nargin < 6
    options = struct([]) ;
end

fields = fieldnames( options ) ;

if strmatch( 'lowf', fields )
    lowf = options.lowf ;
else
    lowf = 1 ;
end

if strmatch( 'hif', fields )
    hif = options.hif ;
else
    hif = 10^10 ;
end

if strmatch( 'npts', fields )
    npts = options.npts ;
else
    npts = 100 ;
end

if isfield( options, 'w' )
    freq = options.w ;
else
    freq = [] ;
end

if strmatch( 's0', fields )
    sigma = options.s0 ;
else
    sigma = 0 ;
end

if strmatch( 'title', fields )
    titlestr = options.title ;
else
    titlestr = 'Bodeplot' ;
end

if isfield( options, 'loglogok' )
    loglogok = options.loglogok ;
else
    loglogok = 0 ;
end

if isfield( options, 'radpers' )
    radpers = options.radpers ;
else
    radpers = 1 ;
end

fighandle_avail = 0 ;
if isfield( options, 'handle' )
	fighandle = options.handle ;
	fighandle_avail = 1 ;
end
if ~radpers
	%use Hz
	hif = 2*pi*hif ;
	lowf = 2*pi*lowf ;
end

if ~loglogok
    if isempty(freq)
        freq = linspace( lowf, hif, npts ) ;
    end
    s = sqrt(-1)*freq; %freq is expected to be in rad/s!
else
    if isempty(freq) 
        freq = logspace( log10(lowf), log10(hif), npts ) ;
    end
    s = sqrt(-1) * freq ;
end

if ~radpers
	freq = s / (2*pi) ;
end

ismimo = size(b, 2) > 1 ;

nfreq = length( s ) ;
H = zeros(1,nfreq) ;
Hmin = H ;
for f = 1:nfreq
    if ismimo    
        HH = full( c' * ( (s(f)*E - A) \ b) + d ) ;
        singvals = svd( HH ) ;
        H(f) = max(singvals) ;
        Hmin(f) = min(singvals) ;
    else    
        H(f) = norm( full( c' * ( (s(f)*E - A) \ b) + d ) ) ;
    end
end
w = abs(s) ;
handle = 0 ;

if fighandle_avail
	handle = figure( fighandle ) ;
	hold on ;
else
	handle = figure ;
end

if loglogok
    if fighandle_avail
        lh = semilogx(abs(freq), 20*log10(abs(H)), 'r-') ;
        if ismimo
		    lh1 = semilogx(abs(freq), 20*log10(abs(Hmin)), 'r--'); 
		    legend( 'Original \sigma_{max}', 'Original \sigma_{min}', ...
                    'Modal Equiv. \sigma_{max}', 'Modal Equiv. \sigma_{min}' ) ;
        else
		    legend( 'Original', 'Modal Equiv.' ) ;
        end
    else
		lh = semilogx(abs(freq), 20*log10(abs(H)), 'k-'); %original transferf
        hold on ;
        if ismimo
		    lh1 = semilogx(abs(freq), 20*log10(abs(Hmin)), 'k--'); %original transferf        
        end
	end
else
    if fighandle_avail
		lh = plot(abs(freq), 20*log10(abs(H)), 'r-'); %original transferf
        if ismimo
		    lh1 = plot(abs(freq), 20*log10(abs(Hmin)), 'r--'); %original transferf        
		    legend( 'Original \sigma_{max}', 'Original \sigma_{min}', ...
                    'Modal Equiv. \sigma_{max}', 'Modal Equiv. \sigma_{min}' ) ;
        else
		    legend( 'Original', 'Modal Equiv.' ) ;
        end
	else
		lh = plot(abs(freq), 20*log10(abs(H)), 'k-'); %original transferf
        hold on ;
        if ismimo
		    lh1 = plot(abs(freq), 20*log10(abs(Hmin)), 'k--'); %original transferf        
        end
	end
end
set( lh, 'LineWidth', 2 ) ;
if ismimo
    set( lh1, 'LineWidth', 2 ) ;
end

if radpers
	lx = xlabel('Frequency (rad/sec)');
else
	lx = xlabel('Frequency (Hz)');
end
set( lx, 'FontSize', 14 ) ;

ly =ylabel('Gain (dB)') ;
set( ly, 'FontSize', 14 ) ;

if ismimo
    title( 'Sigma Min/Max Plot' ) ;
else
    title( 'Bode Magnitude Plot' ) ;
end
