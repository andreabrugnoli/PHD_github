function [magout,phase,w] = black(a,b,c,d,iu,w)
%BLACK  Black-Nichols frequency response for single input systems
%       with frequency scale in rad/s and continuous phase.
%
% Syntaxes:
% BLACK(SYSIN), BLACK(SYSIN,IU) (for continuous or discrete LTI system
%       or for continuous system matrix (PCK)), or
% BLACK(A,B,C,D), BLACK(A,B,C,D,IU) (for continuous state-space 
%       representation) produce a Nichols plot from the single
%       input IU to all the outputs of the system SYSIN or continuous
%       state-space system (A,B,C,D).
%       IU is an index into the inputs of the system and 
%       specifies which input to use for the Nichols response.  The 
%       frequency range and number of points are chosen automatically.
%
%eradique:  BLACK(NUM,DEN) produces the Nichols plot for the polynomial 
%           transfer function G(s) = NUM(s)/DEN(s) where NUM and DEN contain
%           the polynomial coefficients in descending powers of s. 
%
%       BLACK(A,B,C,D,IU,W) or BLACK(SYSIN,IU,W) uses the user-supplied
%       freq. vector W which must contain the frequencies, in radians/sec,
%       at which the Nichols response is to be evaluated.  When invoked 
%       with left hand arguments,
%               [MAG,PHASE,W] = BLACK(A,B,C,D,...)
%               [MAG,PHASE,W] = BLACK(SYSIN,,...) 
%       returns the frequency vector W and matrices MAG and PHASE (in 
%       degrees) with as many columns as outputs and length(W) rows.  No 
%       plot is drawn on the screen.  Draw the Nichols grid with NGRID.
%
%       WARNING: Y-axis is scaled between -100 and 100 Db if ISHOLD=0
%
%       Used DBLACK if you want to specify the sampling period in case
%       of a discrete-time systeme.
%
%       See also: DBLACK,LOGSPACE,NGRID,MARGIN,BODE, and NYQUIST.

%#
%       D. Alazard 
%       Revised 9/07/97  12/98
%       Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

ni = nargin;
no = nargout;
if ni==0, eval('exresp(''black'')'), return, end

error(nargchk(1,6,ni));

% --- Determine which syntax is being used ---

if (ni==1), [a,b,c,d,flag,timer]=test_natss(a);
        if flag==0,error('la variable doit etre un SYSTEME'); end;
        if timer ~=0, 
            if no, [magout,phase,w] = dblack(a,b,c,d,timer);return;
            else dblack(a,b,c,d,timer);return;
            end;
        end;
        w = freqint2(a,b,c,d,30);
        [r,m]=size(d);
        if m>1, error('BLACK ne supporte que des systèmes SIMO');
        else iu=1;
        end;
        [ny,nu] = size(d);

elseif (ni==2), iu = b;
       [a,b,c,d,flag,timer]=test_natss(a);
	if flag==0, error('la variable doit etre un SYSTEME');	end;
        if timer ~=0, 
            if no, [magout,phase,w] = dblack(a,b,c,d,timer,iu);return;
            else dblack(a,b,c,d,timer,iu);return;
            end;
        end;
        w = freqint2(a,b,c,d,30);
	[ny,nu] = size(d); 

elseif (ni==3),	iu=b;w = c;
	[a,b,c,d,flag,timer]=test_natss(a);
	if flag==0, error('la variable doit etre un SYSTEME');	end;
        if timer ~=0, 
            if no, [magout,phase,w] = dblack(a,b,c,d,timer,iu,w);return;
            else dblack(a,b,c,d,timer,iu,w);return;
            end;
        end;
	[ny,nu] = size(d);

elseif (ni==4),	% State space system, w/o iu or frequency vector
	error(abcdchk(a,b,c,d));
	w = freqint2(a,b,c,d,30);
        [r,m]=size(d);
        if m>1, error('BLACK ne supporte que des systèmes SIMO');
        else iu=1;
        end
%	[iu,nargin,mag,phase]=mulresp('black',a,b,c,d,w,no,1);
%	if ~iu, if no, magout = mag; end, return, end
	[ny,nu] = size(d);

elseif (ni==5),	% State space system, with iu but w/o freq. vector
	error(abcdchk(a,b,c,d));
	w = freqint2(a,b,c,d,30);
	[ny,nu] = size(d);

else
	error(abcdchk(a,b,c,d));
	[ny,nu] = size(d);

end


if nu*ny==0, phase=[]; w=[]; if no~=0, magout=[]; end, return, end

% --- Evaluate the frequency response ---


	g = freqresp(a,b,c,d,iu,sqrt(-1)*w);
        [r,m]=size(d);


if no~=0,[magout,phase]=r2p(g);phase=fixphase(phase);end;
%phase = -180+180./pi*atan2(-imag(g),-real(g));
%phase = (180./pi)*unwrap(atan2(imag(g),real(g)));

% If no left hand arguments then plot graph.
if no==0
%
if r>1, disp('Attention: système SIMO !!');end;
  for kk=1:r,
        [mag,ph]=r2p(g(:,kk));
        ph=fixphase(ph);
        plot(ph,20.*log10(mag));grid,xlabel('PHASE'),ylabel('dB');
	if ~ishold
            yl=[min(mag) max(mag)];yl=20.*log10(yl);
 	    yl=[floor(yl(1)/10)*10 ceil(yl(2)/10)*10];
            if (yl(2)>100) yl(2)=100;end;
            if (yl(1)<-100) yl(1)=-100;end;
	    if (yl(2)-yl(1))<20,yl(1)=yl(2)-20;end;
            set(gca,'YLim',yl);
            xl=[min(ph) max(ph)];
 	    xl=[floor(xl(1)/180)*180 ceil(xl(2)/180)*180];
	    if (xl(2)-xl(1))<360,xl(1)=xl(2)-360;end;
	    set(gca,'XLim',xl);
%           set(gca,'YLim',[-100 100]);
        end;
	hold on
%	xl=get(gca,'XLim');
%	xl=[floor(xl(1)/180)*180 ceil(xl(2)/180)*180];
%	if (xl(2)-xl(1))<360,xl(1)=xl(2)-360;end;
%%	xl(1)=min([-360 xl(1)]);
%%	xl(2)=max([0 xl(2)]);
%	set(gca,'XLim',xl);
%  Graduation en x multiples de 60 degres
	vec=get(gca,'XTick');
	ntick=length(vec);
        xl=get(gca,'XLim');
        pas=round(xl(2)-xl(1))/(ntick-1);
        pasi=[10;30;45;60;90;120;180;360];
        [err,ii]=min(abs(pasi-pas));
        pas=pasi(ii);
%	pas=round((xl(2)-xl(1))/ntick/60)*60;
%	if pas>180,pas=180;end;
	vec=[xl(1):pas:xl(2)];
	set(gca,'XTick',vec);
%
%	nmax=floor(xl(2)/360);nmin=ceil(xl(1)/360);
	nmax=ceil(xl(2)/360);nmin=floor(xl(1)/360);
	[mc]=mcirc([0 2.3 10]);
	[m,p]=r2p(-mc);
        clm=get(gca,'ColorOrder');
        set(gca,'ColorOrder',[0.5 0.5 0.5]);
	for ii=nmin:nmax...
		plot(p+ii*360-180,20*log10(m));...
	end
        set(gca,'ColorOrder',clm);
%	mknic(g,w)
	plt=4;
	mag=20 .*log10(mag);
	mon_mkplot(ph,mag,w,plt);
	hold off
   end
end

% Internal function DBLACK.m
function [magout,phase,w] = dblack(a,b,c,d,Ts,iu,w)
%DBLACK Black-Nichols frequency response for discrete-time linear systems
%       with frequency scale in rad/s and continuous phase.
%
% Syntaxes:
% DBLACK(SYSIN,T), DBLACK(SYSIN,T,IU) (for discrete LTI system or system
%        matrix (PCK) with sampling period T), or
% DBLACK(A,B,C,D,T,IU) (for discret state-space representation)
%	produce a Nichols plot from the single 
%       input IU to all the outputs of the discrete system SYSIN or discrete
%       state-space system (A,B,C,D).  
%       IU is an index into the inputs of the system and 
%       specifies which input to use for the Nichols response.  T is the
%       sample period.  The frequency range and number of points are 
%       chosen automatically.
%
% eradique:DBLACK(NUM,DEN,T) produces the Nichols plot for the polynomial 
%       transfer function G(z) = NUM(z)/DEN(z) where NUM and DEN contain
%       the polynomial coefficients in descending powers of z. 
%
%       DBLACK(A,B,C,D,T,IU,W) or DBLACK(SYSIN,T,IU,W) uses the user-
%       supplied freq. vector W which must contain the frequencies, in 
%       rad/sec, at which the Nichols response is to be evaluated.  
%       Aliasing will occur at frequencies greater than the Nyquist 
%       frequency (pi/T). With left hand arguments,
%               [MAG,PHASE,W] = DBLACK(A,B,C,D,Ts,...)
%               [MAG,PHASE,W] = DBLACK(SYSIN,Ts,...) 
%       returns the frequency vector W and matrices MAG and PHASE (in 
%       degrees) with as many columns as outputs and length(W) rows.  No
%       plot is drawn on the screen.  Draw the Nichols grid with NGRID.
%
%       WARNING: Y-axis is scaled between -100 and 100 Db if ISHOLD=0
%
%       overloaded syntax: DBLACK(SYSIN) for discrete LTI system.
%
%       See also: BLACK,LOGSPACE,SEMILOGX,MARGIN,DBODE, and DNYQUIST.

%       M. Alazard 7-10-90
%       Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.
ni = nargin;
no = nargout;
if (ni==0), eval('dexresp(''dblack'')'), return, end

error(nargchk(1,7,ni));

% --- Determine which syntax is being used ---
if (ni==1), 
	[a,b,c,d,flag,Ts]=test_natss(a);
	if flag==0,error('la variable doit etre un SYSTEME');end;
	w = dfrqint2(a,b,c,d,Ts,30);
        [r,m]=size(d);
        if m>1, error('BLACK ne supporte que des systèmes SIMO');
        else iu=1;
        end;
	[ny,nu] = size(d);
	
elseif (ni==2),
	Ts=b;
	[a,b,c,d,flag]=test_natss(a);
	if flag==0,error('la variable doit etre un SYSTEME');
	end;
	w = dfrqint2(a,b,c,d,Ts,30);
        [r,m]=size(d);
        if m>1, error('BLACK ne supporte que des systèmes SIMO');
        else iu=1;
        end;
	[ny,nu] = size(d);

elseif (ni==3),	iu=c;Ts=b;	
	[a,b,c,d,flag]=test_natss(a);
	if flag==0,error('la variable doit etre un SYSTEME');end;
	w = dfrqint2(a,b,c,d,Ts,30);
        [ny,nu] = size(d);
	
elseif (ni==4), w=d;iu=c;Ts=b;	
	[a,b,c,d,flag]=test_natss(a);
	if flag==0,error('la variable doit etre un SYSTEME');end;
        [ny,nu] = size(d);	

elseif (ni==5),	% State space system without iu or freq. vector
	error(abcdchk(a,b,c,d));
	w=dfrqint2(a,b,c,d,Ts,30);
        [r,m]=size(d);
        if m>1, error('DBLACK ne supporte que des systèmes SIMO');
        else iu=1;
        end
%	[iu,nargin,mag,phase]=dmulresp('dblack',a,b,c,d,Ts,w,no,1);
%	if ~iu, if no, magout = mag; end, return, end
	[ny,nu] = size(d);

elseif (ni==6),	% State space system, with iu but w/o freq. vector
	error(abcdchk(a,b,c,d));
	w=dfrqint2(a,b,c,d,Ts,30);
	[ny,nu] = size(d);

else
	error(abcdchk(a,b,c,d));
	[ny,nu] = size(d);

end

if nu*ny==0, phase=[]; w=[]; if no~=0, magout=[]; end, return, end

% Compute frequency response
	g=freqresp(a,b,c,d,iu,exp(sqrt(-1)*w*Ts));
        [r,m]=size(d);

if no~=0,[magout,phase]=r2p(g);phase=fixphase(phase);end;
%phase = -180+180./pi*atan2(-imag(g),-real(g));

% If no left hand arguments then plot graph.
if no==0
if r>1, disp('Attention: système SIMO !!');end;
  for kk=1:r,
        [mag,ph]=r2p(g(:,kk));
        ph=fixphase(ph);
        plot(ph,20.*log10(mag)),grid,xlabel('PHASE'),ylabel('dB')
	if ~ishold
            yl=[min(mag) max(mag)];yl=20.*log10(yl);
 	    yl=[floor(yl(1)/10)*10 ceil(yl(2)/10)*10];
            if (yl(2)>100) yl(2)=100;end;
            if (yl(1)<-100) yl(1)=-100;end;
	    if (yl(2)-yl(1))<20,yl(1)=yl(2)-20;end;
            set(gca,'YLim',yl);
            xl=[min(ph) max(ph)];
 	    xl=[floor(xl(1)/180)*180 ceil(xl(2)/180)*180];
	    if (xl(2)-xl(1))<360,xl(1)=xl(2)-360;end;
	    set(gca,'XLim',xl);
%           set(gca,'YLim',[-100 100]);
        end;
	hold on
%	xl=get(gca,'XLim');
%	xl=[floor(xl(1)/180)*180 ceil(xl(2)/180)*180];
%	if (xl(2)-xl(1))<360,xl(1)=xl(2)-360;end;
%%	xl(1)=min([-360 xl(1)]);
%%	xl(2)=max([0 xl(2)]);
%	set(gca,'XLim',xl);
%  Gratuation en x multiples de 60 degres
	vec=get(gca,'XTick');
	ntick=length(vec);
        xl=get(gca,'XLim');
        pas=round(xl(2)-xl(1))/(ntick-1);
        pasi=[10;30;45;60;90;120;180;360];
        [err,ii]=min(abs(pasi-pas));
        pas=pasi(ii);
%	pas=round((xl(2)-xl(1))/ntick/60)*60;
%	if pas>180,pas=180;end;
	vec=[xl(1):pas:xl(2)];
	set(gca,'XTick',vec);
%
%	nmax=floor(xl(2)/360);nmin=ceil(xl(1)/360);
	nmax=ceil(xl(2)/360);nmin=floor(xl(1)/360);
	[mc]=mcirc([0 2.3 10]);
	[m,p]=r2p(-mc);
        clm=get(gca,'ColorOrder');
        set(gca,'ColorOrder',[0.5 0.5 0.5]);
	for ii=nmin:nmax+1
		plot(p+ii*360-180,20*log10(m));
	end
        set(gca,'ColorOrder',clm);
%	mknic(g,w)
	plt=4;
	mag=20 .*log10(mag);
	mon_mkplot(ph,mag,w,plt);
	hold off
   end
end

%  Internal function FIXPHASE
function pfix = fixphase(praw)
%FIXPHASE Unwrap phase plots so that instead of going from -180 to 180
%         they will have a smooth transition across these branch cuts
%         into phases greater than 180 or less than -180.
[m,n] = size(praw);
row=0;
if m == 1  % row vector
        row=1;
        praw = praw.';
        [m,n] = size(praw);
end
pfix = praw;
for j = 1:n
        p = praw(:,j);
        pd = diff(p);
        ijump = find(abs(pd) > 170);
        for i=1:length(ijump)
                k = ijump(i);
                p(k+1:m) = p(k+1:m) - 360 * sign(pd(k));
        end
        pfix(:,j) = p;
end
if row  % restor shape
  pfix=pfix.';
end

% Intrenal function R2P:
function [mag,ph]=r2p(r,im)
%R2P    Converts rectangular to polar co-ordinates.
%       [MAG,PH]=R2P(C) converts complex vector or
%       matrix C to magnitude and phase in degrees.

i=sqrt(-1);
if nargin==2
   r=r+i*im;
end
mag=abs(r);
ph=imag(log(r)).*(180/pi);

% Internal Function MON_MKPLOT:
function mon_mkplot(x,y,w,plt)
%MON_MKPLOT	Mark points along a plot.
%	MON_MKPLOT(X,Y,W,plt) marks the points (x,y) with
%	the numbers in the vector W
%
%	The text spacing has been selected for Epson and HPGL plotters.
%	Change if necessary for your particular output device.

dx4=0.04;   % These are the ratios for one character to the plot width
dy4=0.09;   %   for 4 plots per page change if necessary
dx1=0.018;   % These are the ratios for one character to the plot width
dy1=0.035;   %   for 1 plot per page change if necessary

nargs=nargin;
error(nargchk(2,4,nargs));
if nargs==2
   w=y;
   y=imag(x);
   x=real(x);
   plt=4;
elseif nargs==3
      if (any(any(imag(x))))
        plt=w;
        w=y;
        y=imag(x);
        x=real(x);
      else
        plt=4;
      end
 % else nargs==4   use the inputs as is
end
if max(size(plt))~=1
   error('Plots/page not a scalar.');
end
if plt~=1
   strdx=dx4;
   strdy=dy4;
else
   strdx=dx1;
   strdy=dy1;
end

hold on;
[mx,nx]=size(x);
if min(mx,nx)==1   % x a vector
   x=x(:);
   y=y(:);
   nx=1;
   mx=length(x);
end
[my,ny]=size(y);
[mw,nw]=size(w);
if (ny~=nx)|(my~=mx)|(min(mw,nw)~=1)|(max(mw,nw)~=mx)
   error('Dimensions of inputs not consistent')
end
str=vec2str(w);
[mstr,nstr]=size(str);

for j=1:nx
  axis;   % auto range axis
  strj=str;
  xj=x(:,j);
  yj=y(:,j);
  wax  = axis;  % Fix axis World axis coordinates
           %  Now eliminate points outside the borders
  i=clip(xj,yj,wax);
  xj=xj(i);
  yj=yj(i);
  strj=strj(i,:);
     % now try to space out the text
     % text area = strdx*nstr*abs((wax(2)-wax(1))),
     %             strdy*abs((wax(4)-wax(3))
  dx=strdx*nstr*abs((wax(2)-wax(1)));
  dy=strdy*abs((wax(4)-wax(3)));
  if length(xj)~=0   % have some points left
     lastx=xj(1);
     lasty=yj(1);
     lxj=length(xj);
     in=zeros(1,lxj);
     in(1)=1;       % always have first point
     for i=2:lxj
         if ((abs(xj(i)-lastx))<dx)&((abs(yj(i)-lasty))<dy)
            % skip the point
            in(i)=0;
         else
            in(i)=1;
            lastx=xj(i);
            lasty=yj(i);
         end
     end   % for i=1:lxj
     xj=xj(logical(in));
     yj=yj(logical(in));
     strj=strj(logical(in),:);
%     text(xj,yj,strj,'Clipping','on');
     for k=1:length(xj),
     text('Units','data',...
	  'Position',[xj(k) yj(k)],...
	     'HorizontalAlignment','left',...
	     'Units','points',...
	     'FontSize',8,...
 	     'Clipping','on',...
 	     'String',strj(k,:),...
             'Units','data');
     end;
     plot(xj,yj,'x');
  else
     disp('No points inside the current plot''s borders.')
  end % if length(xj)~=0
end %  for j = 1:lxj
hold off

%Internal function MCIRC
function c=mcirc(m)

if min(size(m))~=1
   error('M must be a vector')
end
m=10 .^(m/20);
angone=[(1:1:16)./8 2.5 (7:5:22)./2 20 30 60 100];
ly=length(angone);
ir=ly:-1:1;
minusang=-(angone(ir));
angone=[minusang,0,angone].';

angplus=[1599 (+1500:-100:+500) (+400:-40:+80) (+40:-10:-30)...
          (-40:-40:-360) (-400:-100:-1500) -1599].';
angplus=angplus.*(pi/1600);
angminus=angplus(51:-1:1);

lm=length(m);
lang=length(angone);
c=zeros(lang,lm);
for i=1:lm
 if (m(i)<1+2*eps)&(m(i)>1-2*eps)
    %   calculate M=1 lines
    c(:,i)=-0.5+sqrt(-1)*angone;
 elseif m(i)>1
   c(:,i)=(-m(i)^2/(m(i)^2-1))+(m(i)/(m(i)^2-1)).*exp(sqrt(-1).*angminus);
 else  % m(i)<1
   c(:,i)=(-m(i)^2/(m(i)^2-1))+(m(i)/(m(i)^2-1)).*exp(sqrt(-1).*angplus);
 end;
end;


% Internal function vec2str
function str=vec2str(v,form)
nargs=nargin;
error(nargchk(1,2,nargs));
if nargs==1
   form=' %.4g';
end
blank=' ';
str=sprintf(form,v(1));
lstr=length(str);
for i=2:length(v)
    st=sprintf(form,v(i));
    lst=length(st);
    if lst>lstr
       [m,n]=size(str);
       str=[str,setstr(blank.*ones(m,lst-n))];
       lstr=lst;
    else
       st=[st,setstr(blank.*ones(1,lstr-lst))];
    end
    str=[str;st];
end  % for i=1:length(v)

% internal function CLIP.M
function in=clip(x,y,area)
[m,n]=size(area);
if (min(m,n)~=1)|((m*n)~=4)
   error('The AREA vector has the wrong dimensions')
end
ix=(x>=area(1))&(x<=area(2));
iy=(y>=area(3))&(y<=area(4));
in=ix(:)&iy(:);

