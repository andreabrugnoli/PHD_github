function [out] = beamshapeTyRz(in,l)
% beamshapeTyRz(IN,L) plots the bending (in the plane 0xy) mode 
%   shape of the beam associated to the vector (6 components) of 
%   the kinematic variables (Qtilde) IN (column vector). 
%   L is the length of the beam.
%
% U=beamshapeTyRz(IN,L): U is the vector of the deformations of 101
% points regularly spaced on the beam.

x=[0:0.01:1]*l;
f1=1-10*x.^3/l^3+15*x.^4/l^4-6*x.^5/l^5;
f2=x-6*x.^3/l^2+8*x.^4/l^3-3*x.^5/l^4;
f3=0.5*x.^2-3/2*x.^3/l+3/2*x.^4/l^2-0.5*x.^5/l^3;
f4=10*x.^3/l^3-15*x.^4/l^4+6*x.^5/l^5;
f5=-4*x.^3/l^2+7*x.^4/l^3-3*x.^5/l^4;
f6=0.5*x.^3/l-x.^4/l^2+0.5*x.^5/l^3;
P=[1 0 0 0 0 0;
   0 1 0 0 0 0;
   0 0 1 0 0 0;
   1 l 0 1 0 0;
   0 1 0 0 1 0;
   0 0 0 0 0 1];
out=[f1' f2' f3' f4' f5' f6']*P*in;
if nargout==0,
    figure
    plot([0:100],out);
    xlabel('% of length');
    ylabel('Deflection');
end
end

