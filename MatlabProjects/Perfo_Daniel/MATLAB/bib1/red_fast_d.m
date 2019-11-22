function [sysout]=red_fast_d(sysin,w)
% [SYS_OUT]=RED_FAST_D(SYS_IN,W)
%  enlève dans le système SYS_IN (object LTI) les modes stables
%  et rapides dont la partie réelle est infèrieure à W
%  et conserve la transmission directe de SYS_IN.
%
%  Représentation du système réduit système LTI (ss) SYS_OUT.
%     

% D. Alazard 01/95
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.
% Revised 03/2011

sys=canon(sysin,'modal');

[aa,bb,cc,dd]=ssdata(sys);
n=size(aa,1);
as=[];bs=[];cs=[];
gain=dcgain(sysin);
j=1;
i=1;
while i<=n-1,
    if aa(i,i+1)==0,  % real eigenvalue
        if real(aa(i,i))>w,
            as(j,j)=aa(i,i);
            bs(j,:)=bb(i,:);
            cs(:,j)=cc(:,i);
            j=j+1;
        end;
        i=i+1;
    else % complex eigenvalue
        if real(aa(i,i))>w,
            as(j:j+1,j:j+1)=aa(i:i+1,i:i+1);
            bs(j:j+1,:)=bb(i:i+1,:);
            cs(:,j:j+1)=cc(:,i:i+1);
            j=j+2;
        end;
        i=i+2;      
    end
end;
if i==n,  % real eigenvalue
    if real(aa(i,i))>w,
        as(j,j)=aa(i,i);
        bs(j,:)=bb(i,:);
        cs(:,j)=cc(:,i);
        j=j+1;
    end;
end
sysout=ss(as,bs,cs,dd);

