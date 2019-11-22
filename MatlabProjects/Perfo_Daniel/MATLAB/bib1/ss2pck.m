function [pack]=ss2pck(sys)
% SYSO=SS2PCK(SYSI) permet de convertir un système SYSI
%  représenté par un objet LTI en une matrice système SYSO
%  (créée par PCK).
%
% Voir aussi: PCK2SS, PCK et UNPCK.


% S. Delannoy 10/98
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

[ass,bss,css,dss]=ssdata(sys);
pack=pck(ass,bss,css,dss);
