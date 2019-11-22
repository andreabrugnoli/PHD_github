function [a,b,c,d,flag,t]=test_natss(sys)
% [A,B,C,D,FLAG] = TEST_NATSS(SYS) 
% teste la nature (object LTI ou Matrice systeme) du systeme SYS
% et renvoie les matrices A,B,C,D correspondantes .
% FLAG prend une valeur significative de la nature de SYS:
%		FLAG=1 pour un object LTI,
%		FLAG=2 pour une matrice système (PCK),
% 		FLAG=0 si SYS n'est pas un systeme.
% 
% Autre syntaxe:
% [A,B,C,D,FLAG,T] = TEST_NATSS(SYS) renvoie  la periode
% d'échantillonnage T si SYS est un objet LTI discret (FLAG=1)
% (T=0 en continu).
%
% Voir aussi: LTI2PCK, PCK2SS.
%

% S. Delonnoy 01/98
% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

flag=0;a=0;b=0;c=0;d=0;t=0;
if isa(sys,'lti'), [a,b,c,d,t]=ssdata(sys);flag=1;         % le systeme est un SS
elseif isa(sys,'double'), [a,b,c,d]=unpck(sys);flag=2;    % le systeme est un pack
     if isempty(a), flag=0;end;
end;
