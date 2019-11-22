function []=portrait()
% Met la figure courante au format A4 et en mode portrait

% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.

  Ox =  2.0 ;
  Oy =  2.0 ; 
  Px = 29.7 ; 
  Py = 21.0 ;
  Lx=Px-2*Ox;
  Ly=Py-2*Oy;

%  
%   configuration impression
%

  PaperOrientation = 'portrait' ;
  PaperUnits = 'centimeters';
  PaperPosition = [ Ox Oy Ly Lx ] ;
  PaperType = 'a4letter';

%
%   configuration figure
%

% Position = [ Ox+13 Oy-2 Ly Lx-2 ] ; % Pour LINUX, UNIX
  Position = [ Ox-1.5 Oy-0.5 0.7*Ly 0.7*Lx-2 ] ; % Pour WINDOWS
  Units=' centimeters';

sauve=get(gcf,'Units');
sauve1=get(gcf,'PaperUnits');

set(gcf,        'PaperType', PaperType, ...
		'PaperOrientation', PaperOrientation, ...
		'PaperUnits',PaperUnits, ...
                'PaperPosition', PaperPosition, ...
                'Units', 'centimeters', ...
		'Position',Position, ...
                'Units', sauve, ...
		'PaperUnits',sauve1 ...
)  ;





